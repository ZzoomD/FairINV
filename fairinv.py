import time

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import grad
from torch.autograd import Function
import torch.distributed as dist
import os
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GraphConv, global_mean_pool
from torch.optim.lr_scheduler import MultiStepLR
import math
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import fair_metric


class LogWriter:
    def __init__(self, logdir='./logs'):
        self.writer = SummaryWriter(logdir)

    def record(self, loss_item: dict, step: int):
        for key, value in loss_item.items():
            self.writer.add_scalar(key, value, step)


class FairINV(nn.Module):
    def __init__(self, args):
        super(FairINV, self).__init__()
        self.in_dim = args.in_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.out_dim
        self.args = args
        gnn_backbone = ConstructModel(args.in_dim, args.hid_dim, args.encoder, args.layer_num)
        self.gnn_backbone = gnn_backbone.to(args.device)
        self.classifier = nn.Linear(args.hid_dim, args.out_dim).to(args.device)
        self.optimizer_infer = torch.optim.Adam(list(self.gnn_backbone.parameters())+list(self.classifier.parameters()),
                                                lr=args.lr, weight_decay=args.weight_decay)
        
        for m in self.modules():
            self.weights_init(m)

        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_irm = IRMLoss()
        self.criterion_env = nn.BCEWithLogitsLoss(reduction='none')
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def train_model(self, data, **kwargs):
        best_loss = 100
        best_result = 0
        pbar = kwargs.get('pbar', None)
        part_mat_list, edge_weight_inv_list = [], []

        if hasattr(self.args, 'seed_dir'):
            writer = LogWriter(self.args.seed_dir)
        
        for i in range(self.args.partition_times):
            part_mat, edge_weight_inv = self.sens_partition(data, writer)
            part_mat_list.append(part_mat[data.idx_train])
            edge_weight_inv_list.append(edge_weight_inv)

        for epoch in range(self.args.epochs):
            loss_log_list = []
            loss_cls_all, loss_irm_all = 0, 0

            self.gnn_backbone.train()
            self.classifier.train()

            self.optimizer_infer.zero_grad()

            for i, edge_weight_inv in enumerate(edge_weight_inv_list):
                edge_weight_copy = data.edge_index.clone()
                edge_weight_inv = edge_weight_inv.squeeze()
                edge_weight_copy = edge_weight_copy.fill_value(1., dtype=None)
                edge_weight_copy.storage.set_value_(edge_weight_copy.storage.value() * edge_weight_inv.to(edge_weight_copy.device()))
                emb = self.gnn_backbone(data.features, edge_weight_copy)
                output = self.classifier(emb)

                loss_cls = self.criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
                
                group_assign = part_mat_list[i].argmax(dim=1)
                for j in range(part_mat_list[i].shape[-1]):
                    # split groups
                    select_idx = torch.where(group_assign == j)[0]
                    if len(select_idx) == 0:
                        continue

                    # filter subsets
                    sub_logits = output[data.idx_train][select_idx]
                    sub_labels = data.labels[data.idx_train][select_idx]

                    loss_log = self.criterion_cls(sub_logits, sub_labels.unsqueeze(1).float())
                    loss_log_list.append(loss_log.view(-1))

            loss_log_cat = torch.cat(loss_log_list, dim=0)
            Var, Mean = torch.var_mean(loss_log_cat)
            loss_train = Var + self.args.alpha * Mean
            loss_train.backward()
            self.optimizer_infer.step()

            self.gnn_backbone.eval()
            self.classifier.eval()
            with torch.no_grad():
                emb_val = self.gnn_backbone(data.features, data.edge_index)
                output_val = self.classifier(emb_val)

            loss_cls_val = self.criterion_cls(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
            pred = (output_val.squeeze() > 0).type_as(data.labels)
            # utility performance
            auc_val = roc_auc_score(data.labels[data.idx_val].cpu(), output_val[data.idx_val].cpu())
            f1_val = f1_score(data.labels[data.idx_val].cpu(), pred[data.idx_val].cpu())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu(), pred[data.idx_val].cpu())
            # fairness performance
            parity_val, equality_val = fair_metric(pred[data.idx_val].cpu().numpy(),
                                                   data.labels[data.idx_val].cpu().numpy(),
                                                   data.sens[data.idx_val].cpu().numpy())
            
            if self.args.dataset in ['pokec_z', 'pokec_n']:
                if loss_cls_val.item() < best_loss:
                    best_loss = loss_cls_val.item()
                    torch.save(self.state_dict(), f'./weights/FairINV_{self.args.encoder}.pt')
            else:
                if auc_val-parity_val-equality_val > best_result:
                    best_result = auc_val-parity_val-equality_val
                    torch.save(self.state_dict(), f'./weights/FairINV_{self.args.encoder}.pt')

            if 'writer' in locals():
                # log training set loss
                writer.record(loss_item={'train/loss_cls': loss_cls_all, 'train/loss_irm': loss_irm_all,
                                         'train/loss_all': loss_train}, step=epoch)
                # log validation set performance
                writer.record(loss_item={'val/auc': auc_val, 'val/f1': f1_val, 'val/acc': acc_val,
                                         'val/dp': parity_val, 'val/eo': equality_val}, step=epoch)

            if pbar is not None:
                pbar.set_postfix({'loss_train': "{:.2f}".format(loss_train.item())})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    def sens_partition(self, data, writer):
        ref_backbone, ref_classifier = self.train_ref_model(data, writer)

        partition_module = SAP(in_dim=self.in_dim, hid_dim=self.hid_dim, out_dim=self.args.env_num, encoder=self.args.encoder, layer_num=self.args.layer_num).to(self.args.device)
        optimizer_part_mat = torch.optim.Adam(list(partition_module.parameters()), lr=self.args.lr_sp, weight_decay=1e-5)

        emb = ref_backbone(data.features, data.edge_index)
        logits = ref_classifier(emb)
        scale = torch.tensor(1.).cuda().requires_grad_()
        error = self.criterion_env(logits[data.idx_train]*scale,
                                   data.labels[data.idx_train].unsqueeze(1).float())
        
        emb_cat = torch.cat([emb[data.edge_index.storage._row], emb[data.edge_index.storage._col]], dim=1)

        for epoch in range(500):
            loss_penalty_list = []
            partition_module.train()
            optimizer_part_mat.zero_grad()

            part_mat, edge_weight_inv = partition_module(emb_cat.detach(), data.features, data.edge_index, data.labels)
            for env_idx in range(self.args.env_num):
                loss_weight = part_mat[:, env_idx]
                penalty_grad = grad((error.squeeze(1) * loss_weight[data.idx_train]).mean(), [scale], create_graph=True)[0].pow(2).mean()
                loss_penalty_list.append(penalty_grad)

            risk_final = -torch.stack(loss_penalty_list).sum()
            risk_final.backward(retain_graph=True)
            optimizer_part_mat.step()
                
            if 'writer' in locals():
                # log training set loss
                writer.record(loss_item={'sens_part/risk_final': risk_final}, step=epoch)

        with torch.no_grad():
            soft_split_final, edge_weight_inv = partition_module(emb_cat.detach(), data.features, data.edge_index, data.labels)
        return soft_split_final, edge_weight_inv

    def train_ref_model(self, data, writer=None):

        ref_backbone = ConstructModel(self.args.in_dim, self.args.hid_dim, self.args.encoder,
                                       self.args.layer_num).to(self.args.device)
        ref_classifier = nn.Linear(self.args.hid_dim, self.args.out_dim).to(self.args.device)
        optimizer_part = torch.optim.Adam(list(ref_backbone.parameters()) + list(ref_classifier.parameters()),
                                          lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch in range(500):
            ref_backbone.train()
            ref_classifier.train()
            optimizer_part.zero_grad()

            emb = ref_backbone(data.features, data.edge_index)
            output = ref_classifier(emb)
            loss_train = self.criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())

            loss_train.backward()
            optimizer_part.step()
            if 'writer' in locals():
                # log training set loss
                writer.record(loss_item={'pre-train/cls_loss': loss_train.item()}, step=epoch)

        return ref_backbone, ref_classifier

    def forward(self, x, edge_index):
        emb = self.gnn_backbone(x, edge_index)
        output = self.classifier(emb)
        return output

class IRMLoss(_Loss):
    def __init__(self):
        super(IRMLoss, self).__init__()
        # https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
        """
        compute gradients based IRM penalty with DDP fc setting
        :param logits: local logits [bs, C]
        :param labels: local labels [bs]
        :param wsize: world size
        :param cfg: config
        :param updated_split_all: list of all partition matrix
        :return:
        avg IRM penalty of all partitions
        """
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, updated_split_all):
        env_penalty = []
        # assert isinstance(updated_split_all, list), 'retain all previous partitions'
        for updated_split_each in updated_split_all:
            per_penalty = []
            env_num = updated_split_each.shape[-1]
            group_assign = updated_split_each.argmax(dim=1)

            for env in range(env_num):
                # split groups
                select_idx = torch.where(group_assign == env)[0]

                # filter subsets
                sub_logits = logits[select_idx]
                sub_label = labels[select_idx]

                # compute penalty
                scale_dummy = torch.tensor(1.).cuda().requires_grad_()
                loss_env = self.loss(sub_logits * scale_dummy, sub_label)
                loss_grad = grad(loss_env, [scale_dummy], create_graph=True)[0]
                per_penalty.append(torch.sum(loss_grad ** 2))

            env_penalty.append(torch.stack(per_penalty).mean())

        loss_penalty = torch.stack(env_penalty).mean()

        return loss_penalty

class ConstructModel(nn.Module):
    def __init__(self, in_dim, hid_dim, encoder, layer_num):
        super(ConstructModel, self).__init__()
        self.encoder = encoder
        
        if encoder == 'gcn':
            self.model = nn.ModuleList()
            for i in range(layer_num-1):
                if i == 0:
                    self.model.append(GCNConv(in_dim, hid_dim))
                else:
                    self.model.append(GCNConv(hid_dim, hid_dim))
        elif encoder == 'gin':
            self.model = GIN(nfeat=in_dim, nhid=hid_dim, dropout=0.5)
        elif encoder == 'sage':
            self.model = SAGE(nfeat=in_dim, nhid=hid_dim, dropout=0.5)
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None):
        if self.encoder == 'gcn':
            h = x
            for i, layer in enumerate(self.model):
                h = layer(h, edge_index, edge_weight=edge_weight)
        elif self.encoder == 'gin':
            h = self.model(x, edge_index)
        elif self.encoder == 'sage':
            h = self.model(x, edge_index)
        return h


class SAP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, encoder, layer_num):
        super(SAP, self).__init__()
        self.variant_infer = nn.Sequential(
            nn.Linear(in_features=2*hid_dim, out_features=1),
            nn.Sigmoid()
        )

        self.sens_infer_backbone = ConstructModel(in_dim, hid_dim, encoder=encoder, layer_num=layer_num)
        self.sens_infer_classifier = nn.Sequential(
            nn.Linear(in_features=hid_dim+1, out_features=out_dim),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, emb_cat, features, edge_index, labels):
        edge_weight_variant = self.variant_infer(emb_cat)
        edge_weight_variant = edge_weight_variant.squeeze()
        edge_index = edge_index.fill_value(1., dtype=None)
        edge_index.storage.set_value_(edge_index.storage.value() * edge_weight_variant.to(edge_index.device()))
        h = self.sens_infer_backbone(features, edge_index)
        sens_attr_partition = self.sens_infer_classifier(torch.cat([h, labels.unsqueeze(1)], dim=1))
        return sens_attr_partition, 1-edge_weight_variant


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout): 
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid), 
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid), 
        )
        self.conv1 = GINConv(self.mlp1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, edge_index): 
        x = self.conv1(x, edge_index)
        return x

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout): 
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index): 
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x