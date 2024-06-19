# %%
# import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import warnings
from torch_geometric.loader import DataLoader
from datetime import datetime

warnings.filterwarnings('ignore')

from load_data import *
# from models import *
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from fairinv import *
import json


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hid_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'german'])
    parser.add_argument("--layer_num", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn', 'sage', 'gin'])
    parser.add_argument('--aggr', type=str, default='add',
                        choices=['add', 'mean', 'max', 'min', 'sum', 'std', 'var', 'median'],
                        help="aggregation function")
    parser.add_argument('--weight_path', type=str, default='./weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='hyperpapameter to balance the downstream task and invariance learning loss.')
    parser.add_argument('--lr_sp', type=float, default=0.5, help='the learning rate of the sensitive partition.')
    parser.add_argument('--env_num', type=int, default=2,
                        help='the number of the sensitive attribute, also known as environment number.')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--partition_times', type=int, default=3,
                        help='the number for partitioning the sensitive attribute group.')

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model = 'FairINV'

    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    data = FairDataset(args.dataset, args.device)
    data.load_data()

    num_class = 1
    args.in_dim = data.features.shape[1]
    args.nnode = data.features.shape[0]
    args.out_dim = num_class

    """
    Build model, optimizer, and loss fuction
    """
    # FairINV
    fairinv = FairINV(args)

    """
    Train model
    """
    fairinv.train_model(data, pbar=args.pbar)

    """
    evaluation
    """
    fairinv.load_state_dict(torch.load(f'./weights/FairINV_{args.encoder}.pt'))
    fairinv.eval()
    with torch.no_grad():
        output = fairinv(data.features, data.edge_index)

    pred = (output.squeeze() > 0).type_as(data.labels)
    # utility performance
    auc_test = roc_auc_score(data.labels[data.idx_test].cpu(), output[data.idx_test].cpu())
    f1_test = f1_score(data.labels[data.idx_test].cpu(), pred[data.idx_test].cpu())
    acc_test = accuracy_score(data.labels[data.idx_test].cpu(), pred[data.idx_test].cpu())
    # fairness performance
    parity_test, equality_test = fair_metric(pred[data.idx_test].cpu().numpy(),
                                             data.labels[data.idx_test].cpu().numpy(),
                                             data.sens[data.idx_test].cpu().numpy())

    return auc_test, f1_test, acc_test, parity_test, equality_test


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')

    model_num = 1
    results = Results(args.seed_num, model_num, args)

    dir_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + f'_{args.dataset}'
    args.log_dir = os.path.join(args.log_dir, dir_name)

    for seed in range(args.seed_num):
        args.seed_dir = os.path.join(args.log_dir, f"seed{seed}")
        # set seeds
        args.pbar = tqdm(total=args.epochs, desc=f"Seed {seed + 1}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        set_seed(seed)

        # running train
        results.auc[seed, :], results.f1[seed, :], results.acc[seed, :], results.parity[seed, :], \
            results.equality[seed, :] = run(args)

    # reporting results
    results.report_results()
    if args.save_results:
        results.save_results(args)
