# # best hyper-parameter for german dataset
echo '============German============='
CUDA_VISIBLE_DEVICES=0 python train_fairinv.py --hid_dim 16 --lr 1e-2 --epochs 1000 --encoder gcn --dataset german --seed_num 5 --alpha 10 --lr_sp 0.1

# # best hyper-parameter for bail dataset
# echo '============Bail============='
# CUDA_VISIBLE_DEVICES=0 python train_fairinv.py --hid_dim 16 --lr 1e-2 --epochs 1000 --encoder gcn --dataset bail --seed_num 5 --alpha 10 --lr_sp 0.1

# # best hyper-parameter for pokec_z dataset (best validation loss)
# echo '============Pokec_z============='
# CUDA_VISIBLE_DEVICES=0 python train_fairinv.py --hid_dim 16 --lr 1e-2 --epochs 1000 --encoder gcn --dataset pokec_z --seed_num 5 --alpha 10 --lr_sp 0.01

# # best hyper-parameter for pokec_n dataset (best validation loss)
# echo '============Pokec_n============='
# CUDA_VISIBLE_DEVICES=0 python train_fairinv.py --hid_dim 16 --lr 1e-2 --epochs 1000 --encoder gcn --dataset pokec_n --seed_num 5 --alpha 1 --lr_sp 0.5

# # best hyper-parameter for nba dataset
# echo '============nba============='
# CUDA_VISIBLE_DEVICES=0 python train_fairinv.py --hid_dim 16 --lr 1e-2 --epochs 1000 --encoder gcn --dataset nba --seed_num 5 --alpha 1 --lr_sp 0.1