# driving scene
python train_transfer_driving.py --batch-size 15 --tensorboard true --n-views 6
# srun -p Segmentation python train_transfer_driving.py --batch-size 15 --tensorboard true --n-views 6 2>&1 > log/vpn.log &