mkdir -p log
# driving scene
# python train_transfer_driving.py -u --batch-size 15 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt --VPN-weights ./pretrained/VPN_pretrain_RGB.pth.tar
srun -p Segmentation --gres=gpu:1 -n1 --job-name=bw_vpn python -u train_transfer_driving.py --task-id 3-1 --iter-size-G 3 --iter-size-D 1 --batch-size 10 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt 2>&1 > log/vpn.log &
# srun -p Segmentation --gres=gpu:1 -n1 --job-name=MapNet python -u train_transfer_driving.py --batch-size 15 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt 2>&1 > log/vpn.log &
# srun -p Segmentation --gres=gpu:1 -n1 --job-name=bw_vpn python -u train_transfer_driving.py --task-id 2-1 --iter-size-G 2 --iter-size-D 1 --batch-size 10 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt 2>&1 > log/vpn.log &
# srun -p Segmentation --gres=gpu:1 -n1 --job-name=bw_vpn python -u train_transfer_driving.py --task-id 1-1 --iter-size-G 1 --iter-size-D 1 --batch-size 10 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt 2>&1 > log/vpn.log &
# srun -p Segmentation --gres=gpu:1 -n1 --job-name=bw_vpn python -u train_transfer_driving.py --task-id pretrained_1-1 --iter-size-G 1 --iter-size-D 1 --batch-size 10 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt --VPN-weights ./pretrained/VPN_pretrain_RGB.pth.tar 2>&1 > log/vpn.log &
# srun -p Segmentation --gres=gpu:1 -n1 --job-name=bw_vpn python -u train_transfer_driving.py --task-id lr1e-5_1-1 --learning-rate-D 1e-5 --iter-size-G 1 --iter-size-D 1 --batch-size 10 --tensorboard true --n-views 6 --train_source_list tools/train_source_list.txt --train_target_list tools/train_target_list.txt --VPN-weights ./pretrained/VPN_pretrain_RGB.pth.tar 2>&1 > log/vpn.log &
