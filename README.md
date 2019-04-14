# Cross-view Semantic Segmentation for Sensing Surroundings

We release the code of the View Parsing Networks.

### Requirement
- Install the [House3D](https://github.com/facebookresearch/House3D) simulator, or [Gibson](http://gibsonenv.stanford.edu) simulator.
- Software: Ubuntu 16.04.3 LTS, CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0

### Training Command
```
srun -p AD --gres=gpu:8 -n1 --job-name=carla_6 python -u train_carla.py --fc-dim 256 --use-depth false --use-mask false --transform-type fc --input-resolution 400 --label-res 25 --store-name carla_6 --n-views 6 --batch-size 48 -j 10 --data_root ./data/Carla_Dataset_v1/ --train-list tools/train_source_list.txt --eval-list tools/val_source_list.txt 2>&1 > log/carla_6.log &
```