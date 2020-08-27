# Cross-view Semantic Segmentation for Sensing Surroundings

We release the code of the View Parsing Networks, the main model for Cross-view Semantic Segmentation task.


**[Cross-View Semantic Segmentation for Sensing Surroundings
](https://arxiv.org/pdf/1906.03560.pdf)**
<br />
[Bowen Pan](http://people.csail.mit.edu/bpan/), 
[Jiankai Sun](), 
[Ho Yin Tiga Leung](),
[Alex Andonian](https://www.alexandonian.com/), and
[Bolei Zhou](http://bzhou.ie.cuhk.edu.hk/)
<br />
IEEE Robotics and Automation Letters
<br />
In IEEE International Conference on Intelligent Robots and Systems (IROS) 2020
<br />
[[Paper]](https://arxiv.org/pdf/1906.03560.pdf)
[[Project Page]](https://decisionforce.github.io/VPN/)

```
@ARTICLE{pan2019crossview,
     author={B. {Pan} and J. {Sun} and H. Y. T. {Leung} and A. {Andonian} and B. {Zhou}},
     journal={IEEE Robotics and Automation Letters},
     title={Cross-View Semantic Segmentation for Sensing Surroundings},
     year={2020},
     volume={5},
     number={3},
     pages={4867-4873},
}
```


### Requirement
- Install the [House3D](https://github.com/facebookresearch/House3D) simulator, or [Gibson](http://gibsonenv.stanford.edu) simulator.
- Software: Ubuntu 16.04.3 LTS, CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0

## Train and test VPN

### Data processing (use House3D for example)
- Use [get_training_data_from_house3d.py](https://github.com/pbw-Berwin/View-Parsing-Network/blob/master/tools/get_trainning_data_from_house3d.py) to extract data from House3D environment.
- Use [process_data_for_VPN.py](https://github.com/pbw-Berwin/View-Parsing-Network/blob/master/tools/process_data_for_VPN.py) to split training set and validation set, and visualize data.

### Training Command
```
# Training in indoor-room scenarios, using RGB input modality, with 8 input views.
python -u train.py --fc-dim 256 --use-depth false --use-mask false --transform-type fc --input-resolution 400 --label-res 25 --store-name [STORE_NAME] --n-views 8 --batch-size 48 -j 10 --data_root [PATH_TO_DATASET_ROOT] --train-list [PATH_TO_TRAIN_LIST] --eval-list [PATH_TO_EVAL_LIST]

# Training in driving-traffic scenarios, using RGB input modality, with 6 input views.
python -u train_carla.py --fc-dim 256 --use-depth false --use-mask false --transform-type fc --input-resolution 400 --label-res 25 --store-name [STORE_NAME] --n-views 6 --batch-size 48 -j 10 --data_root [PATH_TO_DATASET_ROOT] --train-list [PATH_TO_TRAIN_LIST] --eval-list [PATH_TO_EVAL_LIST]
```

### Testing Command
```
# Training in indoor-room scenarios, using RGB input modality, with 8 input views.
python -u test.py --fc-dim 256 --use-depth false --use-mask false --transform-type fc --input-resolution 400 --label-res 25 --store-name [STORE_NAME] --n-views 8 --batch-size 4 --test-views 8 --data_root [PATH_TO_DATASET_ROOT] --eval-list [PATH_TO_EVAL_LIST] --num-class [NUM_CLASS] -j 10 --weights [PATH_TO_PRETRAIN_MODEL]

# Testing in driving-traffic scenarios, using RGB input modality, with 6 input views.
python -u test_carla.py --fc-dim 256 --use-depth false --use-mask false --transform-type fc --input-resolution 400 --label-res 25 --store-name [STORE_NAME] --n-views 6 --batch-size 4 --test-views 6 --data_root [PATH_TO_DATASET_ROOT] --eval-list [PATH_TO_EVAL_LIST] --num-class [NUM_CLASS] -j 10 --weights [PATH_TO_PRETRAIN_MODEL]
```

## Transfer learning for sim-to-real adaptation

### Data processing (use indoor-room scenarios for example)
- Use [process_transfer_indoor_data.py](https://github.com/pbw-Berwin/View-Parsing-Network/blob/master/tools/process_transfer_indoor_data.py) to split source domain set and target domain set.

### Training Command
```
# Training in indoor-room scenarios, using RGB input modality, with 8 input views.
python -u train_transfer.py --task-id [TASK_NAME] --num-class [NUM_CLASS] --learning-rate-D 3e-6 --iter-size-G 1 --iter-size-D 1 --snapshot-dir ./snapshot --batch-size 20 --tensorboard true --n-views 6 --train_source_list [PATH_TO_TRAIN_LIST] --train_target_list [PATH_TO_EVAL_LIST] --VPN-weights [PATH_TO_PRETRAINED_WEIGHT] --scenarios indoor
```
