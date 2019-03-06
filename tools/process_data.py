import cv2
import os

CARLA_data_root = '../data/Carla_Dataset_v1'
dataset_root = '../data/nuScenes'
scene_list = sorted(os.listdir(CARLA_data_root))
train_list = scene_list[:int(5*len(scene_list)/6)]
# val_list = scene_list[int(5*len(scene_list)/6):]
train_output = []
# val_output = []
# data_type = ['rgb', 'sem', 'depth', 'ins', ]


for scene_id in train_list:
    print('Processing ' + scene_id)
    coor_set = sorted(os.listdir(os.path.join(CARLA_data_root, scene_id, 'CAM_RGB_FRONT')))

    for coor in coor_set:
        coor_path = os.path.join(scene_id, 'CAM_RGB_FRONT', coor)
        train_output.append(coor_path)

with open('train_source_list.txt', 'w') as f:
    print('Writing train_source_list file ...')
    f.write('\n'.join(train_output))

nuScenes_data_root = '../data/nuScenes'
scene_list = sorted(os.listdir(nuScenes_data_root))
train_list = scene_list
train_output = []

for scene_id in train_list:
    print('Processing ' + scene_id)
    coor_set = sorted(os.listdir(os.path.join(nuScenes_data_root, scene_id, 'CAM_RGB_FRONT')))

    for coor in coor_set:
        coor_path = os.path.join(scene_id, 'CAM_RGB_FRONT', coor)
        train_output.append(coor_path)


with open('train_target_list.txt', 'w') as f:
    print('Writing train_target_list file ...')
    f.write('\n'.join(train_output))
