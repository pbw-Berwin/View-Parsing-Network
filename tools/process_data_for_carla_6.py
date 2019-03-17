import cv2
import os

CARLA_data_root = '../data/Carla_Dataset_v1'
scene_list = sorted(os.listdir(CARLA_data_root))
train_list = scene_list[:int(5*len(scene_list)/6)]
val_list = scene_list[int(5*len(scene_list)/6):]
train_output = []
val_output = []
# data_type = ['rgb', 'sem', 'depth', 'ins', ]


for scene_id in train_list:
    print('Processing ' + scene_id)
    coor_set = sorted(os.listdir(os.path.join(CARLA_data_root, scene_id, 'CAM_RGB_FRONT')))

    for coor in coor_set:
        coor_path = os.path.join(scene_id, 'CAM_RGB_FRONT', coor)
        train_output.append(coor_path)

with open('../tools/train_source_list.txt', 'w') as f:
    print('Writing train_source_list file ...')
    f.write('\n'.join(train_output))

for scene_id in val_list:
    print('Processing ' + scene_id)
    coor_set = sorted(os.listdir(os.path.join(CARLA_data_root, scene_id, 'CAM_RGB_FRONT')))

    for coor in coor_set:
        coor_path = os.path.join(scene_id, 'CAM_RGB_FRONT', coor)
        val_output.append(coor_path)

with open('../tools/val_source_list.txt', 'w') as f:
    print('Writing val_source_list file ...')
    f.write('\n'.join(val_output))
