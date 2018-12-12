import cv2
import os
from dominate.tags import *
import dominate

dataset_root = '/data/vision/oliva/scenedataset/syntheticscene/TopViewMaskDataset/'
scene_list = os.listdir(dataset_root)
train_list = scene_list[:int(5*len(scene_list)/6)]
val_list = scene_list[int(5*len(scene_list)/6):]
train_output = []
val_output = []
data_type = ['rgb', 'sem', 'depth', 'ins', ]


for scene_id in train_list:
    print('Processing ' + scene_id)
    coor_set = os.listdir(os.path.join(dataset_root, scene_id))
    for coor in coor_set:
        coor_path = os.path.join(scene_id, coor)
        skip = False
        if not os.path.isfile(os.path.join(dataset_root, coor_path, 'OverviewMask.png')):
            skip = True
        if skip:
            continue
        train_output.append(coor_path)
        web_path = os.path.join(dataset_root, coor_path)
        with dominate.document(title=web_path) as web:
            for mod in data_type:
                h2('Mode=%s, Yaw=%d'%(mod, 0))
                with table(border=1, style='table-layout: fixed;'):
                    with tr():
                        path = 'mode=%s_%d.png'%(mod, 0)
                        with td(style='word-wrap: break-word;', halign='center', valign='top'):
                            img(style='height:256px', src=path)
                h2('Mode=%s, Yaw=%d' % (mod, 45))
                with table(border=1, style='table-layout: fixed;'):
                    with tr():
                        path = 'mode=%s_%d.png' % (mod, 45)
                        with td(style='word-wrap: break-word;', halign='center', valign='top'):
                            img(style='height:256px', src=path)
            h2('Top-view Semantic Mask')
            with table(border=1, style='table-layout: fixed;'):
                with tr():
                    path = 'OverviewMask.png'
                    with td(style='word-wrap: break-word;', halign='center', valign='top'):
                        img(style='width:256px', src=path)
        with open(os.path.join(web_path, 'index.html'), 'w') as fp:
            fp.write(web.render())
            print('Index of ' + os.path.join(scene_id, coor) + ' generated!')

for scene_id in val_list:
    print('Processing ' + scene_id)
    coor_set = os.listdir(os.path.join(dataset_root, scene_id))
    for coor in coor_set:
        coor_path = os.path.join(scene_id, coor)
        skip = False
        if not os.path.isfile(os.path.join(dataset_root, coor_path, 'OverviewMask.png')):
            # print(os.path.join(coor_path, '0', 'topdown_view.jpg') + ' is lost')
            skip = True
        if skip:
            continue
        val_output.append(coor_path)
        web_path = os.path.join(dataset_root, coor_path)
        with dominate.document(title=web_path) as web:
            for mod in data_type:
                h2('Mode=%s, Yaw=%d'%(mod, 0))
                with table(border=1, style='table-layout: fixed;'):
                    with tr():
                        path = 'mode=%s_%d.png'%(mod, 0)
                        with td(style='word-wrap: break-word;', halign='center', valign='top'):
                            img(style='height:256px', src=path)
                h2('Mode=%s, Yaw=%d' % (mod, 45))
                with table(border=1, style='table-layout: fixed;'):
                    with tr():
                        path = 'mode=%s_%d.png' % (mod, 45)
                        with td(style='word-wrap: break-word;', halign='center', valign='top'):
                            img(style='height:256px', src=path)
            h2('Top-view Semantic Mask')
            with table(border=1, style='table-layout: fixed;'):
                with tr():
                    path = 'OverviewMask.png'
                    with td(style='word-wrap: break-word;', halign='center', valign='top'):
                        img(style='width:256px', src=path)
        with open(os.path.join(web_path, 'index.html'), 'w') as fp:
            fp.write(web.render())
            print('Index of ' + os.path.join(scene_id, coor) + ' generated!')


with open('train_list.txt','w') as f:
    print('Writing train file ...')
    f.write('\n'.join(train_output))

with open('val_list.txt', 'w') as f:
    print('Writing validation file ...')
    f.write('\n'.join(val_output))
