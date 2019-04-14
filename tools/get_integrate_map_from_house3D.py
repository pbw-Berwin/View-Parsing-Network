import numpy as np
import random
import csv
import cv2
import pickle
import os

import gym
from gym import spaces
from House3D.house import House
from House3D.core import Environment, MultiHouseEnv
from House3D import objrender
from House3D.objrender import RenderMode
import sys

__all__ = ['CollectDataTask']

###############################################
# Task related definitions and configurations
###############################################
flag_print_debug_info = False  # flag for printing debug info

dist_reward_scale = 1.0  # 2.0
collision_penalty_reward = 0.3  # penalty for collision
correct_move_reward = None  # reward when moving closer to target
stay_room_reward = 0.1  # reward when staying in the correct target room
indicator_reward = 0.5
success_reward = 10  # reward when success

time_penalty_reward = 0.1  # penalty for each time step
delta_reward_coef = 0.5
speed_reward_coef = 1.0

success_distance_range = 1.0
success_stay_time_steps = 5
success_see_target_time_steps = 2  # time steps required for success under the "see" criteria

# sensitivity setting
rotation_sensitivity = 30  # 45   # maximum rotation per time step
default_move_sensitivity = 0.5  # 1.0   # maximum movement per time step

# discrete action space actions, totally <13> actions
# Fwd, L, R, LF, RF, Lrot, Rrot, Bck, s-Fwd, s-L, s-R, s-Lrot, s-Rrot
discrete_actions = [(1., 0., 0.), (0., 1., 0.), (0., -1., 0.), (0.5, 0.5, 0.), (0.5, -0.5, 0.),
                    (0., 0., 1.), (0., 0., -1.),
                    (-0.4, 0., 0.),
                    (0.4, 0., 0.), (0., 0.4, 0.), (0., -0.4, 0.),
                    (0., 0., 0.4), (0., 0., -0.4)]
n_discrete_actions = len(discrete_actions)

# criteria for seeing the object
n_pixel_for_object_see = 450  # need at least see 450 pixels for success under default resolution 120 x 90
n_pixel_for_object_sense = 50
L_pixel_reward_range = n_pixel_for_object_see - n_pixel_for_object_sense
pixel_object_reward = 0.4


#################
# Util Functions
#################

def reset_see_criteria(resolution):
    total_pixel = resolution[0] * resolution[1]
    global n_pixel_for_object_see, n_pixel_for_object_sense, L_pixel_reward_range
    n_pixel_for_object_see = max(int(total_pixel * 0.045), 5)
    n_pixel_for_object_sense = max(int(total_pixel * 0.005), 1)
    L_pixel_reward_range = n_pixel_for_object_see - n_pixel_for_object_sense

class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100: print('')


class CollectDataTask(gym.Env):
    def __init__(self, env,
                 seed=None,
                 reward_type='delta',
                 hardness=None,
                 move_sensitivity=None,
                 segment_input=True,
                 joint_visual_signal=False,
                 depth_signal=True,
                 max_steps=-1,
                 success_measure='see',
                 discrete_action=False):
        self.env = env
        assert isinstance(env, Environment), '[RoomNavTask] env must be an instance of Environment!'
        if env.resolution != (120, 90): reset_see_criteria(env.resolution)
        self.resolution = resolution = env.resolution
        assert reward_type in [None, 'none', 'linear', 'indicator', 'delta', 'speed']
        self.reward_type = reward_type
        self.colorDataFile = self.env.config['colorFile']
        self.segment_input = segment_input
        self.joint_visual_signal = joint_visual_signal
        self.depth_signal = depth_signal
        n_channel = 3
        if segment_input:
            self.env.set_render_mode('semantic')
        else:
            self.env.set_render_mode('rgb')
        if joint_visual_signal: n_channel += 3
        if depth_signal: n_channel += 1
        self._observation_shape = (resolution[0], resolution[1], n_channel)
        self._observation_space = spaces.Box(0, 255, shape=self._observation_shape)

        self.max_steps = max_steps

        self.discrete_action = discrete_action
        if discrete_action:
            self._action_space = spaces.Discrete(n_discrete_actions)
        else:
            self._action_space = spaces.Tuple([spaces.Box(0, 1, shape=(4,)), spaces.Box(0, 1, shape=(2,))])

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # configs
        self.move_sensitivity = (move_sensitivity or default_move_sensitivity)  # at most * meters per frame
        self.rot_sensitivity = rotation_sensitivity
        self.dist_scale = dist_reward_scale or 1
        self.successRew = success_reward
        self.inroomRew = stay_room_reward or 0.2
        self.colideRew = collision_penalty_reward or 0.02
        self.goodMoveRew = correct_move_reward or 0.0

        self.last_obs = None
        self.last_info = None
        self._object_cnt = 0

        # config hardness
        self.hardness = None
        self.availCoors = None
        self._availCoorsDict = None
        self.reset_hardness(hardness)

        # temp storage
        self.collision_flag = False

        # episode counter
        self.current_episode_step = 0

        # config success measure
        assert success_measure in ['stay', 'see']
        self.success_measure = success_measure
        print('[RoomNavTask] >> Success Measure = <{}>'.format(success_measure))
        self.success_stay_cnt = 0
        if success_measure == 'see':
            self.room_target_object = dict()
            self._load_target_object_data(self.env.config['roomTargetFile'])

    def _load_target_object_data(self, roomTargetFile):
        with open(roomTargetFile) as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                c = np.array((row['r'], row['g'], row['b']), dtype=np.uint8)
                room = row['target_room']
                if room not in self.room_target_object:
                    self.room_target_object[room] = []
                self.room_target_object[room].append(c)

    """
    reset the target room type to navigate to
    when target is None, a valid target will be randomly selected
    """

    def reset_target(self, target=None):
        if target is None:
            target = random.choice(self.house.all_desired_roomTypes)
        else:
            assert target in self.house.all_desired_roomTypes, '[RoomNavTask] desired target <{}> does not exist in the current house!'.format(
                target)
        if self.house.setTargetRoom(target):  # target room changed!!!
            _id = self.house._id
            if self.house.targetRoomTp not in self._availCoorsDict[_id]:
                if self.hardness is None:
                    self.availCoors = self.house.connectedCoors
                else:
                    allowed_dist = self.house.maxConnDist * self.hardness
                    self.availCoors = [c for c in self.house.connectedCoors
                                       if self.house.connMap[c[0], c[1]] <= allowed_dist]
                self._availCoorsDict[_id][self.house.targetRoomTp] = self.availCoors
            else:
                self.availCoors = self._availCoorsDict[_id][self.house.targetRoomTp]

    @property
    def house(self):
        return self.env.house

    """
    gym api: reset function
    when target is not None, we will set the target room type to navigate to
    """

    def reset(self, target=None):
        # clear episode steps
        self.current_episode_step = 0
        self.success_stay_cnt = 0
        self._object_cnt = 0

        # reset house
        self.env.reset_house()

        self.house.targetRoomTp = None  # [NOTE] IMPORTANT! clear this!!!!!

        # reset target room
        self.reset_target(target=target)  # randomly reset

        # general birth place
        gx, gy = random.choice(self.availCoors)
        self.collision_flag = False
        # generate state
        x, y = self.house.to_coor(gx, gy, True)
        self.env.reset(x=x, y=y)
        self.last_obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = np.concatenate([self.env.render(mode='rgb'), self.last_obs], axis=-1)
        ret_obs = self.last_obs
        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            ret_obs = np.concatenate([ret_obs, dep_sig], axis=-1)
        self.last_info = self.info
        return ret_obs

    def _apply_action(self, action):
        if self.discrete_action:
            return discrete_actions[action]
        else:
            rot = action[1][0] - action[1][1]
            act = action[0]
            return (act[0] - act[1]), (act[2] - act[3]), rot

    def _is_success(self, raw_dist):
        if raw_dist > 0:
            self.success_stay_cnt = 0
            return False
        if self.success_measure == 'stay':
            self.success_stay_cnt += 1
            return self.success_stay_cnt >= success_stay_time_steps
        # self.success_measure == 'see'
        flag_see_target_objects = False
        object_color_list = self.room_target_object[self.house.targetRoomTp]
        if (self.last_obs is not None) and self.segment_input:
            seg_obs = self.last_obs if not self.joint_visual_signal else self.last_obs[:, :, 3:6]
        else:
            seg_obs = self.env.render(mode='semantic')
        self._object_cnt = 0
        for c in object_color_list:
            cur_n = np.sum(np.all(seg_obs == c, axis=2))
            self._object_cnt += cur_n
            if self._object_cnt >= n_pixel_for_object_see:
                flag_see_target_objects = True
                break
        if flag_see_target_objects:
            self.success_stay_cnt += 1
        else:
            self.success_stay_cnt = 0  # did not see any target objects!
        return self.success_stay_cnt >= success_see_target_time_steps

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def info(self):
        ret = self.env.info
        gx, gy = ret['grid']
        ret['dist'] = dist = self.house.connMap[gx, gy]
        ret['scaled_dist'] = self.house.getScaledDist(gx, gy)
        ret['optsteps'] = int(dist / (self.move_sensitivity / self.house.grid_det) + 0.5)
        ret['collision'] = int(self.collision_flag)
        ret['target_room'] = self.house.targetRoomTp
        return ret

    """
    return all the available target room types of the current house
    """

    def get_avail_targets(self):
        return self.house.all_desired_roomTypes

    """
    reset the hardness of the task
    """

    def reset_hardness(self, hardness=None):
        self.hardness = hardness
        if hardness is None:
            self.availCoors = self.house.connectedCoors
        else:
            allowed_dist = self.house.maxConnDist * hardness
            self.availCoors = [c for c in self.house.connectedCoors
                               if self.house.connMap[c[0], c[1]] <= allowed_dist]
        n_house = self.env.num_house
        self._availCoorsDict = [dict() for i in range(n_house)]
        self._availCoorsDict[self.house._id][self.house.targetRoomTp] = self.availCoors

    """
    recover the state (location) of the agent from the info dictionary
    """

    def set_state(self, info):
        if len(info['pos']) == 2:
            self.env.reset(x=info['pos'][0], y=info['pos'][1], yaw=info['yaw'])
        else:
            self.env.reset(x=info['pos'][0], y=info['pos'][1], z=info['pos'][2], yaw=info['yaw'])

    """
    return 2d topdown map
    """

    def get_2dmap(self):
        return self.env.gen_2dmap()

    """
    show a image.
    if img is not None, it will visualize the img to monitor
    if img is None, it will return a img object of the observation
    Note: whenever this function is called with img not None, the task cannot perform rendering any more!!!!
    """

    def show(self, img=None):
        return self.env.show(img=img,
                             renderMapLoc=(None if img is not None else self.info['grid']),
                             renderSegment=True,
                             display=(img is not None))

    def debug_show(self):
        return self.env.debug_render()


if __name__ == '__main__':
    from House3D.common import load_config

    modes = [RenderMode.RGB, RenderMode.SEMANTIC, RenderMode.INSTANCE, RenderMode.DEPTH]

    api = objrender.RenderAPI(w=400, h=400, device=0)
    cfg = load_config('config.json')

    House_dir = '/data/vision/oliva/scenedataset/activevision/suncg/suncg_data/house'
    dataset_root = '/data/vision/oliva/scenedataset/syntheticscene/IntegralTopViewMaskDataset/'
    topdown_data_root = '/data/vision/oliva/scenedataset/syntheticscene/TopViewMaskDataset/'
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    ruined = []
    available_houses = os.listdir(topdown_data_root)
    for house in ['6088ec426c385150a19762549357f508']:
        #if os.path.isdir(os.path.join(dataset_root, house)):
        #    print(house + ' Already exist!')
        #    continue
        try:
            env = Environment(api, house, cfg)
        except:
            ruined.append(house)
            print('House id ' + house + ' can not be used')
            continue
        print('House ' + house + ' extracting...')
        if not os.path.isdir(os.path.join(dataset_root, house)):
            os.mkdir(os.path.join(dataset_root, house))
        task = CollectDataTask(env, hardness=0.6, discrete_action=True)
        house_obj = task.house
        connMap = house_obj.connMap
        cam = task.env.cam
        ppi = 4
        for k, mod in enumerate(['rgb', 'sem']):
            InteMap = np.uint8(np.zeros((connMap.shape[0] // ppi * ppi, \
                connMap.shape[1] // ppi * ppi, 3)))
            print('Generating House: ' + house)
            pb = SimpleProgressBar()
            for i in range(InteMap.shape[0] // ppi):
                for j in range(InteMap.shape[1] // ppi):
                    percent = (i * InteMap.shape[0] // ppi + j) * 100 / (InteMap.shape[0] * InteMap.shape[1] // (ppi*ppi))
                    pb.update(percent)

                    x, y = house_obj.to_coor(i*ppi+ppi//2, j*ppi+ppi//2, True)
                    info = {'pos': [x, y, 3.6], 'yaw': 0}
                    task.set_state(info)
                    cube_map = task.env.render_cube_map(modes[k])
                    cube_map = cube_map[:, - cube_map.shape[1] // 6 :]
                    cube_map = cube_map[:, :, ::-1]
                    coor_dir = os.path.join(dataset_root, house, \
                            'x_{}_y_{}'.format(i*ppi+ppi//2, j*ppi+ppi//2))
                    if not os.path.isdir(coor_dir):
                        os.mkdir(coor_dir)
                    cv2.imwrite(os.path.join(coor_dir, 'OverviewMask.png'), cube_map)
                    central_pix = cube_map[(cube_map.shape[0]//2-ppi//2):(cube_map.shape[0]//2+ppi//2), (cube_map.shape[1]//2-ppi//2):(cube_map.shape[1]//2+ppi//2)]
                    InteMap[i*ppi:(i+1)*ppi,j*ppi:(j+1)*ppi] = central_pix

                    for yaw in [0, 45]:
                        info = {'pos': [x, y, 1], 'yaw': yaw}
                        task.set_state(info)
                        cube_map = task.env.render_cube_map(modes[k])
                        cube_map = cube_map[:, :4 * cube_map.shape[1] // 6]
                        cube_map = cube_map[:, :, ::-1]
                        cv2.imwrite(os.path.join(coor_dir, 'mode={}_{}.png'.format(mod, yaw)), cube_map)

            cv2.imwrite(os.path.join(dataset_root, house, 'InteMap-{}.png'.format(mod)), InteMap)

    print(ruined)
    print(len(ruined))
