import os
import h5py
import glob
import copy
import numpy as np
from tqdm import tqdm
from datasets.utils import normalize_screen_coordinates
from datasets.utils import normalize_data, unnormalize_data
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import src.cameras as cameras

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]

NAMES_H36M = ['']*16
NAMES_H36M[0] = 'Hip'
NAMES_H36M[1] = 'RHip'
NAMES_H36M[2] = 'RKnee'
NAMES_H36M[3] = 'RAnkle'
NAMES_H36M[4] = 'LHip'
NAMES_H36M[5] = 'LKnee'
NAMES_H36M[6] = 'LAnkle'
NAMES_H36M[7] = 'Spine2'
NAMES_H36M[8] = 'Neck'
NAMES_H36M[9] = 'Head'
NAMES_H36M[10] = 'LUpperArm'
NAMES_H36M[11] = 'LElbow'
NAMES_H36M[12] = 'LWrist'
NAMES_H36M[13] = 'RUpperArm'
NAMES_H36M[14] = 'RElbow'
NAMES_H36M[15] = 'RWrist'

TRAIN_SUBJECTS = ['S0']
TEST_SUBJECTS = ['S0']

class H36MAugmentedDataset(object):

    def __init__(self, path, load_metrics=None, center_2d=False):
        # TODO: Update the fps here if needed
        super(H36MAugmentedDataset, self).__init__()

        # TODO: Update camera later if needed
        self.cameras = None

        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.center_2d = center_2d

        self.cameras = []

        self.load_data(path, load_metrics)

    def load_data(self, path, load_metrics):
        filename = os.path.splitext(os.path.basename(path))[0]

        data = np.load(path, allow_pickle=True)['data'].item()

        indices_to_select_2d = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
        indices_to_select_3d = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

        actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

        d2d = []
        d3d = []
        for action in actions:
            d2d.append(data[action]['2d'])
            d3d.append(data[action]['3d_cam'])
        
        d2d = np.vstack(d2d)
        d3d = np.vstack(d3d)

        if self.center_2d:
            self._data_valid['2d'] = self.root_center(d2d)[:, indices_to_select_2d, :]
        else:
            self._data_valid['2d'] = d2d[:, indices_to_select_2d, :]
        
        self._data_valid['3d'] = self.root_center(d3d)[:, indices_to_select_2d, :]

        if not load_metrics:
            self.mean_3d = np.mean(self._data_valid['3d'], axis=0)
            self.std_3d = np.std(self._data_valid['3d'], axis=0)
            self.mean_2d = np.mean(self._data_valid['2d'], axis=0)
            self.std_2d = np.std(self._data_valid['2d'], axis=0)

            if not os.path.exists(os.path.join("metrics/", filename + "_metrics.npz")):
                np.savez_compressed(
                    os.path.join("metrics/", filename + "_metrics"),
                    mean_2d=self.mean_2d, std_2d=self.std_2d,
                    mean_3d=self.mean_3d, std_3d=self.std_3d)
        else:
            data = np.load(load_metrics)
            self.mean_2d = data['mean_2d']
            self.std_2d = data['std_2d']
            self.mean_3d = data['mean_3d']
            self.std_3d = data['std_3d']

        self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
        self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)

    def root_center(self, data3d):

        for i in range(data3d.shape[0]):
            data3d[i, :, :] -= data3d[i, 0, :]

        return data3d

    def define_actions(self, action=None):
        all_actions = ["N"]

        if action is None:
            return all_actions

        if action not in all_actions:
            raise (ValueError, "Undefined action: {}".format(action))

        return [action]

    def get_2d_valid(self):
        return self._data_valid['2d'].reshape((-1, 16*2))

    def get_3d_valid(self):
        return self._data_valid['3d'].reshape((-1, 16*3))
    
    def get_2d_train(self):
        raise NotImplementedError

    def get_3d_train(self):
        raise NotImplementedError
