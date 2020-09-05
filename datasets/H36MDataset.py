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
                        
skeleton_H36M_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]

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

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = ['S0']
TEST_SUBJECTS = ['S0']

def visualize_2d(joints_2d, filename="debug"):
    plt.scatter(joints_2d[:, 0], joints_2d[:, 1], color='k')

    for i in range(joints_2d.shape[0]):
        # parent = parents[i]

        # plt.annotate(str(i) + " " + NAMES_H36M[i], [joints_2d[i, 0], joints_2d[i, 1]]) 
        color = 'r'
        if i in joints_right:
            color = 'k'
        
        """
        if parent >= 0:
            plt.plot([joints_2d[i, 0], joints_2d[parent, 0]],
                     [joints_2d[i, 1], joints_2d[parent, 1]],
                     color=color)
        """

    # plt.xlim((-1, 1))
    # plt.ylim((1, -1))
    plt.savefig(filename + ".png")

def visualize_3d(joints_3d, filename="debug3d"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], zs=joints_3d[:, 2], color='k')

    for i in range(joints_3d.shape[0]):
        # parent = parents[i]

        # ax.annotate(str(i) + " " + NAMES_H36M[i], [joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2]]) 
        color = 'r'
        if i in joints_right:
            color = 'k'

        """
        if parent >= 0:
            ax.plot([joints_3d[i, 0], joints_3d[parent, 0]],
                     [joints_3d[i, 1], joints_3d[parent, 1]],
                     zs=[joints_3d[i, 2], joints_3d[parent, 2]],
                     color=color)
        """

    ax.set_xlim((-2, 2))
    ax.set_ylim((2, -2))
    ax.set_zlim((-2, 2))
    # plt.savefig(filename + ".png")

    plt.show()


class H36MDataset(object):

    def __init__(self, path, load_metrics=None):
        # TODO: Update the fps here if needed
        super(H36MDataset, self).__init__()

        # TODO: Update camera later if needed
        self.cameras = None

        self._data_train = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}
        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.cameras = []

        self.load_data(path, load_metrics)

    def load_data(self, path, load_metrics):
        filename, _ = os.path.splitext(os.path.basename(path))

        indices_to_select_2d = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
        indices_to_select_3d = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

        self.cameras = cameras.load_cameras(os.path.join(path, "cameras.h5"))

        TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
        TEST_SUBJECTS  = [9, 11]

        actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

        trainset = self.load_3d_data(path, TRAIN_SUBJECTS, actions)
        testset = self.load_3d_data(path, TEST_SUBJECTS, actions)

        d2d_train, d3d_train = self.project_to_cameras(trainset)
        d2d_valid, d3d_valid = self.project_to_cameras(testset)

        self._data_train['2d'] = np.array(d2d_train)[:, indices_to_select_2d, :]
        self._data_train['3d'] = self.root_center(np.array(d3d_train))[:, indices_to_select_2d, :]
        self._data_valid['2d'] = np.array(d2d_valid)[:, indices_to_select_2d, :]
        self._data_valid['3d'] = self.root_center(np.array(d3d_valid))[:, indices_to_select_2d, :]

        if not load_metrics:
            self.mean_3d = np.mean(self._data_train['3d'], axis=0)
            self.std_3d = np.std(self._data_train['3d'], axis=0)
            self.mean_2d = np.mean(self._data_train['2d'], axis=0)
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

        width=1000
        height=1002

        self._data_train['3d'] = normalize_data(self._data_train['3d'], self.mean_3d, self.std_3d, skip_root=True)
        self._data_train['2d'] = normalize_data(self._data_train['2d'], self.mean_2d, self.std_2d)

        self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
        self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d)


    def load_3d_data(self, path, subjects, actions):

        data = {}

        total_data_points = 0
        for subj in subjects:
            for action in actions:
                print('Reading subject {0}, action {1}'.format(subj, action))

                dpath = os.path.join( path, 'S{0}'.format(subj), 'MyPoses/3D_positions', '{0}*.h5'.format(action) )

                fnames = glob.glob( dpath )

                loaded_seqs = 0
                for fname in fnames:
                    seqname = os.path.basename( fname )

                    if action == "Sitting" and seqname.startswith( "SittingDown" ):
                        continue

                    if seqname.startswith( action ):
                        loaded_seqs = loaded_seqs + 1

                        with h5py.File( fname, 'r' ) as h5f:
                            poses = h5f['3D_positions'][:]
                            poses = poses.T

                            data[( subj, action, seqname )] = poses.reshape((-1, 32, 3))

                            total_data_points += poses.shape[0]

        print("Total 3d points loaded: ", total_data_points)

        return data

    
    def root_center(self, data3d):

        for i in range(data3d.shape[0]):
            data3d[i, :, :] -= data3d[i, 0, :]

        return data3d

    def normalization_stats_3d(self, pose_set_3d):

        for key in pose_set_3d.keys():
            poses = pose_set_3d[key]
            
            for i in range(poses.shape[0]):
                poses[i, :, :] -= poses[i, 0, :]

            pose_set_3d[key] = poses

        complete_data = np.vstack(pose_set_3d.values())

        return np.mean(complete_data, axis=0), np.std(complete_data, axis=0)

    def project_to_cameras( self, poses_set ):
        """
        Project 3d poses using camera parameters

        Args
        poses_set: dictionary with 3d poses
        cams: dictionary with camera parameters
        ncams: number of cameras per subject
        Returns
        t2d: dictionary with 2d poses
        """
        t2d = []
        t3d = []

        total_points = 0
        once = False
        for key in poses_set.keys():
            (subj, action, sqename) = key
            t3dw = poses_set[key]

            for cam in range(4):
                R, T, f, c, k, p, name = self.cameras[ (subj, cam+1) ]
                t3dc = cameras.world_to_camera_frame( np.reshape(t3dw, [-1, 3]), R, T)
                pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3dw, [-1, 3]), R, T, f, c, k, p )
                pts2d = np.reshape( pts2d, [-1, 32, 2] )
                total_points += pts2d.shape[0]
                t2d.append(pts2d)
                t3d.append(t3dc.reshape((-1, 32, 3)))

        t2d = np.vstack(t2d)
        t3d = np.vstack(t3d)

        print("Projected points: ", total_points)

        return t2d, t3d

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
        return self._data_train['2d'].reshape((-1, 16*2))

    def get_3d_train(self):
        return self._data_train['3d'].reshape((-1, 16*3))
