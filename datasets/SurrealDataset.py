import copy
import numpy as np
from tqdm import tqdm
from datasets.utils import plot_3d
from datasets.utils import normalize_screen_coordinates
from datasets.utils import normalize_data, unnormalize_data
from matplotlib import pyplot as plt

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]

skeleton_surreal_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]

NAMES_3DPW = ['']*16
NAMES_3DPW[0] = 'Hip'
NAMES_3DPW[1] = 'RHip'
NAMES_3DPW[2] = 'RKnee'
NAMES_3DPW[3] = 'RAnkle'
NAMES_3DPW[4] = 'LHip'
NAMES_3DPW[5] = 'LKnee'
NAMES_3DPW[6] = 'LAnkle'
NAMES_3DPW[7] = 'Spine2'
NAMES_3DPW[8] = 'Neck'
NAMES_3DPW[9] = 'Head'
NAMES_3DPW[10] = 'LUpperArm'
NAMES_3DPW[11] = 'LElbow'
NAMES_3DPW[12] = 'LWrist'
NAMES_3DPW[13] = 'RUpperArm'
NAMES_3DPW[14] = 'RElbow'
NAMES_3DPW[15] = 'RWrist'

def visualize_2d(joints_2d, filename="debug"):
    plt.scatter(joints_2d[:, 0], joints_2d[:, 1], color='k')

    for i in range(joints_2d.shape[0]):
        parent = parents[i]

        plt.annotate(str(i) + " " + NAMES_3DPW[i], [joints_2d[i, 0], joints_2d[i, 1]]) 
        color = 'r'
        if i in joints_right:
            color = 'k'

        if parent >= 0:
            plt.plot([joints_2d[i, 0], joints_2d[parent, 0]],
                     [joints_2d[i, 1], joints_2d[parent, 1]],
                     color=color)

    plt.xlim((0, 256))
    plt.ylim((256, 0))
    plt.savefig(filename + ".png")

class SurrealDataset(object):

    def __init__(self, path_train, path_valid, center_2d=False):
        super(SurrealDataset, self).__init__()

        # TODO: Update camera later if needed
        self.cameras = None

        self._data_train = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}
        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.center_2d = center_2d

        self.load_data(path_train, path_valid)

    def load_data(self, path_train, path_valid):

        data_train = np.load(path_train, allow_pickle=True)
        data_valid = np.load(path_valid, allow_pickle=True)

        indices_to_arrange = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
        indices_to_select = [0, 1, 4, 7, 2, 5, 8, 6, 12, 15, 17, 19, 21, 16, 18, 20]

        self._data_train['2d'] = data_train["data_2d"][:, indices_to_arrange, :][:, indices_to_select,  :]
        self._data_train['3d'] = data_train["data_3d"][:, indices_to_arrange, :][:, indices_to_select,  :]

        self._data_valid['2d'] = data_valid["data_2d"][:, indices_to_arrange, :][:, indices_to_select,  :]
        self._data_valid['3d'] = data_valid["data_3d"][:, indices_to_arrange, :][:, indices_to_select,  :]

        # visualize_2d(self._data_train['2d'][0, :, :], 'debug_surreal')

        print("Normalizing screen coordinates")
        for i in tqdm(range(self._data_train['3d'].shape[0])):
            self._data_train['3d'][i, :] -= self._data_train['3d'][i, 0]

            if self.center_2d:
                self._data_train['2d'][i, :] -= self._data_train['2d'][i, 0]
        
        self.plot_random()

        self.mean_2d = np.mean(self._data_train['2d'], axis=0)
        self.std_2d = np.std(self._data_train['2d'], axis=0)
        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)

        self._data_train['3d'] = normalize_data(self._data_train['3d'], self.mean_3d, self.std_3d, skip_root=True)
        self._data_train['2d'] = normalize_data(self._data_train['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)

        for i in tqdm(range(self._data_valid['3d'].shape[0])):
            self._data_valid['3d'][i, :] -= self._data_valid['3d'][i, 0]

            if self.center_2d:
                self._data_valid['2d'][i, :] -= self._data_valid['2d'][i, 0]
        
        self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
        self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)


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

    def plot_random(self):
        idx = np.random.randint(0, high=self._data_train['3d'].shape[0])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d(self._data_train['3d'][idx, :, :]/1000, ax, parents, joints_left, joints_right)
        plt.show()
