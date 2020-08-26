from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
LEFT = [4, 5, 6, 10, 11, 12]
RIGHT = [1, 2, 3, 13, 14, 15]

def plot_3d(joints, save_path):

    joints = joints / 1000

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], zs=joints[:, 2], color='k')

    for i in range(joints.shape[0]):
        parent = PARENTS[i]

        col = 'k'
        if i in LEFT and parent in LEFT:
            col = 'r'

        if parent >= 0:
            ax.plot([joints[i, 0], joints[parent, 0]],
                    [joints[i, 1], joints[parent, 1]],
                    zs=[joints[i, 2], joints[parent, 2]],
                    color=col)

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((-2, 2))

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    ax.view_init(elev=-88, azim=-90)

    plt.savefig(save_path)

    plt.close()
