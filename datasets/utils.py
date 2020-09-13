import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    return X / w * 2 - [1, h / (w * 1.0)]

def normalize_data(X, mean, std, skip_root=False):
    
    for i in range(X.shape[0]):
        if not skip_root:
            X[i, :] = np.divide(X[i, :] - mean[:], std[:])
        else:
            X[i, 1:] = np.divide(X[i, 1:] - mean[1:], std[1:])

    return X

def unnormalize_data(X, mean, std, skip_root=False):

    for i in range(X.shape[0]):
        if not skip_root:
            X[i, :] = np.divide(X[i, :] - mean[:], std[:])
        else:
            X[i, 1:] = np.multiply(X[i, 1:], std[1:]) + mean[1:]

    return X

def rotate_y(joints, angle):
    matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    return np.matmul(matrix, joints.T).T

def rotate_x(joints, angle):
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), np.sin(angle)],
        [0.0, -np.sin(angle), np.cos(angle)]
    ])

    return np.matmul(matrix, joints.T).T

def plot_3d(joints, subplot, parents, left, right):
    subplot.scatter(joints[:, 0], joints[:, 1], zs=joints[:, 2], color='k')
    
    for i in range(joints.shape[0]):
        parent = parents[i]
        
        if parent < 0:
            continue
            
        color = 'k'
        if i in left and parent in left:
            color = 'r'
        elif i in right and parent in right:
            color = 'b'
        
        subplot.plot([joints[i, 0], joints[parent, 0]],
                     [joints[i, 1], joints[parent, 1]],
                     zs=[joints[i, 2], joints[parent, 2]],
                     color=color)

        subplot.text(joints[i, 0], joints[i, 1], joints[i, 2], str(i), (1, 1, 0))
    
    subplot.set_xlim(-1, 1)
    subplot.set_ylim(1, -1)
    subplot.set_zlim(-1, 1)
    
    subplot.view_init(azim=95, elev=-75)