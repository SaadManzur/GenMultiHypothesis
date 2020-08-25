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

def unnormalize_data(X, mean, std):

    for i in range(X.shape[0]):
        X[i, 1:] = np.multiply(X[i, 1:], std[1:]) + mean[1:]

    return X