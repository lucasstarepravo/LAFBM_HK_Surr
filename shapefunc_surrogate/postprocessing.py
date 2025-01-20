import numpy as np
import math


def rescale_output(predicted_w, neigh_xy_d, h):

    c = 2.22204350233 # This was obtained from the pythonProject file when optimising c for avg_w = 1/(c*s)**2

    avg_dist = np.mean(abs(neigh_xy_d), axis=1)
    avg_weight = 1/(c*avg_dist)**2
    avg_weight = avg_weight[:, np.newaxis]
    predicted_w = predicted_w/h**2 + avg_weight
    return predicted_w


def denormalise_output(predicted_w, h, dtype):

    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')

    w_rescale_std = h

    if dtype.lower() == 'laplace':
        w_rescale_std = w_rescale_std ** 2

    stand_w = predicted_w/w_rescale_std
    return stand_w

