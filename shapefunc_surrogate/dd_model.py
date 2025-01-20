import numpy as np
from shapefunc_surrogate.preprocessing import *
from shapefunc_surrogate.postprocessing import *
import torch
import pickle as pk
import warnings


warnings.filterwarnings("ignore", message="input's size at dim=0 does not match num_features.*")


def monomial_power(polynomial):
    """

    :param polynomial:
    :return:
    """
    monomial_exponent = [(total_polynomial - i, i)
                         for total_polynomial in range(1, polynomial + 1)
                         for i in range(total_polynomial + 1)]
    return np.array(monomial_exponent)


def calc_moments(neigh_xy_d, scaled_w, polynomial):
    mon_power = monomial_power(polynomial)
    monomial = []
    for power_x, power_y in mon_power:
        monomial.append((neigh_xy_d[:, :, 0] ** power_x * neigh_xy_d[:, :, 1] ** power_y) /
                        (math.factorial(power_x) * math.factorial(power_y)))
    moments = np.array(monomial) * scaled_w
    moments = np.sum(moments, axis=2)
    return moments.T


def moments_normalised(stand_feature, predicted_w):
    stand_feature1 = stand_feature.reshape(stand_feature.shape[0], -1, 2)
    moments = calc_moments(stand_feature1, predicted_w, polynomial=2)
    return moments


def ann_predict(model, neigh_xy_dict, h, dtype='laplace'):

    ref_nodes = []
    distances = []

    for key in neigh_xy_dict:
        ref_nodes.append(key)
        distances.append(neigh_xy_dict[key])

    distances_array = np.array(distances)
    distances_array = distances_array[:, 1:, :]

    stand_feature = non_dimension(distances_array, h)
    stand_feature = stand_feature.reshape(stand_feature.shape[0], -1)
    stand_feature = torch.tensor(stand_feature, dtype=torch.float32)

    predicted_w = model.predict(stand_feature)
    scaled_w = denormalise_output(predicted_w, h, dtype=dtype)

    scaled_w = np.insert(scaled_w.detach().numpy(), 0, 0, axis=1)
    scaled_w = scaled_w.T

    scaled_w_dict = {}
    loop = 0

    for ref_node_coor in ref_nodes:
        scaled_w_dict[ref_node_coor] = scaled_w[:, loop]
        loop = loop + 1

    return scaled_w_dict
