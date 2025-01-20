import numpy as np


def non_dimension(features, h):
    stand_feature = features / h

    return stand_feature


def stdv_normalisation(features):

    f_mean = np.mean(features, axis=1, keepdims=True)
    f_std = np.std(features, axis=1, keepdims=True)

    stand_features = (features - f_mean) / f_std

    return stand_features, f_mean, f_std


def spread_norm(features):
    f_std = np.std(features, axis=1, keepdims=True)
    stand_features = features / f_std

    return stand_features, f_std
