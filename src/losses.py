import numpy as np


def cross_entropy(y_true_labels, y_prob_matrix):
    eps = 1e-12
    y_prob_matrix = np.clip(y_prob_matrix, eps, 1 - eps)
    true_class_probabilities = y_prob_matrix[np.arange(len(y_true_labels)), y_true_labels]
    return np.mean(-np.log(true_class_probabilities))


def binary_cross_entropy(y_true, y_prob):
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return np.mean(loss)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mse(y_true, y_pred):
    errors = y_true - y_pred
    return np.mean(errors ** 2)


def mae(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    return np.mean(errors)
