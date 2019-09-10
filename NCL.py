import numpy as np
from sklearn import datasets, preprocessing


def load_dataset():
    """
    Load data from Boston housing regression.

    :return: data scaled and target.
    """
    X, y = datasets.load_boston(return_X_y=True)
    X_scaled = preprocessing.scale(X)
    return X_scaled, y


def ncl(X, y):
    """
    Negative Correlation Learning algorithm for neural networks.
    :param X: dataset with features.
    :param y: targets.
    :return:
    """



def test():
    pass


if __name__ == '__main__':
    test()