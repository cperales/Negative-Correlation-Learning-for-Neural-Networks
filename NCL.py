import numpy as np
from sklearn import datasets, preprocessing, model_selection
import time


def test():
    """
    Testing Negative Correlation Learning algorithm.
    :param X: dataset with features.
    :param y: targets.
    :return:
    """
    # Parameters
    max_iter = 100
    ensemble_size = 5
    h = 20  # Number of neurons in the hidden layer

    # Data
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x=x,
                                                                        y=y,
                                                                        test_size=0.5,
                                                                        random_state=0)
    n_test = x_test.shape[0]
    J = y_train.shape[1]
    lambda_ = 0.5
    eta = 0.1

    start = time.perf_counter()
    model = NCL()
    model.train(x, y, ensemble_size, h, max_iter, lambda_, eta)
    end = time.perf_counter()

    print('Elapsed time =', end - start)
    rmse_value = rmse(model.predict(x_test), y_test)
    print('RMSE =', rmse_value)


def load_dataset():
    """
    Load data from Boston housing regression.

    :return: data scaled and target.
    """
    X, y = datasets.load_boston(return_X_y=True)
    X_scaled = preprocessing.scale(X)
    return X_scaled, y


def rmse(a, b):
    """
    Root Mean Squared Error metric.

    :param a:
    :param b:
    :return: RMSE value
    """
    return np.linalg.norm(a - b, ord=2)


class NeuralNetwork:
    """
    Neural Network.
    """
    h: int
    eta: float

    def __init__(self, h, eta):
        self.h = h
        # self.x = x
        # self.y = y
        self.eta = eta

    def initial(self, x, y):
        """
        It must return and object net.

        :param x:
        :param y:
        :return:
        """
        net = None
        return net

    def forward(self, x_i):
        pass

    def backward(self, x_i):
        pass

    def predict(self, x_test):
        """
        Output of predict y value for x test.

        :param x_test:
        :return:
        """
        f = 0.0
        return f


class NCL:
    """
    Negative Correlation Learning ensemble.
    """
    ensemble_size: int
    max_iter: int
    lambda_: float
    eta: float
    base = None

    def __init__(self):
        pass

    def train(self, x, y, ensemble_size, h, max_iter, lambda_, eta):
        """
        Training ensemble

        :param x: data.
        :param y: target.
        :param ensemble_size:
        :param h:
        :param max_iter:
        :param lambda_:
        :param eta:
        """
        # Parameter
        self.ensemble_size = ensemble_size
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.eta = eta
        n_train, dim = x.shape

        # Initialization
        self.base = list()
        for s in range(self.ensemble_size):
            nn = NeuralNetwork(h, eta)
            nn.initial(x, y)
            self.base.append(nn)
            self.base
        self.base = [NeuralNetwork(x, y, h) for s in range(self.ensemble_size)]
        curve = np.inf * np.ones(1, self.max_iter)

        # Training
        for iter_ in range(self.max_iter):  # Each epoch
            for i in range(n_train):  # Each training element
                x_i = x[i, :]
                y_i = y[i, :]
                f_bar = self.predict(x_i)
                for s in range(self.ensemble_size):  # Each base learner
                    penalty = - self.lambda_ * (self.base.forward(x_i, s) - f_bar)
                    self.base[s].backward(x_i, y_i, penalty, self.eta)

    def predict(self, x_test):
        """
        :param x:
        :return: f_bar
        """
        f_bar = np.sum([self.base[s].predict(x_test) for s in range(self.ensemble_size)])
        return f_bar


if __name__ == '__main__':
    test()
