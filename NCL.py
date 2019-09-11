import numpy as np
from sklearn import datasets, preprocessing, model_selection
from scipy.special import expit
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
    n: int
    dim: int
    h: int
    j: int
    eta: float
    input_weight = None
    hidden_bias = None
    output_weight = None
    output_bias = None

    @staticmethod
    def activation(x):
        """
        Activation function. From MATLAB example, log-sigmoid.
        TODO: modify for other activation functions.

        :param x:
        :return:
        """
        return expit(x)

    def __init__(self, h, eta):
        self.h = h
        # self.x = x
        # self.y = y
        self.eta = eta

    def initial(self, x, y):
        """
        Random values for net are initialized.

        :param x:
        :param y:
        :return:
        """
        self.n = x.shape[0]
        self.dim = x.shape[1]
        self.j = y.shape[1]
        self.input_weight = np.random.rand(self.dim, self.h)
        self.hidden_bias = np.random.rand(1, self.h)
        self.output_weight = np.random.rand(self.h, self.j)
        self.output_bias = np.random.rand(1, self.j)

    def backward(self, x, y, penalty):
        hidden_layer, output_layer = self.forward(x)
        nc = output_layer - y + penalty

        delta_output_weight = output_layer * np.dot((1.0 - output_layer).T, hidden_layer)
        delta_output_bias = output_layer * (1.0 - output_layer)
        delta_input_weight = self.output_weight * (hidden_layer * )
        eta = self.eta

    def forward(self, x_test):
        """
        Output of predict y value for x test.

        :param x_test:
        :return:
        """
        temp_h = (np.dot(x_test, self.input_weight.T) + self.hidden_bias).T
        hidden_layer = self.activation(temp_h)
        temp_o = (np.dot(hidden_layer, self.output_weight.T) + self.output_bias).T
        output_layer = self.activation(temp_o)
        return hidden_layer, output_layer

    def predict(self, x_test):
        hidden_layer, output_layer = self.forward(x_test)
        return output_layer


class NCL:
    """
    Negative Correlation Learning ensemble.
    """
    ensemble_size: int
    max_iter: int
    lambda_: float
    eta: float
    base = None
    rmse_array: np.array

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
        n_train, dim = x.shape

        # # Initialization
        # self.base = list()
        # for s in range(self.ensemble_size):
        #     nn = NeuralNetwork(h, eta)
        #     nn.initial(x, y)
        #     self.base.append(nn)
        self.base = [NeuralNetwork(h, eta).initial(x, y) for s in range(self.ensemble_size)]
        self.rmse_array = np.inf * np.ones(1, self.max_iter)

        # Training
        for iter_ in range(self.max_iter):  # Each epoch
            for i in range(n_train):  # Each training element
                x_i = x[i, :]
                y_i = y[i, :]
                f_bar = self.predict(x_i)
                for s in range(self.ensemble_size):  # Each base learner
                    penalty = - self.lambda_ * (self.base[s].predict(x_i) - f_bar)
                    self.base[s].backward(x_i, y_i, penalty)
            self.rmse_array[iter_] = rmse(self.predict(x), y)

    def predict(self, x_test):
        """
        :param x:
        :return: f_bar
        """
        f_bar = np.mean([self.base[s].predict(x_test) for s in range(self.ensemble_size)],
                        axis=1)
        return f_bar


if __name__ == '__main__':
    test()
