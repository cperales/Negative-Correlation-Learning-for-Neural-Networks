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
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
                                                                        test_size=0.5,
                                                                        random_state=0)
    print('Size of training =', x_train.shape[0])
    print('Size of testing =', x_test.shape[0])
    print('Dimension of data =', x_train.shape[1])
    lambda_ = 0.5
    eta = 0.1

    start = time.perf_counter()
    model = NCL()
    model.train(x_train, y_train, ensemble_size, h, max_iter, lambda_, eta)
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
    y = y.reshape((len(y), 1))
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


def sigmoid(x):
    """
    Sigmoid function. It can be replaced with scipy.special.expit.
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    """
    Derivate of the sigmoid function.
    We assume y is already sigmoided.
    :param y:
    :return:
    """
    return y * (1.0 - y)


class NeuralNetwork:
    """
    Neural Network.
    """
    n: int
    dim: int
    h: int
    j: int
    learning_rate: float
    input_weight = None
    hidden_bias = None
    output_weight = None
    output_bias = None

    def __init__(self, h, eta):
        self.h = h
        # self.x = x
        # self.y = y
        self.learning_rate = eta

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
        self.input_weight = 2.0 * np.random.rand(self.dim, self.h) - 1.0
        self.hidden_bias = np.random.rand(1, self.h)
        self.output_weight = 2.0 * np.random.rand(self.h, self.j) - 1.0
        self.output_bias = np.random.rand(1, self.j)
        return self

    def backward(self, x, y, penalty=0.0):
        hidden_layer, output = self.forward(x)
        nc_error = output - y + penalty

        adj_output_weight = sigmoid_derivative(nc_error)
        delta_output_weight = np.dot(hidden_layer.T, adj_output_weight)
        delta_output_bias = - adj_output_weight

        adj_input_weight = adj_output_weight * self.output_weight * sigmoid_derivative(self.input_weight)
        adj_input_weight = np.dot(np.dot(adj_output_weight, self.output_weight),
                                  sigmoid_derivative(self.input_weight))
        delta_input_weight = np.dot(adj_input_weight, x)
        delta_hidden_bias = - adj_input_weight

        self.output_weight -= self.learning_rate * delta_output_weight
        self.output_bias -= self.learning_rate * delta_output_bias
        self.input_weight -= self.learning_rate * delta_input_weight
        self.hidden_bias -= self.learning_rate * delta_hidden_bias

    def backward_1(self, x, y, penalty=0.0):
        hidden_layer, output_layer = self.forward(x)
        nc_error = output_layer - y + penalty

        adj_output_weight = np.dot(sigmoid_derivative(output_layer).T, hidden_layer)
        delta_output_weight = np.dot(hidden_layer.T,  adj_output_weight)
        delta_output_bias = nc_error * sigmoid_derivative(output_layer)


        adj_input_weight = np.dot(delta_output_weight, self.output_weight.T) * sigmoid_derivative(hidden_layer).T
        delta_input_weight = np.dot(x.T, adj_input_weight)
        # delta_input_weight = np.dot(x.T, nc_error * self.output_weight * sigmoid_derivative(hidden_layer))

        delta_hidden_bias = np.dot(delta_output_weight, self.output_weight.T) * sigmoid_derivative(hidden_layer).T

        self.output_weight -= self.learning_rate * delta_output_weight
        self.output_bias -= self.learning_rate * delta_output_bias
        self.input_weight -= self.learning_rate * delta_input_weight
        self.hidden_bias -= self.learning_rate * nc_error * delta_hidden_bias

    def forward(self, x_test):
        """
        Output of predict y value for x test.

        :param x_test:
        :return:
        """
        temp_h = (np.dot(x_test, self.input_weight) + self.hidden_bias)
        hidden_layer = sigmoid(temp_h)
        temp_o = (np.dot(hidden_layer, self.output_weight) + self.output_bias)
        output_layer = sigmoid(temp_o)
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
    learning_rate: float
    base = None
    rmse_array: np.array

    def __init__(self):
        pass

    def train(self, x, y, ensemble_size, h, max_iter, lambda_, learning_rate):
        """
        Training ensemble

        :param x: data.
        :param y: target.
        :param ensemble_size:
        :param h:
        :param max_iter:
        :param lambda_:
        :param learning_rate:
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
        self.base = [NeuralNetwork(h, learning_rate).initial(x, y) for s in range(self.ensemble_size)]
        self.rmse_array = np.inf * np.ones(self.max_iter)

        # Training
        for iter_ in range(self.max_iter):  # Each epoch
            f_bar = self.predict(x)
            for s in range(self.ensemble_size):  # Each base learner
                penalty = - self.lambda_ * (self.base[s].predict(x) - f_bar)
                self.base[s].backward(x, y, penalty)
            # for i in range(n_train):  # Each training element
            #     x_i = x[i, :]
            #     y_i = y[i, :]
            #     f_bar = self.predict(x_i)
            #     for s in range(self.ensemble_size):  # Each base learner
            #         penalty = - self.lambda_ * (self.base[s].predict(x_i) - f_bar)
            #         self.base[s].backward(x_i, y_i, penalty)
            self.rmse_array[iter_] = rmse(self.predict(x), y)

    def predict(self, x_test):
        """
        :param x:
        :return: f_bar
        """
        f_bar = np.mean([self.base[s].predict(x_test) for s in range(self.ensemble_size)],
                        axis=0)
        return f_bar


if __name__ == '__main__':
    test()
