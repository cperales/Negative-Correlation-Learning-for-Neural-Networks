import numpy as np
from sklearn import datasets, preprocessing, model_selection
from scipy.special import expit
import time
import matplotlib.pyplot as plt


def test():
    """
    Testing Negative Correlation Learning algorithm.
    :param X: dataset with features.
    :param y: targets.
    :return:
    """
    # Parameters
    max_iter = 1000
    ensemble_size = 25
    h = 20  # Number of neurons in the hidden layer

    # Data
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
                                                                        test_size=0.5,
                                                                        random_state=0)

    # x_train = np.array([[0, 0, 1],
    #                     [1, 1, 1],
    #                     [1, 0, 1],
    #                     [0, 1, 1],
    #                     [1, 1, 1],
    #                     [0, 0, 0]])
    # y_train = np.array([[0],
    #                     [1],
    #                     [1],
    #                     [0],
    #                     [1],
    #                     [0]])
    # x_test = np.array([1, 0, 0])
    # y_test = np.array([[1]])

    print('Size of training =', x_train.shape[0])
    print('Size of testing =', x_test.shape[0])
    print('Dimension of data =', x_train.shape[1])
    lambda_ = 0.5
    eta = 0.1

    start = time.perf_counter()
    model = NCL(classification=True)
    model.train(x_train, y_train, ensemble_size, h, max_iter, lambda_, eta)
    end = time.perf_counter()

    print('Elapsed time =', end - start)
    pred = model.predict(x_test)
    print('Prediction =', pred)
    rmse_value = rmse(pred, y_test)
    print('RMSE =', rmse_value)
    plt.plot(model.rmse_array)
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')
    plt.show()


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

    def __init__(self, h, eta, classification=True):
        self.h = h
        # self.x = x
        # self.y = y
        self.learning_rate = eta
        if classification is False:
            self.activation_output = lambda x: np.array(x, dtype=np.float)
            self.der_activation_output = self.activation_output
        else:
            self.activation_output = sigmoid
            self.der_activation_output = sigmoid_derivative

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
        self.input_weight = 2.0 * np.random.random((self.dim, self.h)) - 1.0
        self.hidden_bias = np.random.random((self.h, 1))
        self.output_weight = 2.0 * np.random.random((self.h, self.j)) - 1.0
        self.output_bias = np.random.random((self.j, 1))
        return self

    def backward(self, x, y, penalty):
        hidden_layer, output = self.forward(x)
        error = output - y
        print('Error =', np.linalg.norm(error), ', NC penalty =', np.linalg.norm(penalty))
        print()
        nc_error = error + penalty

        # Output layer
        output_delta = nc_error * self.der_activation_output(output)
        self.output_bias -= np.mean(self.learning_rate * output_delta)
        self.output_weight -= self.learning_rate * np.dot(hidden_layer.T, output_delta)

        # Hidden layer
        hidden_delta = np.dot(output_delta, self.output_weight.T) * sigmoid_derivative(hidden_layer)
        self.hidden_bias -= np.mean(self.learning_rate * hidden_delta, axis=0).reshape(self.h, 1)
        self.input_weight -= self.learning_rate * np.dot(x.T, hidden_delta)

    def forward(self, x_test):
        """
        Output of predict y value for x test.

        :param x_test:
        :return:
        """
        temp_h = np.dot(x_test, self.input_weight) + self.hidden_bias.T
        hidden_layer = sigmoid(temp_h)
        temp_o = np.dot(hidden_layer, self.output_weight) + self.output_bias.T
        output_layer = self.activation_output(temp_o)
        return hidden_layer, output_layer

    def predict(self, x_test):
        _, output_layer = self.forward(x_test)
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
    classification: bool

    def __init__(self, classification=True):
        self.classification = classification

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

        self.base = [NeuralNetwork(h, learning_rate, classification=self.classification).initial(x, y)
                     for s in range(self.ensemble_size)]
        self.rmse_array = np.inf * np.ones(self.max_iter)

        # Training
        for iter_ in range(self.max_iter):  # Each epoch
            f_bar = self.predict(x)
            for s in range(self.ensemble_size):  # Each base learner
                penalty = - self.lambda_ * (self.base[s].predict(x) - f_bar)
                self.base[s].backward(x, y, penalty)
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