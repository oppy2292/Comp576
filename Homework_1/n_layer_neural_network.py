__author__ = 'Yashwanth Lagisetty'

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def generate_model():
    """
    generate data
    :return: X: input data, y: labels as one-hot-encoding
    """
    np.random.seed(0)
    X, labels = datasets.make_moons(200, noise=0.2)
    y = np.zeros((200, 2))
    y[np.arange(labels.size), labels] = 1
    return X, y, labels


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


class Layer(object):
    """
    This class forward and backward passes through a layer
    """

    def __init__(self, in_dim, out_dim, actFun_type='tanh', seed=12, out_layer=False):
        """
        :param weights: weight matrix for layer
        :param bias: bias vector for layer
        :param actFun_type: activation function type
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.actFun_type = actFun_type
        self.seed = seed
        self.out_layer = out_layer

        np.random.seed(seed)
        self.weights = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)
        self.bias = np.zeros((1, self.out_dim))

    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        Function was Recycled from three_layer_neural_network.py
        """
        act = 0
        if type == 'tanh':
            # tanh(x) = sinh(x)/cosh(x) = (e^2x - 1)/(e^2x+1)
            act = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
        elif type == 'sigmoid':
            act = 1 / (1 + np.exp(-z))
        elif type == 'ReLU':
            act = np.maximum(0, z)
        return act

    def diff_actFun(self, z, type):
        """
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        Function was Recycled from three_layer_neural_network.py
        """
        diff_act = 0
        if type == 'tanh':
            num = 4 * np.exp(2 * z)
            den = (np.exp(2 * z) + 1) ** 2
            diff_act = num / den
        elif type == 'sigmoid':
            num = np.exp(-z)
            den = (1 + np.exp(-z)) ** 2
            diff_act = num / den
        elif type == 'ReLU':
            diff_act = (z > 0) * 1
        return diff_act

    def feedforward(self, X):
        """
        Computes forward pass for the layer
        :param X: input data
        :return: output of forward pass
        """

        # Perform Affine transformation first
        self.aff = np.dot(X, self.weights) + self.bias

        if self.out_layer:
            exp = np.exp(self.aff)
            self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        else:
            self.activ = self.actFun(self.aff, self.actFun_type)

        return None

    def backprop(self, act_prev, ddx_prev):
        """
        Computes backpropagation for layer
        :param act_prev: if current layer l, act_prev is output of layer l-1
        :param ddx_prev: if current layer l, ddx_prev is ddx of layer l+1
        :return: ddx ddw, partials w.r.t input and weights
        """
        del1 = ddx_prev * self.diff_actFun(self.aff, self.actFun_type)

        ddw = np.dot(del1.T, act_prev).T
        ddx = np.dot(del1, self.weights.T)
        ddb = np.sum(del1, axis=0)
        return ddx, ddw, ddb


class DeepNeuralNetwork(object):
    """
    Neural network class
    """

    def __init__(self, input_dims, n_units, output_dim, actFun_type='tanh', reg_lambda=0.01, epsilon=0.01, seed=12):
        """
        :param input_dims: input dimension
        :param n_units: list of hidden units in each HIDDEN layer e.g. [3,4,5]
        :param output_dim: output dimensions
        :param actFun_type: type of activation function (tanh, sigmoid, ReLU)
        :param reg_lambda: regularization coefficient
        :param epsilon: learning rate
        :param seed: random seed
        """
        self.input_dims = input_dims
        self.n_units = n_units
        self.output_dims = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        np.random.seed(seed)

        # Initialize weights and biases for network
        # Expand units list to include input and output dims
        self.n_all = self.n_units.copy()
        self.n_all.insert(0, self.input_dims)
        self.n_all.append(self.output_dims)

        # Number of weight matrices needed is one less than number of TOTAL layers
        # self.num_W = len(n_all) - 1
        # self.weights = {i:0 for i in range(self.num_W)}
        # self.biases = {i:0 for i in range(self.num_W)}
        # for i in range(self.num_W):
        # To initialize W_l we initialize by dims of layer l (n_all[i]) and layer l+1 (n_all[i+1])
        # self.weights[i] = np.random.randn(n_all[i],n_all[i+1]) / np.sqrt(n_all[i])
        # self.biases[i] = np.zeros((1,n_all[i+1]))

        # Initialize Layer objects for each hidden layer
        self.layers = {}
        for i, v in enumerate(self.n_all[:-1]):
            if i == len(self.n_all) - 2:
                self.layers[i] = Layer(in_dim=self.n_all[i], out_dim=self.n_all[i + 1], actFun_type=self.actFun_type,
                                       out_layer=True,seed=seed)
            else:
                self.layers[i] = Layer(in_dim=self.n_all[i], out_dim=self.n_all[i + 1], actFun_type=self.actFun_type,seed=seed)

        # Initialize dictionary to hold activations of each layer to be used in backprop later
        # self.act_cache = {i:0 for i in self.layers.keys()}

    def feedforward(self, X):
        """
        Feedforward computes the forward pass through n-layer neural network
        :param X: input data
        :return: probabilities for class 0 and 1
        """

        # Compute layer 1 affine transformation and activation function
        self.layers[0].feedforward(X)
        for key in list(self.layers.keys())[1:-1]:
            self.layers[key].feedforward(self.layers[key - 1].activ)

        self.layers[len(self.n_all) - 2].feedforward(self.layers[len(self.n_all) - 3].activ)
        self.probs = self.layers[len(self.n_all) - 2].probs

        # Compute softmax probabilities on output from last hidden layer
        # out = np.dot(self.layers[self.num_W-2].activ,self.weights[self.num_W-1]) + self.biases[self.num_W-1]
        # out = out - np.max(out,axis=1,keepdims=True)
        # self.probs = np.exp(out)/np.sum(np.exp(out),axis=1,keepdims=True)

        return self.probs

    def calculate_loss(self, X, y):
        """
        Calculate the loss for prediction
        :param X: input data
        :param y: input data labels
        :return: loss for prediction
        """
        num_samps = len(X)
        self.feedforward(X)

        data_loss = y * np.log(self.probs)
        data_loss = np.sum(data_loss, axis=1)
        data_loss = np.sum(data_loss)
        data_loss = -1 * data_loss

        sum_weights = 0
        for key in self.layers.keys():
            w = np.sum(np.square(self.layers[key].weights))
            sum_weights += w

        data_loss += self.reg_lambda / 2 * sum_weights
        return (1. / num_samps) * data_loss

    def backprop(self, X, y):
        """
        Backprop calculates the backpropagation of error via gradients through all parameters of the network
        :param X: input data
        :param y: input data labels
        :return:
        """

        num_examples = len(X)
        norm = -1
        del1 = y - self.probs
        del1 = norm * del1

        # include update in fit model
        self.ddw = {}
        self.ddb = {}
        self.ddx = {}

        for i in reversed(range(len(self.n_all) - 1)):
            if i == len(self.n_all) - 2:
                self.ddw[i] = np.dot(self.layers[i - 1].activ.T, del1)
                self.ddb[i] = np.sum(del1, axis=0)
                self.ddx[i] = np.dot(del1, self.layers[i].weights.T)
            elif i == 0:
                ddx, ddw, ddb = self.layers[0].backprop(act_prev=X, ddx_prev=self.ddx[i + 1])
                self.ddw[i] = ddw
                self.ddx[i] = ddx
                self.ddb[i] = ddb
            else:
                ddx, ddw, ddb = self.layers[i].backprop(act_prev=self.layers[i - 1].activ, ddx_prev=self.ddx[i + 1])
                self.ddw[i] = ddw
                self.ddx[i] = ddx
                self.ddb[i] = ddb

    def fit_model(self, X, y, num_passes=20000, print_loss=True):
        """
        fit_model uses forward pass and backwards pass to train the model
        :param X: intput data
        :param y: input data labels
        :param num_passes: number of iterations
        :param print_loss: Print loss output statement
        """
        # self.act_cache[-1] = X
        for p in range(0, num_passes):
            self.feedforward(X)
            self.backprop(X, y)

            for key in self.layers.keys():
                self.ddw[key] += self.reg_lambda * self.layers[key].weights
                self.layers[key].weights += -self.epsilon * self.ddw[key]
                self.layers[key].bias += -self.epsilon * self.ddb[key]

            if print_loss and p % 1000 == 0:
                print("Loss after iteration %i: %f" % (p, self.calculate_loss(X, y)))

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    X, y, labels = generate_model()

    model = DeepNeuralNetwork(input_dims=2, output_dim=2, n_units=[4, 4], actFun_type='tanh', epsilon=0.01,seed=12)
    model.fit_model(X, y)

    #####Check classification accuracy####
    preds = model.predict(X)
    acc = np.sum(labels == preds) / len(X)
    print(acc)
    ######################################
    model.visualize_decision_boundary(X, labels)


if __name__ == "__main__":
    main()
