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
    return X, y

class Layer(object):
    """
    This class forward and backward passes through a layer
    """

    def __init__(self,weights,bias,actFun_type='tanh'):
        """
        :param weights: weight matrix for layer
        :param bias: bias vector for layer
        :param actFun_type: activation function type
        """

        self.weights = weights
        self.bias = bias
        self.actFun_type = actFun_type

    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        Function was Recycled from three_layer_neural_network.py
        """

        if type == 'tanh':
            # tanh(x) = sinh(x)/cosh(x) = (e^2x - 1)/(e^2x+1)
            act = (np.exp(2*z)-1)/(np.exp(2*z)+1)
        elif type == 'sigmoid':
            act = 1/(1+np.exp(-z))
        elif type == 'ReLU':
            act = np.maximum(0,z)
        return act

    def diff_actFun(self, z, type):
        """
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        Function was Recycled from three_layer_neural_network.py
        """

        if type == 'tanh':
            num = 4*np.exp(2*z)
            den = (np.exp(2*z)+1)**2
            diff_act = num/den
        elif type == 'sigmoid':
            num = np.exp(-z)
            den = (1+np.exp(-z))**2
            diff_act = num/den
        elif type == 'ReLU':
            diff_act = (z>0)*1
        return diff_act

    def feedforward(self,X):
        """
        Computes forward pass for the layer
        :param X: input data
        :return: output of forward pass
        """

        # Perform Affine transformation first
        self.aff = np.dot(X,self.weights) + self.bias
        self.act = self.actFun(self.aff,self.actFun_type)

        return self.act

    def backprop(self,X):
        """
        Computes backpropagation for layer
        :return: ddx ddw, partials w.r.t input and weights
        """
        ddw = np.dot(self.diff_actFun(self.aff,self.actFun_type).T,X).T
        ddx = np.dot(self.diff_actFun(self.aff,self.actFun_type),self.weights.T)
        return ddx,ddw


class DeepNeuralNetwork(object):
    """
    Neural network class
    """

    def __init__(self, input_dims, n_units=3, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        """
        :param input_dims: input dimension
        :param n_units: list of hidden units in each HIDDEN layer e.g. [3,4,5]
        :param output_dim: output dimensions
        :param actFun_type: type of activation function (tanh, sigmoid, ReLU)
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        """
        self.input_dims = input_dims
        self.n_units = n_units
        self.output_dims = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # Initialize weights and biases for network
        # Expand units list to include input and output dims
        n_all = self.n_units.copy()
        n_all.insert(0,self.input_dims)
        n_all.append(self.output_dims)

        # Number of weight matrices needed is one less than number of TOTAL layers
        self.num_W = len(n_all) - 1
        self.weights = {i:0 for i in range(num_W)}
        self.biases = {i:0 for i in range(num_W)}
        for i in range(self.num_W):
            # To initialize W_l we initialize by dims of layer l (n_all[i]) and layer l+1 (n_all[i+1])
            self.weight[i] = np.random.randn(n_all[i],n_all[i+1]) / np.sqrt(n_all[i])
            self.bias[i] = np.zeros((1,n_all[i+1]))

        # Initialize Layer objects for each hidden layer
        self.layers = {}
        for i in range(self.num_W-1):
            self.layers[i] = Layer(weights=self.weights[i],bias=self.biases[i],actFun_type=self.actFun_type)

    def feedforward(self,X):
        """
        Feedforward computes the forward pass through n-layer neural network
        :param X: input data
        :return: probabilities for class 0 and 1
        """

        # Compute layer 1 affine transformation and activation function
        out = self.layers[0].feedforward(X)
        for key in list(self.layers.keys())[1:]:
            out = self.layers[key].feedforward(out)

        # Compute softmax probabilities on output from last hidden layer
        out = np.dot(out,self.weights[self.num_W-1])+self.biases[num_W-1]
        self.probs = np.exp(out)/np.sum(np.exp(out),axis=1)[:,np.newaxis]

        return self.probs

    def calculate_loss(self,X,y):
        """
        Calculate the loss for prediction
        :param X: input data
        :param y: input data labels
        :return: loss for prediction
        """

    def backprop(self,X,y):
        """
        Backprop calculates the backpropagation of error via gradients through all parameters of the network
        :param X: input data
        :param y: input data labels
        :return: backprop errors
        """
