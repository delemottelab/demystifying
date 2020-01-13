import logging
import sys

import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import scipy.special

relu = "relu"
logistic_sigmoid = "logistic"
tanh = "tanh"
softmax = "softmax"
identity = "identity"
logger = logging.getLogger("relevance_propagation")

"""
TODO implement more activation functions or import another library for this, see for example https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/master/python/modules
Heavily inspired by http://www.heatmapping.org/tutorial/, visited 2019-05-02
"""


def layer_activation_for_string(activation):
    """
    :return: an activation function. None if not found
    """
    if activation == logistic_sigmoid:
        return LogisticSigmoid()
    elif activation == relu:
        return ReLU()
    elif activation == tanh:
        return Tanh()
    elif activation == softmax:
        return SoftMax()
    elif activation == identity:
        return Identity()
    else:
        return None


def layer_for_string(activation, weight, bias):
    kwargs = {
        'weight': weight,
        'bias': bias
    }
    if activation == logistic_sigmoid:
        return FirstLinear(min_val=0, max_val=1, **kwargs)
    elif activation == relu:
        return NextLinear(**kwargs)
    elif activation == tanh:
        return FirstLinear(min_val=-1, max_val=1, **kwargs)
    else:
        # we don't support softmax or linear now since we don't know the bounds.
        # They are often used in the output layer only though
        return None


class RelevancePropagator(object):

    def __init__(self, layers):
        self.layers = layers

    def propagate(self, X, T):
        # Reinstantiating the neural network
        network = Network(self.layers)
        Y = network.forward(X)
        # Performing relevance propagation
        D = network.relprop(Y * T)
        return D


class Network:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for l in self.layers: X = l.forward(X)
        return X

    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R


class ReLU:

    def forward(self, X):
        self.Z = X > 0
        return X * self.Z

    def relprop(self, R):
        return R


class LogisticSigmoid:

    def forward(self, X):
        return scipy.special.expit(X)

    def relprop(self, R):
        return R


class Tanh:

    def forward(self, X):
        return np.tan(X)

    def relprop(self, R):
        return R


class SoftMax:

    def forward(self, X):
        self.X = X
        self.Y = np.exp(X) / np.exp(X).sum(keepdims=True, axis=1)
        return self.Y

    def relprop(self, R):
        return R


class Identity:

    def forward(self, X):
        return X

    def relprop(self, R):
        return R


class Linear:
    def __init__(self, weight=None, bias=None):
        self.W = weight
        self.B = bias

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.W) + self.B


class FirstLinear(Linear):
    """For z-beta rule"""

    def __init__(self, min_val=None, max_val=None, **kwargs):
        Linear.__init__(self, **kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def relprop(self, R):
        # min_val, max_val = np.min(self.X, axis=0), np.max(self.X, axis=0)
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X = self.X
        L = np.zeros(X.shape) + self.min_val
        H = np.zeros(X.shape) + self.max_val
        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        # a constant just corresponds to a diagonal matrix with that constant along the diagonal
        Wt = W if isinstance(W, int) or isinstance(W, float) else W.T
        R = X * np.dot(S, Wt) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
        return R


class NextLinear(Linear):
    """For z+ rule"""

    def relprop(self, R):
        V = np.maximum(0, self.W)
        Z = np.dot(self.X, V) + 1e-9
        S = R / Z
        C = np.dot(S, V.T)
        R = self.X * C
        return R
