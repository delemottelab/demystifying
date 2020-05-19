import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from .. import relevance_propagation as relprop
from .mlp_feature_extractor import MlpFeatureExtractor
import scipy.special

logger = logging.getLogger("elm")


class ElmFeatureExtractor(MlpFeatureExtractor):

    def __init__(self,
                 name="ELM",
                 **kwargs):
        MlpFeatureExtractor.__init__(self,
                                     name=name,
                                     **kwargs)

    def train(self, train_set, train_labels):
        logger.debug("Training ELM with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        elm = SingleLayerELMClassifier(**self.classifier_kwargs)
        elm.fit(train_set, train_labels)
        return elm


class SingleLayerELMClassifier(object):
    def __init__(self, hidden_layer_sizes=(1000), activation=relprop.relu, alpha=1):
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes, )
        if len(hidden_layer_sizes) != 1:
            raise Exception("ELM currently only support one layer")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.coefs_ = None
        self.intercepts_ = None
        self.alpha = alpha  # regularization constant
        self.out_activation_ = "identity"

    def fit(self, x, t):
        (N, n) = x.shape
        W1 = self._random_matrix(n)
        b1 = self._random_matrix(1)
        H = self._g_ELM(np.matmul(x, W1) + b1)
        W2 = np.matmul(self._pseudo_inverse(H), t)
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, np.zeros((1, t.shape[1]))]

    def _random_matrix(self, x):
        # return np.random.rand(x, self.hidden_layer_sizes[0])
        return np.random.normal(0, 0.25, (x, self.hidden_layer_sizes[0]))

    def _g_ELM(self, x):
        if self.activation == relprop.relu:
            Z = x > 0
            return x * Z
        elif self.activation == relprop.logistic_sigmoid:
            return scipy.special.expit(x)
        elif self.activation == relprop.tanh:
            return np.tanh(x)
        elif self.activation == relprop.softmax:
            return np.exp(x) / np.exp(x).sum(keepdims=True, axis=1)
        elif self.activation == relprop.identity:
            return x
        else:
            raise Exception("Activation {} function not supported".format(self.activation))

    def _pseudo_inverse(self, x):
        # see eq 3.17 in bishop
        try:
            inner = np.matmul(x.T, x)
            if self.alpha is not None:
                tikonov_matrix = np.eye(x.shape[1]) * self.alpha
                inner += np.matmul(tikonov_matrix.T, tikonov_matrix)
            inv = np.linalg.inv(inner)
            return np.matmul(inv, x.T)
        except np.linalg.linalg.LinAlgError as ex:
            logger.debug("inner is a singular matrix")
            # Moore Penrose inverse rule, see paper on ELM
            inner = np.matmul(x, x.T)
            if self.alpha is not None:
                tikonov_matrix = np.eye(x.shape[0]) * self.alpha
                inner += np.matmul(tikonov_matrix, tikonov_matrix.T)
            inv = np.linalg.inv(inner)
            return np.matmul(x.T, inv)

    def predict(self, x):
        H = self._g_ELM(np.matmul(x, self.coefs_[0]) + self.intercepts_[0])
        t = np.matmul(H, self.coefs_[1])
        for row_idx, row in enumerate(t):
            c_idx = row.argmax()
            t[row_idx, :] = 0
            t[row_idx, c_idx] = 1

        return t
