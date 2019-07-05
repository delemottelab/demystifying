from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.stats import entropy
from .feature_extractor import FeatureExtractor

logger = logging.getLogger("KL divergence")


class KLFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 name="KL",
                 cluster_split_method="one_vs_rest",
                 bin_width=None,
                 **kwargs):
        FeatureExtractor.__init__(self,
                                  name=name,
                                  **kwargs)

        self.bin_width = bin_width
        if bin_width is None:
            logger.debug('Using standard deviation of each feature as bin size.')
        self.feature_importances = None
        self.cluster_split_method = cluster_split_method
        logger.debug("Initializing KL with the following parameters: bin_width %s, cluster_split_method %s",
                     bin_width, cluster_split_method)

    def train(self, data, labels):
        logger.debug("Training KL with %s samples and %s features ...", data.shape[0], data.shape[1])
        if self.cluster_split_method == "one_vs_rest":
            self._train_one_vs_rest(data, labels)
        elif self.cluster_split_method == "one_vs_one":
            self._train_one_vs_one(data, labels)
        else:
            raise Exception("Unsupported split method: {}".format(self.cluster_split_method))

    def _KL_divergence(self, x, y):
        """
        Compute Kullback-Leibler divergence
        """
        n_features = x.shape[1]

        DKL = np.zeros(n_features)
        if self.bin_width is not None:
            tmp_bin_width = self.bin_width

        for i_feature in range(n_features):
            xy = np.concatenate((x[:, i_feature], y[:, i_feature]))
            bin_min = np.min(xy)
            bin_max = np.max(xy)

            if self.bin_width is None:
                tmp_bin_width = np.std(x[:, i_feature])
                if tmp_bin_width == 0:
                    tmp_bin_width = 0.1  # Set arbitrary bin width if zero
            else:
                tmp_bin_width = self.bin_width

            if tmp_bin_width >= (bin_max - bin_min):
                DKL[i_feature] = 0
            else:
                bin_n = int((bin_max - bin_min) / tmp_bin_width)
                x_prob = np.histogram(x[:, i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                y_prob = np.histogram(y[:, i_feature], bins=bin_n, range=(bin_min, bin_max), density=True)[0] + 1e-9
                DKL[i_feature] = 0.5 * (
                        entropy(x_prob, y_prob) + entropy(y_prob, x_prob))  # An alternative is to use max
        return DKL

    def get_feature_importance(self, model, data, labels):
        """
        Get the feature importance of KL divergence by comparing each cluster to all other clusters
        """
        logger.debug("Extracting feature importance using KL ...")
        return self.feature_importances

    def _train_one_vs_rest(self, data, labels):
        n_clusters = labels.shape[1]
        n_features = data.shape[1]

        self.feature_importances = np.zeros((n_features, n_clusters))
        for i_cluster in range(n_clusters):
            data_cluster = data[labels[:, i_cluster] == 1, :]
            data_rest = data[labels[:, i_cluster] == 0, :]
            self.feature_importances[:, i_cluster] = self._KL_divergence(data_cluster, data_rest)
        return self

    def _train_one_vs_one(self, data, labels):
        raise NotImplementedError()
