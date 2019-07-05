from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import sklearn.neural_network

from .mlp_feature_extractor import MlpFeatureExtractor
from .. import relevance_propagation as relprop

logger = logging.getLogger("mlp_ae")


class MlpAeFeatureExtractor(MlpFeatureExtractor):

    def __init__(self,
                 name="AE",
                 activation=relprop.logistic_sigmoid,
                 use_reconstruction_for_lrp=False,
                 **kwargs):
        MlpFeatureExtractor.__init__(self, name=name, supervised=False, activation=activation, **kwargs)
        self.use_reconstruction_for_lrp = use_reconstruction_for_lrp
        logger.debug("Initializing MLP AE with the following parameters:"
                     " use_reconstruction_for_lrp %s", use_reconstruction_for_lrp)

    def train(self, train_set, train_labels):
        logger.debug("Training %s with %s samples and %s features ...", self.name, train_set.shape[0],
                     train_set.shape[1])
        classifier_kwargs = self.classifier_kwargs
        classifier_kwargs['hidden_layer_sizes'] = list(classifier_kwargs['hidden_layer_sizes']) + [train_set.shape[1]]
        classifier = sklearn.neural_network.MLPRegressor(**classifier_kwargs)
        classifier.fit(train_set, train_set)  # note same output as input
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using MLP Autoencoder ...")
        target_values = classifier.predict(data) if self.use_reconstruction_for_lrp else data
        res = MlpFeatureExtractor.get_feature_importance(self, classifier, data, target_values)
        return res.mean(axis=1)
