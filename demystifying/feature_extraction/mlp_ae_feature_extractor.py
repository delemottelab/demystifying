from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import sklearn.neural_network

from .mlp_feature_extractor import MlpFeatureExtractor
from .. import relevance_propagation as relprop

logger = logging.getLogger("mlp_ae")


class MlpAeFeatureExtractor(MlpFeatureExtractor):

    def __init__(self,
                 supervised=False,
                 name="AE",
                 activation=relprop.logistic_sigmoid,
                 use_reconstruction_for_lrp=False,
                 **kwargs):
        MlpFeatureExtractor.__init__(self, name=name, supervised=supervised, activation=activation, **kwargs)
        self.use_reconstruction_for_lrp = use_reconstruction_for_lrp
        logger.debug("Initializing MLP AE with the following parameters:"
                     " use_reconstruction_for_lrp %s", use_reconstruction_for_lrp)
        # If no hidden layers are specific or it is set to auto we will set this hyperparameter on the fly
        self._automatic_layer_size = self.classifier_kwargs.get("hidden_layer_sizes", "auto") in [None, "auto"]

    def train(self, train_set, train_labels):
        if self.supervised and train_labels is not None:
            return self._train_unsupervised_methods_per_class(train_set, train_labels)
        else:
            classifier_kwargs = self.classifier_kwargs
            if self._automatic_layer_size:
                # Use a rule of thumb with decreasing the layer half at the time
                layer_sizes = []
                nfeatures = train_set.shape[1]
                n_hidden_nodes = nfeatures / 2
                while n_hidden_nodes > 0 and n_hidden_nodes > nfeatures / 8:
                    layer_sizes.append(int(n_hidden_nodes + 0.5))
                    n_hidden_nodes /= 2
            else:
                layer_sizes = list(classifier_kwargs['hidden_layer_sizes'])
            # add output layer same size as features
            layer_sizes += [train_set.shape[1]]
            classifier_kwargs['hidden_layer_sizes'] = layer_sizes
            model = sklearn.neural_network.MLPRegressor(**classifier_kwargs)
            model.fit(train_set, train_set)  # note same output as input
            return model

    def get_feature_importance(self, model, samples, labels):
        if self.supervised and labels is not None:
            return self._get_feature_importance_for_unsupervised_per_class(model, samples, labels)
        logger.debug("Extracting feature importance using MLP Autoencoder ...")
        target_values = model.predict(samples) if self.use_reconstruction_for_lrp else samples
        res = MlpFeatureExtractor.get_feature_importance(self, model, samples, target_values)
        return res.mean(axis=1)
