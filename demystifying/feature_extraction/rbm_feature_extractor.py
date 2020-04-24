import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

from .. import relevance_propagation as relprop
from .feature_extractor import FeatureExtractor
from sklearn.neural_network import BernoulliRBM
from .. import utils
import scipy

logger = logging.getLogger("rbm")


class RbmFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 supervised=False,
                 name="RBM",
                 randomize=True,
                 relevance_method="from_lrp",
                 variance_cutoff='auto',
                 classifier_kwargs={
                     'n_components': 1,
                 },
                 **kwargs):

        FeatureExtractor.__init__(self,
                                  supervised=supervised,
                                  name=name,
                                  **kwargs)
        self.relevance_method = relevance_method
        self.variance_cutoff = variance_cutoff
        self.randomize = randomize
        self.classifier_kwargs = classifier_kwargs.copy()
        if not self.randomize:
            self.classifier_kwargs['random_state'] = 89274
        logger.debug("Initializing RBM with the following parameters: "
                     " randomize %s, relevance_method %s, relevance_method %s, variance_cutoff %s,"
                     " classifier_kwargs %s",
                     randomize, relevance_method, relevance_method, variance_cutoff, classifier_kwargs)

    def train(self, train_set, train_labels):
        if self.supervised and train_labels is not None:
            return self._train_unsupervised_methods_per_class(train_set, train_labels)
        else:
            model = BernoulliRBM(**self.classifier_kwargs)
            model.fit(train_set)
            return model

    def get_feature_importance(self, model, samples, labels):
        if self.supervised and labels is not None:
            return self._get_feature_importance_for_unsupervised_per_class(model, samples, labels)
        logger.debug("RBM psuedo-loglikelihood: " + str(model.score_samples(samples).mean()))
        if self.relevance_method == "from_lrp":
            nframes, nfeatures = samples.shape

            labels_propagation = model.transform(samples)  # same as perfect classification

            # Calculate relevance
            # see https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html
            layers = self._create_layers(model)

            propagator = relprop.RelevancePropagator(layers)
            relevance = propagator.propagate(samples, labels_propagation)

            # Rescale relevance according to min and max relevance in each frame
            logger.debug("Rescaling feature importance extracted using RBM in each frame between min and max ...")

            for i in range(relevance.shape[0]):
                ind_negative = np.where(relevance[i, :] < 0)[0]
                relevance[i, ind_negative] = 0
                relevance[i, :] = (relevance[i, :] - np.min(relevance[i, :])) / (
                        np.max(relevance[i, :]) - np.min(relevance[i, :]) + 1e-9)

            if self.supervised:
                return relevance
            # Average relevance per cluster
            nclusters = labels.shape[1]

            result = np.zeros((nfeatures, nclusters))
            frames_per_cluster = np.zeros((nclusters))

            for frame_idx, frame in enumerate(labels):
                cluster_idx = labels[frame_idx].argmax()
                frames_per_cluster[cluster_idx] += 1

            for frame_idx, rel in enumerate(relevance):
                cluster_idx = labels[frame_idx].argmax()
                result[:, cluster_idx] += rel / frames_per_cluster[cluster_idx]

            return result

        elif self.relevance_method == "from_components":

            # Extract components and compute their variance
            components = model.components_
            projection = scipy.special.expit(np.matmul(samples, components.T))
            components_var = projection.var(axis=0)

            # Sort components according to their variance
            ind_components_var_sorted = np.argsort(-components_var)
            components_var_sorted = components_var[ind_components_var_sorted]
            components_var_sorted /= components_var_sorted.sum()
            components_sorted = components[ind_components_var_sorted, :]

            return utils.compute_feature_importance_from_components(components_var_sorted,
                                                                    components_sorted,
                                                                    self.variance_cutoff)
        else:
            raise Exception("Method {} not supported".format(self.relevance_method))

    def _create_layers(self, classifier):
        return [relprop.FirstLinear(min_val=0, max_val=1, weight=classifier.components_.T,
                                    bias=classifier.intercept_hidden_),
                relprop.LogisticSigmoid()
                ]
