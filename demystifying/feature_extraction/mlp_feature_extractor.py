from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .. import relevance_propagation as relprop
from .feature_extractor import FeatureExtractor
from ..postprocessing import PerFrameImportancePostProcessor

logger = logging.getLogger("mlp")


class MlpFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 name="MLP",
                 activation=relprop.relu,
                 randomize=True,
                 supervised=True,
                 one_vs_rest=False,
                 per_frame_importance_outfile=None,
                 per_frame_importance_samples=None,
                 per_frame_importance_labels=None,
                 classifier_kwargs={},
                 **kwargs):
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=supervised,
                                  **kwargs)
        self.backend = "scikit-learn"  # Only available option for now, more to come probably
        if activation not in [relprop.relu, relprop.logistic_sigmoid]:
            Exception("Relevance propagation currently only supported for relu or logistic")
        self.activation = activation
        self.randomize = randomize
        self.classifier_kwargs = classifier_kwargs.copy()
        if classifier_kwargs.get('activation', None) is not None and \
                classifier_kwargs.get('activation') != self.activation:
            logger.warn("Conflicting activation properiies. '%s' will be overwritten with '%s'",
                        classifier_kwargs.get('activation'),
                        self.activation)
        self.classifier_kwargs['activation'] = self.activation
        if not self.randomize:
            self.classifier_kwargs['random_state'] = 89274
        self.frame_importances = None
        self.per_frame_importance_outfile = per_frame_importance_outfile
        self.per_frame_importance_samples = per_frame_importance_samples
        self.per_frame_importance_labels = per_frame_importance_labels
        if self.use_regression:
            self.one_vs_rest = False
        else:
            self.one_vs_rest = one_vs_rest

        logger.debug("Initializing MLP with the following parameters:"
                     " activation function %s, randomize %s, classifier_kwargs %s,"
                     " per_frame_importance_outfile %s, backend %s, per_frame_importance_samples %s, one_vs_rest %s",
                     activation, randomize, classifier_kwargs, per_frame_importance_outfile, self.backend,
                     None if per_frame_importance_samples is None else per_frame_importance_samples.shape,
                     self.one_vs_rest)

    def _train_one_vs_rest(self, data, labels):
        n_clusters = labels.shape[1]
        n_points = data.shape[0]

        classifiers = []

        for i_cluster in range(n_clusters):
            classifiers.append(self._create_classifier())
            binary_labels = np.zeros((n_points, 2))
            binary_labels[labels[:, i_cluster] == 1, 0] = 1
            binary_labels[labels[:, i_cluster] != 1, 1] = 1
            classifiers[i_cluster].fit(data, binary_labels)

        return classifiers

    def train(self, train_set, train_labels):
        """
        TODO code duplication below for on_vs_the_rest logic, refactor with KL and RF into common superclass
        :param train_set:
        :param train_labels:
        :return:
        """
        # Construct and train classifier
        logger.debug("Training %s with %s samples and %s features ...", self.name, train_set.shape[0],
                     train_set.shape[1])
        if self.one_vs_rest:
            return self._train_one_vs_rest(train_set, train_labels)
        else:
            classifier = self._create_classifier()
            classifier.fit(train_set, train_labels)
        return classifier

    def _normalize_relevance_per_frame(self, relevance_per_frame):
        for i in range(relevance_per_frame.shape[0]):
            # Not removing negative relevance in per frame analysis
            # ind_negative = np.where(relevance_per_frame[i, :] < 0)[0]
            # relevance_per_frame[i, ind_negative] = 0
            relevance_per_frame[i, :] = (relevance_per_frame[i, :] - np.min(relevance_per_frame[i, :])) / \
                                        (np.max(relevance_per_frame[i, :]) - np.min(relevance_per_frame[i, :]) + 1e-9)
        return relevance_per_frame

    def _perform_lrp(self, classifier, data, labels):
        nclusters = labels.shape[1] if self.supervised else 1
        nfeatures = data.shape[1]
        relevance_per_cluster = np.zeros((nfeatures, nclusters))
        per_frame_relevance = np.zeros(data.shape)
        for c_idx in range(nclusters):
            # Get all frames belonging to a cluster
            if self.supervised:
                frame_indices = labels[:, c_idx] == 1
                cluster_data = data[frame_indices]
                cluster_labels = np.zeros((len(cluster_data), nclusters))
                cluster_labels[:, c_idx] = 1  # Only look at one class at the time
            else:
                # TODO refactor to break unsupervised code out of here. Unsupervised method have no concept of clusters/labels
                cluster_labels = labels
                frame_indices = [i for i in range(len(data))]
                cluster_data = data
            if len(cluster_data) == 0:
                continue
            # Now see what makes these frames belong to that class
            # Time for LRP
            layers = self._create_layers(classifier)
            propagator = relprop.RelevancePropagator(layers)
            cluster_frame_relevance = propagator.propagate(cluster_data, cluster_labels)
            # Rescale relevance according to min and max relevance in each frame
            cluster_frame_relevance = self._normalize_relevance_per_frame(cluster_frame_relevance)
            relevance_per_cluster[:, c_idx] = cluster_frame_relevance.mean(axis=0)
            per_frame_relevance[frame_indices] += cluster_frame_relevance
        per_frame_relevance = self._normalize_relevance_per_frame(per_frame_relevance)
        return per_frame_relevance, relevance_per_cluster

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using MLP ...")
        if self.one_vs_rest:
            return self._get_feature_importance_binaryclass(classifier, data, labels)
        else:
            return self._get_feature_importance_multiclass(classifier, data, labels)

    def _get_feature_importance_binaryclass(self, classifiers, data, labels):
        n_features = data.shape[1]
        n_frames = data.shape[0]
        n_states = labels.shape[1] if len(labels.shape) > 1 else 1
        feature_importances = np.zeros((n_features, self.n_clusters))
        for i_cluster in range(n_states):
            # TODO a bit inefficent approach below where we consistenly compute LRP for all other clusters and don't use those results.
            cluster_frames = labels[:, i_cluster] == 1
            binary_labels = np.zeros((n_frames, 2))
            binary_labels[cluster_frames, 0] = 1
            binary_labels[~cluster_frames, 1] = 1
            relevance_per_frame, relevance_per_cluster = self._perform_lrp(classifiers[i_cluster], data, binary_labels)
            feature_importances[:, i_cluster] = relevance_per_cluster[:, 0]
            if self.per_frame_importance_outfile is not None:
                cluster_frame_importances, other_labels = self._compute_frame_relevance(classifiers[i_cluster],
                                                                                        relevance_per_frame,
                                                                                        data,
                                                                                        labels)
                if self.frame_importances is None:
                    self.frame_importances = np.zeros((len(other_labels), cluster_frame_importances.shape[1]))
                other_cluster_frames = other_labels[:, 0] == 1
                if len(other_labels[other_cluster_frames]) == 0:
                    # No frames in this state, just move on
                    continue
                nclusters_per_frame = other_labels[other_cluster_frames].sum(axis=1)[:, np.newaxis]
                self.frame_importances[other_cluster_frames, :] += cluster_frame_importances[
                                                                       other_cluster_frames] / nclusters_per_frame
        return feature_importances

    def _get_feature_importance_multiclass(self, classifier, data, labels):
        relevance_per_frame, relevance_per_cluster = self._perform_lrp(classifier, data, labels)

        if self.per_frame_importance_outfile is not None:
            frame_importances, _ = self._compute_frame_relevance(classifier, relevance_per_frame, data, labels)
            self.frame_importances = frame_importances if self.frame_importances is None else self.frame_importances + frame_importances
        return relevance_per_cluster

    def _compute_frame_relevance(self, classifier, relevance_per_frame, data, labels):
        if self.per_frame_importance_samples is not None:
            if self.indices_for_filtering is None:
                other_samples = self.per_frame_importance_samples
            else:
                other_samples = self.per_frame_importance_samples[:, self.indices_for_filtering]
            if self.per_frame_importance_labels is None:
                other_labels = classifier.predict(other_samples)
            else:
                other_labels = self.per_frame_importance_labels
            other_samples = self.scaler.transform(other_samples)
            frame_relevance, _ = self._perform_lrp(classifier, other_samples, other_labels)
        else:
            logger.info("Using same trajectory for per frame importance as was used for training.")
            if self.n_splits != 1:
                logger.error(
                    "Cannot average frame importance to outfile if n_splits != 1. n_splits is now set to %s",
                    self.n_splits)
            if self.shuffle_datasets:
                logger.error("Data set has been shuffled, per frame importance will not be properly mapped")
            frame_relevance = relevance_per_frame
            other_labels = labels
        # for every feature in every frame...
        frame_importances = np.zeros(
            (data if self.per_frame_importance_samples is None else self.per_frame_importance_samples).shape) - 1
        if self.indices_for_filtering is not None:
            frame_importances[:, self.indices_for_filtering] = 0
        niters = self.n_iterations * self.n_splits
        for frame_idx, rel in enumerate(frame_relevance):
            if self.indices_for_filtering is None:
                frame_importances[frame_idx] += rel / niters
            else:
                frame_importances[frame_idx, self.indices_for_filtering] += rel / niters
        return frame_importances, other_labels

    def _create_layers(self, classifier):
        weights = classifier.coefs_
        biases = classifier.intercepts_
        layers = []
        for idx, weight in enumerate(weights):

            if idx == 0:
                l = relprop.FirstLinear(min_val=0, max_val=1, weight=weight, bias=biases[idx])
            else:
                l = relprop.layer_for_string(self.activation, weight=weight, bias=biases[idx])
            if l is None:
                raise Exception(
                    "Cannot create layer at index {} for activation function {}".format(idx, self.activation))
            layers.append(l)
            if idx < len(weights) - 1:
                # Add activation to all except output layer
                activation = relprop.layer_activation_for_string(self.activation)
                if activation is None:
                    raise Exception("Unknown activation function {}".format(self.activation))
                layers.append(activation)
            else:
                if self.backend == 'scikit-learn':
                    # For scikit implementation see  # https://stats.stackexchange.com/questions/243588/how-to-apply-softmax-as-activation-function-in-multi-layer-perceptron-in-scikit
                    # or https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
                    out_activation = relprop.layer_activation_for_string(classifier.out_activation_)
                    if out_activation is None:
                        raise Exception("Unknown activation function {}".format(self.activation))
                    layers.append(out_activation)
                else:
                    raise Exception("Unsupported MLP backend {}".format(self.backend))

        return layers

    def _create_classifier(self):
        return MLPRegressor(**self.classifier_kwargs) if self.use_regression \
            else MLPClassifier(**self.classifier_kwargs)

    def postprocessing(self, **kwargs):
        return PerFrameImportancePostProcessor(extractor=self,
                                               per_frame_importance_outfile=self.per_frame_importance_outfile,
                                               frame_importances=self.frame_importances,
                                               **kwargs)
