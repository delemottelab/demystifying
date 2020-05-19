from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from .. import utils as utils, filtering
from sklearn.model_selection import KFold
from ..postprocessing import PostProcessor

logger = logging.getLogger("Extracting features")


class FeatureExtractor(object):

    def __init__(self,
                 samples=None,
                 labels=None,
                 scaling=True,
                 filter_by_distance_cutoff=False,
                 lower_bound_distance_cutoff=filtering.lower_bound_distance_cutoff_default,
                 upper_bound_distance_cutoff=filtering.upper_bound_distance_cutoff_default,
                 use_inverse_distances=False,
                 n_splits=3,
                 n_iterations=5,
                 name='FeatureExtractor',
                 error_limit=None,
                 supervised=True,
                 remove_outliers=False,
                 label_names=None,
                 use_regression=None,
                 shuffle_datasets=True):
        if samples is None:
            raise ValueError("Samples cannot be None")
        self.samples = samples
        self.supervised = supervised
        if use_regression is None:
            # If unsupervised or there are decimals in the labels, then we assume it is about regression
            use_regression = abs(np.sum(np.round(labels) - labels)) > 1e-8 if self.supervised else True
        self.use_regression = use_regression
        self.labels, self.cluster_indices = utils.format_labels(labels, use_regression=use_regression)
        self.mixed_classes = not use_regression and self.labels is not None and np.any(
            self.labels.sum(axis=1).max() != 1)
        self.n_clusters = 0 if self.labels is None else self.labels.shape[1]
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.scaling = scaling
        self.filter_by_distance_cutoff = filter_by_distance_cutoff
        self.name = name
        if error_limit is None:
            # We expect the error to be below random guessing (assuming balanced datasets)
            error_limit = 100 if self.use_regression else 100 * (1 - 1. / self.n_clusters) + 1e-4
        self.error_limit = error_limit
        self.use_inverse_distances = use_inverse_distances
        self.lower_bound_distance_cutoff = lower_bound_distance_cutoff
        self.upper_bound_distance_cutoff = upper_bound_distance_cutoff
        self.remove_outliers = remove_outliers
        self.shuffle_datasets = shuffle_datasets
        self.feature_importance = None
        self.std_feature_importance = None
        self.test_set_errors = None
        self.indices_for_filtering = None
        self.scaler = None
        self.label_names = label_names
        logger.debug("Initializing superclass FeatureExctractor '%s' with the following parameters: "
                     " n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, lower_bound_distance_cutoff %s, "
                     " upper_bound_distance_cutoff %s, remove_outliers %s, use_inverse_distances %s, shuffle_datasets %s"
                     "use_regression %s",
                     name, n_splits, n_iterations, scaling, filter_by_distance_cutoff, lower_bound_distance_cutoff,
                     upper_bound_distance_cutoff, remove_outliers, use_inverse_distances, shuffle_datasets,
                     use_regression)

    def split_train_test(self):
        """
        Split the data into n_splits training and test sets
        """
        if self.n_splits < 2:
            logger.debug("Using all data in training and validation sets")
            all_indices = np.empty((1, len(self.samples)))
            for i in range(len(self.samples)):
                all_indices[0, i] = i
            all_indices = all_indices.astype(int)
            return all_indices, all_indices

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle_datasets)

        train_inds = []
        test_inds = []

        for train_ind, test_ind in kf.split(self.samples):
            train_inds.append(train_ind)
            test_inds.append(test_ind)
        return train_inds, test_inds

    def get_train_test_set(self, train_ind, test_ind):
        """
        Get the train and test set given their sample/label indices
        """
        train_set = self.samples[train_ind, :]
        test_set = self.samples[test_ind, :]

        test_labels = None if self.labels is None else self.labels[test_ind, :]
        train_labels = None if self.labels is None else self.labels[train_ind, :]

        return train_set, test_set, train_labels, test_labels

    def train(self, train_set, train_labels):
        pass

    def _train_unsupervised_methods_per_class(self, train_set, train_labels):
        # Compute unsupervised learning per cluster
        return np.array([
            self.train(train_set[train_labels[:, cl] == 1], None)
            for cl in range(self.n_clusters)
        ])

    def get_feature_importance(self, model, samples, labels):
        pass

    def _get_feature_importance_for_unsupervised_per_class(self, model, samples, labels):
        imps = np.empty((samples.shape[1], len(model)))
        for cl in range(self.n_clusters):
            imps[:, cl] = self.get_feature_importance(model[cl], samples[labels[:, cl] == 1], None)
        return imps

    def extract_features(self):

        logger.info("Performing feature extraction with %s on data of shape %s", self.name, self.samples.shape)

        # Create a list of feature indices
        # This is needed when filtering is applied and re-mapping is further used
        original_samples = np.copy(self.samples)
        if self.filter_by_distance_cutoff:
            self.samples, self.indices_for_filtering = filtering.filter_by_distance_cutoff(self.samples,
                                                                                           lower_bound_cutoff=self.lower_bound_distance_cutoff,
                                                                                           upper_bound_cutoff=self.upper_bound_distance_cutoff,
                                                                                           inverse_distances=self.use_inverse_distances)

        if self.scaling:
            # Note that we must use the same scalers for all data
            # It is important for some methods (relevance propagation in NN) that all data is scaled between 0 and 1
            self.samples, self.scaler = utils.scale(self.samples, remove_outliers=self.remove_outliers)

        train_inds, test_inds = self.split_train_test()
        errors = np.zeros(self.n_splits * self.n_iterations)

        feats = []

        for i_split in range(self.n_splits):

            for i_iter in range(self.n_iterations):

                train_set, test_set, train_labels, test_labels = self.get_train_test_set(train_inds[i_split],
                                                                                         test_inds[i_split])

                # Train model
                model = self.train(train_set, train_labels)

                if self.supervised and hasattr(model, "predict"):
                    # Test classifier
                    error = utils.check_for_overfit(test_set, test_labels, model)
                    errors[i_split * self.n_iterations + i_iter] = error

                    do_compute_importance = errors[i_split * self.n_iterations + i_iter] < self.error_limit

                else:
                    do_compute_importance = True

                if do_compute_importance:
                    # Get features importance
                    feature_importance = self.get_feature_importance(model, train_set, train_labels)
                    feats.append(feature_importance)
                else:
                    logger.warn("At iteration %s of %s error %s is too high - not computing feature importance",
                                i_split * self.n_iterations + i_iter + 1, self.n_splits * self.n_iterations, error)

        feats = np.asarray(feats)
        self._on_all_features_extracted(feats, errors, original_samples.shape[1])
        self.samples = np.copy(original_samples)  # TODO why do we do this?
        logger.debug("Done with feature extraction for %s", self.name)
        return self

    def _on_all_features_extracted(self, feats, errors, n_features):
        std_feats = np.std(feats, axis=0)
        feats = np.mean(feats, axis=0)

        if len(feats.shape) == 1 and len(std_feats.shape) == 1:
            feats = feats.reshape((feats.shape[0], 1))
            std_feats = std_feats.reshape((std_feats.shape[0], 1))

        if self.filter_by_distance_cutoff:
            # Remapping features if filtering was applied
            # If no filtering was applied, return feats and std_feats
            feats, std_feats = filtering.remap_after_filtering(feats, std_feats, n_features,
                                                               self.indices_for_filtering)

        self.feature_importance = feats
        self.std_feature_importance = std_feats
        self.test_set_errors = errors

    def postprocessing(self, **kwargs):
        return PostProcessor(extractor=self, **kwargs)
