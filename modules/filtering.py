from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

logger = logging.getLogger("filtering")
lower_bound_distance_cutoff_default = 0.5
upper_bound_distance_cutoff_default = 0.7


def filter_by_distance_cutoff(data,
                              lower_bound_cutoff=lower_bound_distance_cutoff_default,
                              upper_bound_cutoff=upper_bound_distance_cutoff_default,
                              inverse_distances=True):
    """
    Contact cutoff based filtering
    """

    number_of_features = data.shape[1]
    logger.info("Number of features before distance cutoff based filtering is %s", number_of_features)

    if inverse_distances:
        data = 1 / data

    data_filtered_ind = []
    for i in range(data.shape[1]):
        data_min = np.min(data[:, i])
        data_max = np.max(data[:, i])
        if data_min <= lower_bound_cutoff and data_max >= upper_bound_cutoff:
            data_filtered_ind.append(i)

    logger.info("Number of features after distance cutoff based filtering is %s", len(data_filtered_ind))

    data_filtered = data[:, data_filtered_ind]
    indices_for_filtering = np.arange(0, data.shape[1], 1)
    indices_for_filtering = indices_for_filtering[data_filtered_ind]

    if inverse_distances:
        data_filtered = 1 / data_filtered

    return data_filtered, indices_for_filtering


def remap_after_filtering(feats, std_feats, n_features, res_indices_for_filtering):
    """
    After filtering remaps features to the matrix with initial dimensions
    """

    n_clusters_for_output = feats.shape[1]

    feats_remapped = (-1) * np.ones((n_features, n_clusters_for_output))
    feats_remapped[res_indices_for_filtering, :] = feats

    if std_feats is None:
        std_feats_remapped = None
    else:
        std_feats_remapped = (-1) * np.ones((n_features, n_clusters_for_output))
        std_feats_remapped[res_indices_for_filtering, :] = std_feats

    return feats_remapped, std_feats_remapped


def filter_feature_importance(relevances, std_relevances):
    """
    Filter feature importances based on significance
    Return filtered residue feature importances above median
    """
    logger.info("Filtering feature importances by median ...")

    n_states = relevances.shape[1]

    # indices of residues pairs which were not filtered during features filtering
    indices_not_filtered = np.where(relevances[:, 0] >= 0)[0]

    for i in range(n_states):
        global_median = np.median(relevances[indices_not_filtered, i])

        # Identify insignificant features
        ind_below_median = np.where(relevances[indices_not_filtered, i] <= global_median)[0]
        # Remove insignificant features
        ind = indices_not_filtered[ind_below_median]
        relevances[ind, i] = 0
        std_relevances[ind, i] = 0

    return relevances, std_relevances
