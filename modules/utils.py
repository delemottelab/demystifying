from __future__ import absolute_import, division, print_function

import collections
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.spatial.distance import squareform
from biopandas.pdb import PandasPdb
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("utils")


def vectorize(data):
    """
    Vectorizes the input
    """
    if (len(data.shape)) == 3 and (data.shape[1] == data.shape[2]):
        data_vect = []
        for i in range(data.shape[0]):
            data_vect.append(squareform(data[i, :, :]))
        data_vect = np.asarray(data_vect)
    elif (len(data.shape)) == 2:
        data_vect = data
    else:
        raise Exception("The input array has wrong dimensionality")
    return data_vect


def keep_datapoints(data, clustering, points_to_keep=[]):
    """
    Keeps selected datapoints in a sample (used when clustering is not clean)
    """
    if len(points_to_keep) == 0:
        data_keep = data
        clustering_keep = clustering
    else:
        logger.info("Discarding points ...")
        logger.info("Number of points before discarding is %s", data.shape[0])
        points_to_keep = np.asarray(points_to_keep)
        for i in range(len(points_to_keep)):
            if i == 0:
                data_keep = data[points_to_keep[i, 0]:points_to_keep[i, 1]]
                clustering_keep = clustering[points_to_keep[i, 0]:points_to_keep[i, 1]]
            else:
                data_keep = np.vstack((data_keep, data[points_to_keep[i, 0]:points_to_keep[i, 1]]))
                clustering_keep = np.concatenate(
                    (clustering_keep, clustering[points_to_keep[i, 0]:points_to_keep[i, 1]]))
        logger.info("Number of points after discarding is %s", data_keep.shape[0])

    return data_keep, clustering_keep


def scale(data, remove_outliers=False):
    """
    Scales the input and removes outliers
    """
    if remove_outliers:
        perc_2 = np.zeros(data.shape[1])
        perc_98 = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            perc_2[i] = np.percentile(data[:, i], 2)
            perc_98[i] = np.percentile(data[:, i], 98)

        for i in range(data.shape[1]):
            perc_2_ind = np.where(data[:, i] < perc_2[i])[0]
            perc_98_ind = np.where(data[:, i] > perc_98[i])[0]
            data[perc_2_ind, i] = perc_2[i]
            data[perc_98_ind, i] = perc_98[i]

    scaler = MinMaxScaler()
    scaler.fit(data)

    data_scaled = scaler.transform(data)

    return data_scaled, scaler


def format_labels(labels):
    if labels is None:
        return None, None
    elif isinstance(labels, list) or len(labels.shape) == 1:
        labels = create_class_labels(labels)
        cluster_indices = np.copy(labels)
        return labels, cluster_indices
    elif len(labels.shape) == 2:
        labels = np.copy(labels)
        cluster_indices = []
        for frame_idx, cluster in enumerate(labels):
            frame_clusters = [c_idx for c_idx in range(labels.shape[1]) if cluster[c_idx] == 1]
            cluster_indices.append(frame_clusters)
        cluster_indices = np.array(cluster_indices)
        return labels, cluster_indices
    else:
        raise Exception("Invalid format of lablels. Must be list or 2D np array")


def create_class_labels(cluster_indices):
    """
    Transforms a vector of cluster indices to a matrix where a 1 on the ij element means that the ith frame was in cluster state j+1
    """
    all_cluster_labels = set()
    for t in cluster_indices:
        if isinstance(t, collections.Iterable):
            # We a frame that may belong to mutliple clusters
            for t2 in t:
                all_cluster_labels.add(t2)
        else:
            all_cluster_labels.add(t)
    nclusters = len(all_cluster_labels)
    nframes = len(cluster_indices)
    labels = np.zeros((nframes, nclusters), dtype=int)
    for i, t in enumerate(cluster_indices):
        if isinstance(t, collections.Iterable):
            t = [int(t2) for t2 in t]
        else:
            t = int(t)
        labels[i, t] = 1
    return labels


def check_for_overfit(data_scaled, clustering_prob, classifier):
    """
    Checks if the classifier is overfitted
    Computes an error in the form: sum(1-Ptrue+sum(Pfalse))/N_clusters
    """

    clustering_predicted = classifier.predict(data_scaled)

    # Calculate the error as sum(1-Ptrue+sum(Pfalse))/N_clusters
    error_per_frame = np.zeros((clustering_prob.shape[0]))
    number_of_clusters = clustering_prob.shape[1]

    for i in range(clustering_prob.shape[0]):
        error_per_frame[i] = (1 - np.dot(clustering_prob[i], clustering_predicted[i]) + \
                              np.dot(1 - clustering_prob[i], clustering_predicted[i])) / \
                             number_of_clusters
    error = np.average(error_per_frame) * 100
    return error


def rescale_feature_importance(relevances, std_relevances):
    """
    Min-max rescale feature importances
    :param feature_importance: array of dimension nfeatures * nstates
    :param std_feature_importance: array of dimension nfeatures * nstates
    :return: rescaled versions of the inputs with values between 0 and 1
    """

    logger.info("Rescaling feature importances ...")
    if len(relevances.shape) == 1:
        relevances = relevances[:, np.newaxis]
        std_relevances = std_relevances[:, np.newaxis]
    n_states = relevances.shape[1]
    n_features = relevances.shape[0]

    # indices of residues pairs which were not filtered during features filtering
    indices_not_filtered = np.where(relevances[:, 0] >= 0)[0]

    for i in range(n_states):
        max_val, min_val = relevances[indices_not_filtered, i].max(), relevances[indices_not_filtered, i].min()
        scale = max_val - min_val
        offset = min_val
        if scale < 1e-9:
            scale = max(scale, 1e-9)
        relevances[indices_not_filtered, i] = (relevances[indices_not_filtered, i] - offset) / scale
        std_relevances[indices_not_filtered, i] /= scale

    return relevances, std_relevances


def get_default_feature_to_resids(n_features):
    n_residues = 0.5 * (1 + np.sqrt(8 * n_features + 1))
    n_residues = int(n_residues)
    idx = 0
    feature_to_resids = np.empty((n_features, 2))
    for res1 in range(n_residues):
        for res2 in range(res1 + 1, n_residues):
            feature_to_resids[idx, 0] = res1
            feature_to_resids[idx, 1] = res2
            idx += 1
    return feature_to_resids


def get_feature_to_resids_from_pdb(n_features, pdb_file):
    pdb = PandasPdb()
    pdb.read_pdb(pdb_file)

    resid_numbers = np.unique(np.asarray(list(pdb.df['ATOM']['residue_number'])))
    n_residues = len(resid_numbers)

    n_residues_check = 0.5 * (1 + np.sqrt(8 * n_features + 1))
    if n_residues != n_residues_check:
        sys.exit("The number of residues in pdb file (" + str(
            n_residues) + ") is incompatible with number of features (" + str(n_residues_check) + ")")

    idx = 0
    feature_to_resids = np.empty((n_features, 2))
    for res1 in range(n_residues):
        for res2 in range(res1 + 1, n_residues):
            feature_to_resids[idx, 0] = resid_numbers[res1]
            feature_to_resids[idx, 1] = resid_numbers[res2]
            idx += 1
    return feature_to_resids


def _get_n_components(explained_variance, variance_cutoff):
    if isinstance(variance_cutoff, str) and "_components" in variance_cutoff:
        return int(variance_cutoff.replace("_components", ""))
    elif variance_cutoff is None or variance_cutoff == 'auto':
        n_components = 1
        for i in range(1, explained_variance.shape[0]):
            prev_var, var = explained_variance[i - 1], explained_variance[i]
            if prev_var / var >= 10:
                logger.debug("Computed band gap to find number of components Set it to %s",
                             n_components)
                break
            n_components += 1
        return n_components
    elif isinstance(variance_cutoff, int) or isinstance(variance_cutoff, float):
        n_components = 1
        total_var_explained = explained_variance[0]
        for i in range(1, explained_variance.shape[0]):
            if total_var_explained + explained_variance[i] <= variance_cutoff:
                total_var_explained += explained_variance[i]
                n_components += 1
        return n_components
    else:
        raise Exception("Invalid variance cutoff %s" % variance_cutoff)


def compute_feature_importance_from_components(explained_variance, components, variance_cutoff):
    """
    Computes the feature importance per feature based on the components up to a cutoff in total variance explained
    :param explained_variance:
    :param components:
    :param variance_cutoff: can be a numerical value, 'auto' or '{n}_components' where '{n}' is the number of components to use
    :return:
    """
    n_components = _get_n_components(explained_variance, variance_cutoff)
    logger.debug("Using %s components", n_components)
    importance = None
    for i, c in enumerate(components[0:n_components]):
        c = np.abs(c)
        c *= explained_variance[i]
        importance = c if importance is None else importance + c
    return importance


def compute_mse_accuracy(measured_importance, relevant_residues=None, true_importance=None):
    """
     **MSE accuracy**: Based on the normalized Mean squared error (MSE) $1-\frac{|\bar{x}-\bar{\phi}|^2}{|\bar{x}|^2+|\bar{\phi}|^2}$ where $\bar{\phi}$ is the measured importance and $\bar{x}$ is the true importance
     In other word, we take the error from the measured distribution to the expected and normalize it. We take 1 minus this value so to get accuracy instead of error.
    The code is written to handle weighted true importance (i.e. when not all residues are equally relevant as well)

    :param measured_importance: np.array of dimension 1 with values between 0 and 1. The value at an index corresponds to the relevance of that residue
    :param relevant_residues: the indices of the truly relevant residues, either as 1D or 2D array
    :param true_importance: (optional) Overrides the parameter relevant_residues. Should be the same shape as meaured_importance.
    :return:
    """
    if true_importance is None and relevant_residues is None:
        raise Exception("Either true_importance or relevant_residues must be set")
    if true_importance is None:
        true_importance = np.zeros(measured_importance.shape)
        relevant_residues = np.array(relevant_residues, dtype=int).squeeze()
        true_importance[relevant_residues.flatten()] = 1

    norm = np.linalg.norm(measured_importance - true_importance)
    tot = np.linalg.norm(true_importance) ** 2 + np.linalg.norm(measured_importance) ** 2
    return 1 - norm ** 2 / max(1e-4, tot)


def compute_relevant_fraction_accuracy(importance, relevant_residues):
    """
     relevant_fraction** = $\sum_{x} / \sum_{\phi} $ where $x$ = relevance of truly relevant atoms and $\phi$ = relevance of all atoms
    :param importance:
    :param relevant_residues:
    :return:
    """
    imp_sum = importance.sum()
    return importance[relevant_residues].sum() / max(imp_sum, 1e-4)


def strip_name(name):
    if name is None:
        return None
    parts = []
    for n in name.split("_"):
        n = n.split("-")[0]
        n = n.replace("components", "PC")
        parts.append(n)
    return "\n".join(parts)


def to_scientific_number_format(num):
    if num == 1:
        return "1"
    f = "%.e" % (num)
    f = f.replace("e-0", "e-")
    f = f.replace("e+", "e")
    f = f.replace("e0", "e")
    return f


def _to_numerical(postprocessors, postprocessor_to_number_func):
    res = np.empty(postprocessors.shape, dtype=float)
    for indices, pp in np.ndenumerate(postprocessors):
        num = postprocessor_to_number_func(pp)
        res[indices] = np.nan if num is None else num

    return res


def to_accuracy(postprocessors):
    return _to_numerical(postprocessors, lambda p: p.accuracy)


def to_accuracy_per_cluster(postprocessors):
    return _to_numerical(postprocessors, lambda p: p.accuracy_per_cluster)


def to_separation_score(postprocessors):
    return _to_numerical(postprocessors, lambda p: p.separation_score)


def find_best(postprocessors):
    accuracy = to_accuracy(postprocessors).mean(axis=0)
    ind = accuracy.argmax()
    return postprocessors[:, ind]


def make_list(obj):
    """
    :param it:
    :return: empty list if obj is None, a singleton list of obj is not a list, else obj
    """
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]
