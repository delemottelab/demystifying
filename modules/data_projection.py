from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

logger = logging.getLogger("projection")


class DataProjector():
    def __init__(self, samples, labels):
        """
        Class that performs dimensionality reduction using the relevances from each estimator.
        :param postprocessor:
        :param samples:
        """
        self.samples = samples
        self.cluster_indices = labels.argmax(axis=1)

        self.n_clusters = labels.shape[1]

        self.projection = None

        self.separation_score = None
        self.projection_class_entropy = None
        self.cluster_projection_class_entropy = None
        return

    def project(self, feature_importances):
        """
        Project distances. Performs:
          1. Raw projection (projection onto cluster feature importances if feature_importances_per_cluster is given)
          2. Basis vector projection (basis vectors identified using graph coloring).
        """

        self.projection = self._project_on_relevance_basis_vectors(self.samples, feature_importances)

        return self

    def score_projection(self, projection=None, use_GMM=True):
        """
        Score the resulting projection by approximating each cluster as a Gaussian mixture (or Gaussian)
        distribution and classify points using the posterior distribution over classes.
        The number of correctly classified points divided by the number of points is the projection score.
        :return: itself
        """

        if projection is None:
            if self.projection is not None:
                proj = np.copy(self.projection)
                logger.info("Scoring projections.")
            else:
                logger.warn("No projection data.")
                return self
        else:
            proj = np.copy(projection)

        priors = self._set_class_prior()
        n_points = proj.shape[0]
        new_classes = np.zeros(n_points)
        class_entropies = np.zeros(n_points)
        try:
            if use_GMM:
                GMMs = self._fit_GM(proj)
            else:
                means, covs = self._fit_Gaussians(proj)

            for i_point in range(n_points):
                if use_GMM:
                    posteriors = self._compute_GM_posterior(proj[i_point, :], priors, GMMs)
                else:
                    posteriors = self._compute_gaussian_posterior(proj[i_point, :], priors, means, covs)
                class_entropies[i_point] = entropy(posteriors)
                new_classes[i_point] = np.argmax(posteriors)

            # Compute separation score
            correct_separation = new_classes == self.cluster_indices
            if projection is None:
                self.separation_score = correct_separation.sum() / n_points
                self.projection_class_entropy = class_entropies.mean()
            else:
                return correct_separation.sum() / n_points, class_entropies.mean()
        except Exception as ex:
            logger.exception(ex)
            logger.warning('Could not calculate projection prediction score and entropy.')
            class_entropies = np.nan * np.ones(class_entropies.shape)
            if projection is None:
                self.separation_score = np.nan
                self.projection_class_entropy = np.nan
            else:
                return np.nan, np.nan

        # Compute per-cluster projection entropy
        self.cluster_projection_class_entropy = np.zeros(self.n_clusters)
        for i_cluster in range(self.n_clusters):
            inds = self.cluster_indices == i_cluster
            self.cluster_projection_class_entropy[i_cluster] = class_entropies[inds].mean()

        return self

    def persist(self):
        """
        Write projected data to files.
        """
        if self.projection is not None:
            np.save(self.directory + "relevance_raw_projection", self.projection)
        if self.basis_vector_projection is not None:
            np.save(self.directory + "relevance_basis_vector_projection", self.basis_vector_projection)
        return

    def _compute_gaussian_posterior(self, x, priors, means, covs):
        """
        Compute Gaussian class posteriors
        :param point:
        :param priors:
        :param means:
        :param covs:
        :return:
        """
        posteriors = np.zeros(self.n_clusters)
        for i_cluster in range(self.n_clusters):
            density = multivariate_normal.pdf(x, mean=means[i_cluster], cov=covs[i_cluster])
            posteriors[i_cluster] = density * priors[i_cluster]

        posteriors /= posteriors.sum()
        return posteriors

    def _compute_GM_posterior(self, x, priors, GMMs):
        """
        Compute class posteriors, where each class has a GM distribution.
        :param point:
        :param priors: Prior distribution over classes
        :param GMMs: List with each cluster's GMM-density.
        :return:
        """
        posteriors = np.zeros(self.n_clusters)
        n_dims = GMMs[0].covariances_[0].shape[0]
        for i_cluster in range(self.n_clusters):
            gmm = GMMs[i_cluster]
            density = 0.0
            for i_component in range(gmm.weights_.shape[0]):
                density += gmm.weights_[i_component] * multivariate_normal.pdf(x, mean=gmm.means_[i_component],
                                                                               cov=gmm.covariances_[
                                                                                       i_component] + 1e-7 * np.eye(
                                                                                   n_dims))
            posteriors[i_cluster] = density * priors[i_cluster]

        posteriors /= posteriors.sum()
        return posteriors

    def _estimate_GMM(self, x, n_component_lim=[1, 3]):
        """
        Find the GMM that best fit the data in x using Bayesian information criterion.
        """
        min_comp = n_component_lim[0]
        max_comp = n_component_lim[1]

        lowest_BIC = np.inf

        counter = 0
        for i_comp in range(min_comp, max_comp + 1):
            GMM = GaussianMixture(i_comp)
            GMM.fit(x)
            BIC = GMM.bic(x)
            if BIC < lowest_BIC:
                lowest_BIC = BIC
                best_GMM = GaussianMixture(i_comp)
                best_GMM.weights_ = GMM.weights_
                best_GMM.means_ = GMM.means_
                best_GMM.covariances_ = GMM.covariances_

            counter += 1

        return best_GMM

    def _fit_GM(self, proj, n_component_lim=[1, 3]):
        """
        Fit a Gaussian mixture model to the data in cluster
        :param proj:
        :return:
        """
        models = []

        for i_cluster in range(self.n_clusters):
            cluster = proj[self.cluster_indices == i_cluster]

            GMM = self._estimate_GMM(cluster, n_component_lim)
            models.append(GMM)

        return models

    def _fit_Gaussians(self, proj):
        """
        Compute mean and covariance of each cluster
        :return:
        """
        means = []
        covs = []
        n_dims = proj.shape[1]

        for i_cluster in range(self.n_clusters):
            cluster = proj[self.cluster_indices == i_cluster, :]
            means.append(cluster.mean(axis=0))
            covs.append(np.cov(cluster.T, rowvar=True) + 1e-7 * np.eye(n_dims))

        return means, covs

    def _set_class_prior(self):
        prior = np.zeros(self.n_clusters)

        for i_cluster in range(self.n_clusters):
            prior[i_cluster] = np.sum(self.cluster_indices == i_cluster)
        return prior

    def _project_on_relevance_basis_vectors(self, distances, relevance_basis_vectors):
        """
        Project all input distances onto the detected basis vectors.
        """

        projected_data = np.dot(distances, relevance_basis_vectors)

        return projected_data
