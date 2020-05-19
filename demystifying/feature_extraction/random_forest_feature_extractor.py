import logging
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .feature_extractor import FeatureExtractor

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("RF")


class RandomForestFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 name="RF",
                 classifier_kwargs={
                     'n_estimators': 30,
                 },
                 randomize=True,
                 one_vs_rest=True,
                 **kwargs):

        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=True,
                                  **kwargs)
        self.randomize = randomize
        self.classifier_kwargs = classifier_kwargs.copy()
        if not self.randomize:
            self.classifier_kwargs['random_state'] = 89274
        if self.use_regression:
            self.one_vs_rest = False
        else:
            self.one_vs_rest = one_vs_rest

        logger.debug("Initializing RF with the following parameters: "
                     " randomize %s, one_vs_rest %s, classifier_kwargs %s",
                     self.randomize, self.one_vs_rest, self.classifier_kwargs)

    def _train_one_vs_rest(self, data, labels):
        n_clusters = labels.shape[1]
        n_points = data.shape[0]

        classifiers = []

        for i_cluster in range(n_clusters):
            classifiers.append(self._create_classifier())
            tmp_labels = np.zeros(n_points)
            tmp_labels[labels[:, i_cluster] == 1] = 1

            classifiers[i_cluster].fit(data, tmp_labels)

        return classifiers

    def train(self, train_set, train_labels):
        # Construct and train classifier
        logger.debug("Training RF with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        if self.one_vs_rest:
            return self._train_one_vs_rest(train_set, train_labels)
        else:
            classifier = self._create_classifier()
            if self.use_regression:
                train_labels = train_labels.squeeze()
            classifier.fit(train_set, train_labels)
        return classifier

    def get_feature_importance(self, classifier, data, labels):
        logger.debug("Extracting feature importance using RF ...")
        n_features = data.shape[1]
        feature_importances = np.zeros((n_features, self.n_clusters))
        for i_cluster in range(self.n_clusters):
            if self.one_vs_rest:
                feature_importances[:, i_cluster] = classifier[i_cluster].feature_importances_
            else:
                feature_importances[:, i_cluster] = classifier.feature_importances_
        return feature_importances

    def _create_classifier(self):
        return RandomForestRegressor(**self.classifier_kwargs) if self.use_regression \
            else RandomForestClassifier(**self.classifier_kwargs)
