import logging
import sys

from sklearn.decomposition import PCA

from .feature_extractor import FeatureExtractor
from .. import utils

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("PCA")


class PCAFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 supervised=False,
                 variance_cutoff='auto',
                 name="PCA",
                 classifier_kwargs={},
                 **kwargs):
        kwargs['n_iterations'] = 1
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=supervised,
                                  **kwargs)

        logger.debug("Initializing PCA with the following parameters: variance_cutoff %s, classifier_kwargs %s",
                     variance_cutoff, classifier_kwargs)
        self.variance_cutoff = variance_cutoff
        self.classifier_kwargs = classifier_kwargs

    def train(self, train_set, train_labels):
        if self.supervised and train_labels is not None:
            return self._train_unsupervised_methods_per_class(train_set, train_labels)
        else:
            model = PCA(**self.classifier_kwargs)
            model.fit(train_set)
            return model

    def get_feature_importance(self, model, samples, labels):
        if self.supervised and labels is not None:
            return self._get_feature_importance_for_unsupervised_per_class(model, samples, labels)
        importance = utils.compute_feature_importance_from_components(model.explained_variance_ratio_,
                                                                      model.components_,
                                                                      self.variance_cutoff)
        return importance


