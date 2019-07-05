import logging
import sys

from sklearn.decomposition import PCA

from .feature_extractor import FeatureExtractor
from .. import utils

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("PCA")


class PCAFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 variance_cutoff='auto',
                 name="PCA",
                 classifier_kwargs={},
                 **kwargs):
        kwargs['n_iterations'] = 1
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=False,
                                  **kwargs)

        logger.debug("Initializing PCA with the following parameters: variance_cutoff %s, classifier_kwargs %s",
                     variance_cutoff, classifier_kwargs)
        self.variance_cutoff = variance_cutoff
        self.classifier_kwargs = classifier_kwargs

    def train(self, train_set, train_labels):
        logger.debug("Training PCA with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        model = PCA(**self.classifier_kwargs)
        model.fit(train_set)
        return model

    def get_feature_importance(self, model, samples, labels):
        logger.debug("Extracting feature importance using PCA ...")
        importance = utils.compute_feature_importance_from_components(model.explained_variance_ratio_,
                                                                      model.components_,
                                                                      self.variance_cutoff)
        return importance


