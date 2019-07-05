from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from .feature_extractor import FeatureExtractor

logger = logging.getLogger("KL divergence")


class RandomFeatureExtractor(FeatureExtractor):
    """Class which randomly assigns importance to features"""

    def __init__(self,
                 name="RAND",
                 **kwargs):
        FeatureExtractor.__init__(self,
                                  name=name,
                                  supervised=True,
                                  **kwargs)

    def train(self, data, labels):
        pass

    def get_feature_importance(self, model, data, labels):
        """
        returns random values per feature between 0 and 1
        """
        return np.random.random((data.shape[1], labels.shape[1]))
