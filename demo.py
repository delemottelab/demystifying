from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

from demystifying import feature_extraction as fe, visualization
from demystifying.data_generation import DataGenerator

logger = logging.getLogger("demo")

if __name__ == "__main__":
    # Create data for which we know the ground truth
    dg = DataGenerator(natoms=20, nclusters=2, natoms_per_cluster=2, nframes_per_cluster=400)
    data, labels = dg.generate_data()
    logger.info("Generated data of shape %s and %s clusters", data.shape, labels.shape[1])

    # Identify important residues using a random forest
    extractor = fe.RandomForestFeatureExtractor(samples=data, labels=labels)
    extractor.extract_features()

    # Postprocess the results to convert importance per feature into importance per residue
    p = extractor.postprocessing()
    p.average()
    p.evaluate_performance()
    p.persist()

    # Visualize the importance per residue and put dashed lines on the residues we know are important
    visualization.visualize([[p]], highlighted_residues=dg.moved_atoms)

    logger.info("Done")
