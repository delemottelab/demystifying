from __future__ import absolute_import, division, print_function

import logging

from demystifying import feature_extraction as fe, visualization
from demystifying.data_generation import DataGenerator

logger = logging.getLogger("demo")
logger.setLevel('INFO')

# Create data for which we know the ground truth
dg = DataGenerator(natoms=20, nclusters=2, natoms_per_cluster=2, nframes_per_cluster=500)
samples, labels = dg.generate_data()
feature_to_resids = dg.feature_to_resids
logger.info("Generated samples and labels of shapes %s and %s", samples.shape, labels.shape)

# Identify important residues using a random forest
extractor = fe.RandomForestFeatureExtractor(samples=samples, labels=labels)
extractor = fe.PCAFeatureExtractor(samples=samples)  # Uncomment for unsupervised learning
extractor.extract_features()

# Postprocess the results to convert importance per feature into importance per residue
postprocessor = extractor.postprocessing()
postprocessor.average()
postprocessor.persist()

# Visualize the importance per residue
# Dashed lines show the residues we know are important (i.e. the atoms moved by the toy model)
visualization.visualize([[postprocessor]], highlighted_residues=dg.moved_atoms)

logger.info(
    "Below we list all features and their importance. Those with highest importance are good candidates for CVs")
for feature_index, importance in postprocessor.get_important_features(sort=True):
    if importance < 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        break
    logger.info("Feature %d has importance %.2f. Corresponds to residues %s", feature_index, importance,
                feature_to_resids[int(feature_index)])

logger.info("Done")
