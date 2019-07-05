from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import mdtraj as md
from modules import feature_extraction as fe, visualization, traj_preprocessing as tp

logger = logging.getLogger("VSD")


def run_VSD(working_dir="bio_input/VSD/", cluster_for_prediction=None, dt_for_prediction=10, multiclass=False):
    data = np.load(working_dir + 'frame_i_j_contacts_dt1.npy')
    cluster_indices = np.loadtxt(working_dir + 'clusters_indices.dat')

    kwargs = {
        'samples': data,
        'labels': cluster_indices,
        'filter_by_distance_cutoff': True,
        'use_inverse_distances': True,
        'n_splits': 3,
        'n_iterations': 5,
        'scaling': True,
        'shuffle_datasets': True
    }

    if cluster_for_prediction is not None:
        cluster_traj = md.load("{}/{}_dt{}.xtc".format(working_dir, cluster_for_prediction, dt_for_prediction),
                               top=working_dir + "alpha.pdb")
        other_samples, _, _ = tp.to_distances(
            traj=cluster_traj,
            scheme="closest-heavy",
            pairs="all-residues",
            use_inverse_distances=True,
            ignore_nonprotein=True,
            periodic=True)
        logger.debug("Loaded cluster samples for prediction of shape %s for state %s", other_samples.shape,
                     cluster_for_prediction)
        cluster_traj = None  # free memory
    else:
        other_samples = False
    feature_extractors = [
        fe.RandomForestFeatureExtractor(
            classifier_kwargs={
                'n_estimators': 100},
            one_vs_rest=not multiclass,
            **kwargs),
        fe.KLFeatureExtractor(bin_width=0.1, **kwargs),
        fe.MlpFeatureExtractor(
            classifier_kwargs={
                'hidden_layer_sizes': [100, ],
                'max_iter': 100000,
                'alpha': 0.0001},
            activation="relu",
            one_vs_rest=not multiclass,
            per_frame_importance_samples=other_samples,
            per_frame_importance_labels=None,  # If None the method will use predicted labels for LRP
            per_frame_importance_outfile="{}/mlp_perframe_importance_{}/"
                                         "VSD_mlp_perframeimportance_{}_dt{}.txt".format(working_dir,
                                                                                         "multiclass" if multiclass else "binaryclass",
                                                                                         cluster_for_prediction,
                                                                                         dt_for_prediction),
            **kwargs)
    ]

    common_peaks = {
        "R1-R4": [294, 297, 300, 303],
        "K5": [306],
        "R6": [309],
    }
    do_computations = True
    filetype = "svg"
    for extractor in feature_extractors:
        logger.info("Computing relevance for extractors %s", extractor.name)
        extractor.extract_features()
        p = extractor.postprocessing(working_dir=working_dir,
                                     pdb_file=working_dir + "alpha.pdb",
                                     filter_results=False)
        if do_computations:
            p.average()
            p.evaluate_performance()
            p.persist()
        else:
            p.load()

        visualization.visualize([[p]],
                                show_importance=True,
                                show_performance=False,
                                show_projected_data=False,
                                highlighted_residues=common_peaks,
                                outfile=working_dir + "{extractor}/importance_per_residue_{suffix}.{filetype}".format(
                                    suffix="",
                                    extractor=extractor.name,
                                    filetype=filetype))
        if do_computations:
            visualization.visualize([[p]],
                                    show_importance=False,
                                    show_performance=True,
                                    show_projected_data=False,
                                    outfile=working_dir + "{extractor}/performance_{suffix}.{filetype}".format(
                                        extractor=extractor.name,
                                        suffix="",
                                        filetype=filetype))

            visualization.visualize([[p]],
                                    show_importance=False,
                                    show_performance=False,
                                    show_projected_data=True,
                                    outfile=working_dir + "{extractor}/projected_data_{suffix}.{filetype}".format(
                                        extractor=extractor.name,
                                        suffix="",
                                        filetype=filetype))

    logger.info("Done")


if __name__ == "__main__":
    run_VSD(cluster_for_prediction="gamma")
