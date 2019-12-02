from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import mdtraj as md
import numpy as np
import glob
from modules import utils, filtering, feature_extraction as fe, visualization, traj_preprocessing as tp

logger = logging.getLogger("beta2")
utils.remove_outliers = False


def _get_important_residues(supervised, feature_type):
    npxxy = [322, 323, 324, 325]
    ligand_interactions = [109, 113, 114, 117, 193, 195, 203, 204, 207, 286, 289, 290, 293, 308, 309, 312]
    most_conserved_TM_residues = [51, 79, 131, 158, 211, 288, 323]
    if supervised:
        if "rmsd" in feature_type:
            return {
                # 'Ligand interactions': ligand_interactions,
                'Connector': [121, 282],
                'M82': [82],  # , 286, 316],
                'DRY': [130, 131, 132],
                # 'NPxxY': npxxy,
                # 'Most conserved TM residues': most_conserved_TM_residues
            }
        else:
            return {
                'Ligand interactions': ligand_interactions,
                'D79': [79],
                'E268': [268],
                'L144': [144],
            }
    else:
        return {
            'NPxxY': npxxy,
            'End of TM6': [268, 272, 275, 279],
            'L144': [144],
        }


def _load_trajectory_for_predictions(ligand_type):
    if ligand_type not in ['apo', 'holo']:
        raise NotImplementedError
    infile = "/home/oliverfl/MEGA/PHD/projects/relevance_propagation/results/apo-holo/trajectories/asp79-{}-swarms-nowater-nolipid".format(
        ligand_type)
    traj = md.load(infile + ".xtc", top=infile + ".pdb")
    samples, feature_to_resids, pairs = tp.to_distances(traj)
    return samples, None


def run_beta2(
        working_dir="bio_input/new-ligands/",
        n_iterations=1,
        n_splits=1,
        shuffle_datasets=True,
        overwrite=False,
        dt=1,
        feature_type="ca_inv",  # "closest-heavy_inv", "CA_inv", "cartesian_ca", "cartesian_noh" or "compact_ca_inv"
        filetype="svg",
        classtype="multiclass",
        supervised=True,
        load_trajectory_for_predictions=False,
        filter_by_distance_cutoff=False,
        ligand_type='holo'):
    results_dir = "{}/results/".format(working_dir)

    samples_dir = "{}/samples/{}/".format(working_dir, feature_type)
    data = np.load("{}/samples_dt1.npz".format(samples_dir, dt))['array']
    feature_to_resids = np.load("{}/feature_to_resids.npy".format(samples_dir, feature_type))
    labels = np.loadtxt("{wd}/samples/cluster_indices.txt".format(wd=working_dir))
    label_names = np.array([
        'carazolol',  # 0
        'apo',  # 1
        'adrenaline',  # 2
        'alprenolol',  # 3
        'p0g',  # 4
        'salmeterol',  # 5
        'timolol'  # 6
    ])
    suffix = str(-1) + "clusters_" + str(n_iterations) + "iterations_" \
             + ("distance-cutoff_" if filter_by_distance_cutoff else "") + feature_type
    if len(data) != len(labels) or data.shape[1] != len(feature_to_resids):
        raise Exception("Inconsistent input data. The number of features or the number of frames to no match")
    logger.info("Loaded data of shape %s for feature type %s", data.shape, feature_type)
    mixed_classes = False
    # ## Define the different methods to use
    # Every method is encapsulated in a so called FeatureExtractor class which all follow the same interface
    cutoff_offset = 0.2 if "closest-heavy" in feature_type else 0
    kwargs = {
        'samples': data,
        'labels': labels,
        'label_names': label_names,
        'filter_by_distance_cutoff': filter_by_distance_cutoff,
        'lower_bound_distance_cutoff': filtering.lower_bound_distance_cutoff_default - cutoff_offset,
        'upper_bound_distance_cutoff': filtering.upper_bound_distance_cutoff_default - cutoff_offset,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        'shuffle_datasets': shuffle_datasets
        # 'upper_bound_distance_cutoff': 1.,
        # 'lower_bound_distance_cutoff': 1.
    }
    unsupervised_feature_extractors = [
        fe.PCAFeatureExtractor(classifier_kwargs={'n_components': None},
                               variance_cutoff='auto',
                               # variance_cutoff='1_components',
                               name='PCA',
                               **kwargs),
        fe.RbmFeatureExtractor(classifier_kwargs={'n_components': 1},
                               relevance_method='from_lrp',
                               name='RBM',
                               **kwargs),
        # fe.MlpAeFeatureExtractor(
        #     classifier_kwargs={
        #         'hidden_layer_sizes': (100, 30, 2, 30, 100,),  # int(data.shape[1]/2),),
        #         # max_iter=10000,
        #         'alpha': 0.01,
        #         'activation': "logistic"
        #     },
        #     use_reconstruction_for_lrp=True,
        #     **kwargs),
    ]
    if load_trajectory_for_predictions:
        other_samples, other_labels = _load_trajectory_for_predictions(ligand_type)
    else:
        other_samples, other_labels = None, None
    supervised_feature_extractors = [
        # fe.ElmFeatureExtractor(
        #     activation="relu",
        #     n_nodes=data.shape[1] * 2,
        #     alpha=0.1,
        #     **kwargs),
        fe.KLFeatureExtractor(**kwargs),
        fe.RandomForestFeatureExtractor(
            one_vs_rest=True,
            classifier_kwargs={'n_estimators': 500},
            **kwargs),
        fe.MlpFeatureExtractor(
            name="MLP" if other_samples is None else "MLP_predictor_{}".format(ligand_type),
            classifier_kwargs={
                # 'hidden_layer_sizes': [int(min(100, data.shape[1]) / (i + 1)) + 1 for i in range(3)],
                'hidden_layer_sizes': (30,),
                # 'max_iter': 10000,
                'alpha': 0.1,
                'activation': "relu"
            },
            # per_frame_importance_samples=other_samples,
            # per_frame_importance_labels=other_labels,
            # per_frame_importance_outfile="/home/oliverfl/projects/gpcr/mega/Result_Data/beta2-dror/apo-holo/trajectories"
            #                              "/mlp_perframe_importance_{}/"
            #                              "{}_mlp_perframeimportance_{}clusters_{}cutoff.txt"
            #     .format(ligand_type, feature_type, nclusters, "" if filter_by_distance_cutoff else "no"),
            **kwargs),
    ]

    if supervised is None:
        feature_extractors = unsupervised_feature_extractors + supervised_feature_extractors
    else:
        feature_extractors = supervised_feature_extractors if supervised else unsupervised_feature_extractors
    logger.info("Done. using %s feature extractors", len(feature_extractors))
    highlighted_residues = _get_important_residues(supervised, feature_type)
    # # Run the relevance analysis
    postprocessors = []
    for extractor in feature_extractors:
        do_computations = True
        if os.path.exists(results_dir):
            existing_files = glob.glob("{}/{}/importance_per_residue.npy".format(results_dir, extractor.name))
            if len(existing_files) > 0 and not overwrite:
                logger.debug("File %s already exists. skipping computations", existing_files[0])
                do_computations = False
        if do_computations:
            logger.info("Computing relevance for extractors %s", extractor.name)
            extractor.extract_features()
        p = extractor.postprocessing(working_dir=results_dir,
                                     pdb_file=working_dir + "/trajectories/all.pdb",
                                     # pdb_file=working_dir + "/trajectories/protein_noh.pdb",
                                     feature_to_resids=feature_to_resids,
                                     filter_results=False)
        if do_computations:
            p.average()
            p.evaluate_performance()
            p.persist()
        else:
            p.load()

        postprocessors.append([p])
        # # Visualize results
        visualization.visualize([[p]],
                                show_importance=True,
                                show_performance=False,
                                show_projected_data=False,
                                mixed_classes=mixed_classes,
                                highlighted_residues=highlighted_residues,
                                outfile=results_dir + "/{extractor}/importance_per_residue_{suffix}_{extractor}.{filetype}".format(
                                    suffix=suffix,
                                    extractor=extractor.name,
                                    filetype=filetype))

        if do_computations:
            visualization.visualize([[p]],
                                    show_importance=False,
                                    show_performance=True,
                                    show_projected_data=False,
                                    mixed_classes=mixed_classes,
                                    outfile=results_dir + "/{extractor}/performance_{suffix}_{extractor}.{filetype}".format(
                                        suffix=suffix,
                                        extractor=extractor.name,
                                        filetype=filetype))
            visualization.visualize([[p]],
                                    show_importance=False,
                                    show_performance=False,
                                    show_projected_data=True,
                                    mixed_classes=mixed_classes,
                                    outfile=results_dir + "/{extractor}/projected_data_{suffix}_{extractor}.{filetype}".format(
                                        suffix=suffix,
                                        extractor=extractor.name,
                                        filetype=filetype))
    logger.info("Done. The settings were n_iterations = {n_iterations}, n_splits = {n_splits}."
                "\nFiltering (filter_by_distance_cutoff={filter_by_distance_cutoff})".format(**kwargs))


if __name__ == "__main__":
    run_beta2(feature_type="rmsd_local",
              n_iterations=10,
              n_splits=1,
              supervised=True,
              shuffle_datasets=True,
              overwrite=False,
              load_trajectory_for_predictions=False,
              ligand_type='apo',
              filter_by_distance_cutoff=False)
