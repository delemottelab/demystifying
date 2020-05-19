import argparse
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

from demystifying import feature_extraction as fe, visualization
from demystifying import relevance_propagation as relprop

logger = logging.getLogger("CaM")


def run_CaM(parser):
    # Known important residues
    common_peaks = [109, 144, 124, 145, 128, 105, 112, 136, 108, 141, 92]

    shuffle_data = True

    args = parser.parse_args()
    working_dir = args.out_directory
    n_runs = args.number_of_runs
    samples = np.load(args.feature_list)

    cluster_indices = np.loadtxt(args.cluster_indices)

    # Shift cluster indices to start at 0
    cluster_indices -= cluster_indices.min()

    if shuffle_data:
        # Permute blocks of 100 frames
        n_samples = samples.shape[0]
        n_samples = int(n_samples / 100) * 100
        inds = np.arange(n_samples)
        inds = inds.reshape((int(n_samples / 100), 100))
        perm_inds = np.random.permutation(inds)
        perm_inds = np.ravel(perm_inds)

        samples = samples[perm_inds]
        cluster_indices = cluster_indices[perm_inds]

    pdb_file = args.pdb_file

    labels = cluster_indices

    lower_distance_cutoff = 1.0
    upper_distance_cutoff = 1.0
    n_components = 20

    # Check if samples format is correct
    if len(samples.shape) != 2:
        sys.exit("Matrix with features should have 2 dimensions")

    kwargs = {'samples': samples, 'labels': labels,
              'filter_by_distance_cutoff': True, 'lower_bound_distance_cutoff': lower_distance_cutoff,
              'upper_bound_distance_cutoff': upper_distance_cutoff, 'use_inverse_distances': True,
              'n_splits': args.number_of_k_splits, 'n_iterations': args.number_of_iterations, 'scaling': True}

    feature_extractors = [
        fe.PCAFeatureExtractor(variance_cutoff=0.75, **kwargs),
        fe.RbmFeatureExtractor(relevance_method="from_components", **kwargs),
        fe.MlpAeFeatureExtractor(activation=relprop.relu, classifier_kwargs={
            'solver': 'adam',
            'hidden_layer_sizes': (100,)
        }, **kwargs),
        fe.RandomForestFeatureExtractor(one_vs_rest=True, classifier_kwargs={'n_estimators': 500}, **kwargs),
        fe.KLFeatureExtractor(**kwargs),
        fe.MlpFeatureExtractor(classifier_kwargs={'hidden_layer_sizes': (120,),
                                                  'solver': 'adam',
                                                  'max_iter': 1000000
                                                  }, activation=relprop.relu, **kwargs),
    ]

    postprocessors = []
    for extractor in feature_extractors:

        tmp_pp = []
        for i_run in range(n_runs):
            extractor.extract_features()
            # Post-process data (rescale and filter feature importances)
            p = extractor.postprocessing(working_dir=working_dir, rescale_results=True,
                                         filter_results=False, feature_to_resids=None, pdb_file=pdb_file)
            p.average().evaluate_performance()
            p.persist()

            # Add common peaks
            tmp_pp.append(p)

        postprocessors.append(tmp_pp)

    visualization.visualize(postprocessors,
                            show_importance=True,
                            show_projected_data=False,
                            show_performance=False,
                            highlighted_residues=common_peaks,
                            outfile="{}/importance-per-residue.png".format(working_dir)
                            )
    logger.info("Done")



def create_arg_parser():
    parser = argparse.ArgumentParser(epilog='Feature importance extraction.')
    parser.add_argument('-od', '--out_directory', help='Folder where files are written.', default='bio_input/CaM/')
    parser.add_argument('-y', '--cluster_indices', help='Cluster indices.',
                        default='bio_input/CaM/cluster_labels/cluster_indices_cterm_4cal_spectral.txt')
    parser.add_argument('-f', '--feature_list', help='Matrix with features [nSamples x nFeatures]',
                        default='bio_input/CaM/samples/inverse_CA_cterm_cterm_holo_CaM.npy')
    parser.add_argument('-n_iter', '--number_of_iterations', help='Number of iterations to average each k-split over.',
                        type=int, default=10)
    parser.add_argument('-n_runs', '--number_of_runs', help='Number of iterations to average performance on.', type=int,
                        default=3)
    parser.add_argument('-n_splits', '--number_of_k_splits', help='Number of k splits in K-fold cross-validation.',
                        type=int, default=10)
    parser.add_argument('-pdb', '--pdb_file', help='PDB file to which the results will be mapped.',
                        default='bio_input/CaM/C-CaM.pdb')
    return parser


if __name__ == "__main__":
    run_CaM(create_arg_parser())
