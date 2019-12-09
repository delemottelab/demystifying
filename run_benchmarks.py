from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib as mpl

mpl.use('Agg')  # TO DISABLE GUI. USEFUL WHEN RUNNING ON CLUSTER WITHOUT X SERVER
import argparse
import numpy as np
from benchmarking import computing
from modules import visualization, utils

logger = logging.getLogger("benchmarking")


def _fix_extractor_type(extractor_types):
    extractor_types = utils.make_list(extractor_types)
    if len(extractor_types) == 1:
        et = extractor_types[0]
        if et == "supervised":
            return ["KL", "RF", "MLP", "RAND"]
        elif et == "unsupervised":
            return ["PCA", "RBM", "AE", "RAND"]
        elif et == "all":
            return ["PCA", "RBM", "AE", "KL", "RF", "MLP", "RAND"]
    return extractor_types


def create_argparser():
    _bool_lambda = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser(
        epilog='Benchmarking for demystifying')
    parser.add_argument('--extractor_type', nargs='+', help='list of extractor types (MLP, KL, PCA, ..)', type=str,
                        required=True)
    parser.add_argument('--output_dir', type=str, help='Root directory for output files',
                        default="output/benchmarking/")
    parser.add_argument('--test_model', nargs='+', type=str, help='Toy model displacement: linear or non-linear',
                        default="linear")
    parser.add_argument('--feature_type', nargs='+',
                        type=str, help='Toy model feature type: cartesian_rot, inv-dist, etc.',
                        default="cartesian_rot")
    parser.add_argument('--noise_level', nargs='+', type=float,
                        help='Strength of noise added to atomic coordinates at each frame',
                        default=1e-2)
    parser.add_argument('--displacement', type=float, help='Strength of displacement for important atoms', default=1e-1)
    parser.add_argument('--overwrite', type=_bool_lambda,
                        help='Overwrite existing results with new (if set to False no new computations will be performed)',
                        default=False)
    parser.add_argument('--visualize', type=_bool_lambda, help='Generate output figures', default=True)
    parser.add_argument('--iterations_per_model', type=int, help='', default=10)
    parser.add_argument('--accuracy_method', nargs='+', type=str, help='', default='mse')
    parser.add_argument('--n_atoms', nargs='+', type=int, help='Number of atoms in toy model', default=100)

    return parser


def do_run(args, extractor_types, noise_level, test_model, feature_type, accuracy_method, natoms):
    visualize = args.visualize
    output_dir = args.output_dir
    fig_filename = "{feature_type}_{test_model}_{noise_level}noise_{natoms}atoms_{accuracy_method}.svg".format(
        feature_type=feature_type,
        test_model=test_model,
        noise_level=noise_level,
        accuracy_method=accuracy_method,
        natoms=natoms
    )
    best_processors = []
    for et in extractor_types:
        try:
            postprocessors = computing.compute(extractor_type=et,
                                               output_dir=output_dir,
                                               feature_type=feature_type,
                                               overwrite=args.overwrite,
                                               accuracy_method=accuracy_method,
                                               iterations_per_model=args.iterations_per_model,
                                               noise_level=noise_level,
                                               # Disable visualization here since we all these
                                               # individual performance plot take up disk space and don't give us much
                                               visualize=False,
                                               natoms=natoms,
                                               test_model=test_model)
            if visualize:
                visualization.show_single_extractor_performance(postprocessors=postprocessors,
                                                                extractor_type=et,
                                                                filename=fig_filename,
                                                                output_dir=output_dir,
                                                                accuracy_method=accuracy_method)
            best_processors.append(utils.find_best(postprocessors))
        except Exception as ex:
            logger.exception(ex)
            logger.warn("Failed for extractor %s ", et)
            raise ex
    best_processors = np.array(best_processors)
    if visualize:
        fig_filename = fig_filename.replace(".svg", "_{}.svg".format("-".join(extractor_types)))
        visualization.show_all_extractors_performance(best_processors,
                                                      extractor_types,
                                                      feature_type=feature_type,
                                                      filename=fig_filename,
                                                      output_dir=output_dir,
                                                      accuracy_method=accuracy_method)
        return best_processors


def run_all(args):
    extractor_types = _fix_extractor_type(args.extractor_type)
    n_atoms = utils.make_list(args.n_atoms)
    for feature_type in utils.make_list(args.feature_type):
        for noise_level in utils.make_list(args.noise_level):
            for test_model in utils.make_list(args.test_model):
                for accuracy_method in utils.make_list(args.accuracy_method):
                    best_processors = []
                    for natoms in n_atoms:
                        bp = do_run(args, extractor_types, noise_level,
                                    test_model, feature_type, accuracy_method, natoms)
                        best_processors.append(bp)
                    if visualization and len(n_atoms) > 1:
                        visualization.show_system_size_dependence(n_atoms=n_atoms,
                                                                  postprocessors=np.array(best_processors),
                                                                  extractor_types=extractor_types,
                                                                  noise_level=noise_level,
                                                                  test_model=test_model,
                                                                  feature_type=feature_type,
                                                                  output_dir=args.output_dir,
                                                                  accuracy_method=accuracy_method)


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    logger.info("Starting script run_benchmarks with arguments %s", args)
    run_all(args)
    logger.info("Done!")
