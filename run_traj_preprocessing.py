from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import mdtraj as md
import os
import numpy as np
import argparse
from modules import traj_preprocessing as tp

logger = logging.getLogger("runPreprocessing")


def create_argparser():
    parser = argparse.ArgumentParser(
        epilog='Demystifying stuff since 2018. By delemottelab')
    parser.add_argument('--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Relative path to output directory from the working dir',
                        required=False, default=None)
    parser.add_argument('--traj', type=str, help='Relative path to trajectory file from the working dir', required=True)
    parser.add_argument('--topology', type=str, help='Relative path to topology file from the working dir',
                        required=False,
                        default=None)
    parser.add_argument('--feature_type', type=str, help='Choice of feature type', required=True)
    parser.add_argument('--dt', type=int, help='Timestep between frames', default=1)
    parser.add_argument('--nresidues_per_rmsd', type=int, help='Number of residues for local RMSD', default=4)
    parser.add_argument('--query', type=str, help='MDTraj query for atom selection (only used in some cases)',
                        default="protein and element != 'H'")
    parser.add_argument('--reference_structure', type=str, help='Reference structure for RMSD computations',
                        default=None)
    parser.add_argument('--alternative_reference_structure', type=str,
                        help='Alternative reference structure for RMSD computations',
                        default=None)
    return parser


def run_preprocessing(args):
    dt = args.dt
    traj = md.load(args.working_dir + "/" + args.traj,
                   top=None if args.topology is None else args.working_dir + args.topology,
                   stride=dt)
    if args.reference_structure is not None:
        reference_structure = md.load(args.working_dir + "/" + args.reference_structure)
    else:
        reference_structure = None
    if args.alternative_reference_structure is not None:
        alternative_reference_structure = md.load(args.working_dir + "/" + args.alternative_reference_structure)
    else:
        alternative_reference_structure = None
    logger.info("Loaded trajectory %s", traj)
    if args.feature_type == 'ca_inv':
        samples, feature_to_resids, pairs = tp.to_distances(traj, scheme='ca', use_inverse_distances=True)
    elif args.feature_type == 'closest-heavy_inv':
        samples, feature_to_resids, pairs = tp.to_distances(traj, scheme='closest-heavy', use_inverse_distances=True)
    elif args.feature_type == 'compact_ca_inv':
        samples, feature_to_resids, pairs = tp.to_compact_distances(traj, scheme='ca', use_inverse_distances=True)
    elif args.feature_type == 'cartesian_ca':
        samples, feature_to_resids, pairs = tp.to_cartesian(traj)
    elif args.feature_type == 'cartesian_noh':
        samples, feature_to_resids, pairs = tp.to_cartesian(traj, query="protein and element != 'H'")
    elif args.feature_type == 'cartesian_query':
        samples, feature_to_resids = tp.to_cartesian(traj, query=args.query)
    elif args.feature_type == 'rmsd_local':
        samples, feature_to_resids = tp.to_local_rmsd(traj, atom_query=args.query,
                                                      reference_structure=reference_structure,
                                                      alternative_reference_structure=alternative_reference_structure,
                                                      nresidues_per_rmsd=args.nresidues_per_rmsd)
    else:
        raise NotImplementedError("feature_type %s not supported" % args.feature_type)

    out_dir = (args.working_dir if args.output_dir is None else args.output_dir) + "/"
    out_dir += args.feature_type
    out_dir += "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(out_dir + "samples_dt%s" % dt, array=samples)
    np.save(out_dir + "feature_to_resids", feature_to_resids)
    logger.info("Finished. Saved results in %s", out_dir)


if __name__ == "__main__":
    logger.info("----------------Starting trajectory preprocessing------------")
    parser = create_argparser()
    args = parser.parse_args()
    logger.info("Starting with arguments: %s", args)
    run_preprocessing(args)
