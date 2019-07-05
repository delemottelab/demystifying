from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import mdtraj as md
import numpy as np
from . import filtering

logger = logging.getLogger("trajPreprocessing")


def to_distances(traj,
                 scheme="ca",
                 pairs="all",
                 filter_by_distance_cutoff=False,
                 lower_bound_distance_cutoff=filtering.lower_bound_distance_cutoff_default,
                 upper_bound_distance_cutoff=filtering.upper_bound_distance_cutoff_default,
                 use_inverse_distances=True,
                 ignore_nonprotein=True,
                 periodic=True,
                 ):
    if pairs is None:
        pairs = "all"
    top = traj.topology
    if scheme == 'all-heavy':
        atoms = traj.top.select("{} and element != 'H'".format("protein" if ignore_nonprotein else "all"))
        if pairs is None or 'all' in pairs:
            pairs = []
            for idx1, a1 in enumerate(atoms):
                for idx2 in range(idx1 + 1, len(atoms)):
                    a2 = atoms[idx2]
                    pairs.append([a1, a2])
        samples = md.compute_distances(traj, pairs, periodic=periodic, opt=True)
        pairs = np.array([
            [top.atom(a1), top.atom(a2)] for a1, a2 in pairs
        ])
        feature_to_resids = np.array([
            [a1.residue.resSeq, a2.residue.resSeq] for a1, a2 in pairs
        ], dtype=int)
    else:
        chunk_size = 1000  # To use less memory, don't process entire traj at once
        start = 0
        samples = None
        if "all-residues" == pairs:
            # Unlike MDtrajs 'all' option, this also includes neightbours
            contacts = []
            for r1 in range(traj.top.n_residues):
                for r2 in range(r1 + 1, traj.top.n_residues):
                    contacts.append((r1, r2))
        else:
            contacts = pairs
        while start < len(traj):
            end = start + chunk_size
            s, pairs = md.compute_contacts(traj[start:end], contacts=contacts, scheme=scheme,
                                           ignore_nonprotein=ignore_nonprotein,
                                           periodic=periodic)
            if samples is None:
                samples = s
            else:
                samples = np.append(samples, s, axis=0)
            start = end
        pairs = np.array([[top.residue(r1), top.residue(r2)] for [r1, r2] in pairs])
        feature_to_resids = np.array([
            [r1.resSeq, r2.resSeq] for r1, r2 in pairs
        ], dtype=int)

    if filter_by_distance_cutoff:
        samples, indices_for_filtering = filtering.filter_by_distance_cutoff(samples,
                                                                             lower_bound_distance_cutoff,
                                                                             upper_bound_distance_cutoff,
                                                                             inverse_distances=False)
    if use_inverse_distances:
        samples = 1 / samples
    return samples, feature_to_resids, pairs


def to_compact_distances(traj, **kwargs):
    """
    compact-dist uses as few distances as possible

    :param traj:
    :param kwargs: see to_distances
    :return:
    """
    samples, feature_to_resids, pairs = to_distances(traj, **kwargs)
    if samples.shape[1] <= 6:
        return samples, feature_to_resids, pairs

    indices_to_include = []
    last_r1 = None
    count_for_current_residue = None
    for idx, (r1, r2) in enumerate(pairs):
        # TODO we actually include a few distances too many here I think, but the order of magnitude is optimal
        if last_r1 != r1:
            count_for_current_residue = 0
        if last_r1 is None or count_for_current_residue < 4:
            indices_to_include.append(idx)
        count_for_current_residue += 1
        last_r1 = r1

    feature_to_resids = feature_to_resids[indices_to_include]
    samples = samples[:, indices_to_include]
    return samples, feature_to_resids, pairs


def to_cartesian(traj, query="protein and name 'CA'"):
    """
    :param traj:
    :param query:
    :return: array with every frame along first axis, and xyz coordinates in sequential order for
    every atom returned by the query
    """
    atom_indices, atoms = _get_atoms(traj.top, query)
    xyz = traj.atom_slice(atom_indices=atom_indices).xyz
    natoms = xyz.shape[1]
    samples = np.empty((xyz.shape[0], 3 * natoms))
    feature_to_resids = np.empty((3 * natoms, 1), dtype=int)
    for idx in range(natoms):
        start, end = 3 * idx, 3 * idx + 3
        samples[:, start:end] = xyz[:, idx, :]
        feature_to_resids[start:end] = atoms[idx].residue.resSeq
    return samples, feature_to_resids


def to_local_rmsd(traj, atom_query, nresidues_per_rmsd, reference_structure,
                  alternative_reference_structure=None,
                  only_sequential_residues=True):
    """
    :param traj:
    :param atom_query:
    :param nresidues_per_rmsd:
    :param reference_structure: a trajectory to compute the RMSD to
    :param alternative_reference_structure: (optional) compute the difference in RMSD between this structure and 'referencet_structure'
    :param only_sequential_residues: if True, residues across gaps will not be taken into account
    :return: array with every frame along first axis, and RMSDs coordinates in sequential order for
    every atom returned by the query
    """

    atom_indices, atoms = _get_atoms(traj.top, atom_query)
    subtraj = traj.atom_slice(atom_indices=atom_indices)
    samples = np.empty((len(traj), nresidues_per_rmsd))

    # Create groups of residues to compute rmsd for
    residue_sets = []
    all_residues = [a.residue.resSeq for a in atoms]
    all_residues = [r for r in sorted(set(all_residues))]
    for idx in range(len(all_residues)):
        if idx + nresidues_per_rmsd >= len(all_residues):
            break
        # Get sequential residues
        res_set = [all_residues[ii] for ii in range(idx, idx + nresidues_per_rmsd)]
        is_okay_set = True
        if only_sequential_residues:
            for set_idx in range(1, nresidues_per_rmsd):
                if res_set[set_idx] - res_set[set_idx - 1] != 1:
                    is_okay_set = False
                    break
        if is_okay_set:
            residue_sets.append(res_set)
    # Compute the RMSDs in a second step
    feature_to_resids = []
    samples = []

    def compute_rmsd(ref_traj, residues):
        q = "{} and (resSeq {})".format(atom_query, " or resSeq ".join([str(r) for r in residues]))
        try:
            indices, ref_indices = _select_atoms_incommon(q, subtraj.top, ref_traj.top, exception_on_missing_atoms=True)
        except MissingAtomException as ex:
            logger.exception(ex)
            return None
        if indices is None or len(indices) < nresidues_per_rmsd:
            return None
        return md.rmsd(subtraj.atom_slice(indices), reference=ref_traj.atom_slice(ref_indices))

    for residue_idx, residues in enumerate(residue_sets):
        ref_rmsd = compute_rmsd(reference_structure, residues)
        if ref_rmsd is None:
            continue
        elif alternative_reference_structure is None:
            rmsd = ref_rmsd
        else:
            alt_rmsd = compute_rmsd(alternative_reference_structure, residues)
            if alt_rmsd is None:
                continue
            rmsd = ref_rmsd - alt_rmsd
        samples.append(rmsd)
        feature_to_resids.append(residues)
    samples = np.array(samples).T
    feature_to_resids = np.array(feature_to_resids, dtype=int)
    return samples, feature_to_resids


def _get_atoms(top, query):
    atom_indices = top.select(query)
    atoms = [top.atom(a) for a in atom_indices]
    return atom_indices, atoms


class MissingAtomException(Exception):
    pass


def _select_atoms_incommon(query, top, ref_top, exception_on_missing_atoms=False):
    """
    Matches atoms returned by the query for both topologies by name and returns the atom indices for the respective topology
    """
    _, atoms = _get_atoms(top, query)
    _, ref_atoms = _get_atoms(ref_top, query)
    ref_atoms, missing_atoms = _filter_atoms(ref_atoms, atoms)
    if len(missing_atoms) > 0 and exception_on_missing_atoms:
        raise MissingAtomException("%s atoms in reference not found topology. They will be ignored. %s" %
                                   (len(missing_atoms), missing_atoms))
    atoms, missing_atoms = _filter_atoms(atoms, ref_atoms)
    if len(missing_atoms) > 0 and exception_on_missing_atoms:
        raise MissingAtomException("%s atoms in topology not found reference. They will be ignored. %s" %
                                   (len(missing_atoms), missing_atoms))
    duplicate_atoms = _find_duplicates(atoms)
    if len(duplicate_atoms) > 0 and exception_on_missing_atoms:
        raise MissingAtomException("%s duplicates found in topology %s" % (len(duplicate_atoms), duplicate_atoms))
    duplicate_atoms = _find_duplicates(ref_atoms)
    if len(duplicate_atoms) > 0 and exception_on_missing_atoms:
        raise MissingAtomException("%s duplicates found in reference %s" % (len(duplicate_atoms), duplicate_atoms))
    if len(atoms) != len(ref_atoms) and exception_on_missing_atoms:
        raise MissingAtomException("number of atoms in result differ: %s vs %s" % (len(atoms), len(ref_atoms)))
    return np.array([a.index for a in atoms]), np.array([a.index for a in ref_atoms])


def _filter_atoms(atoms, ref_atoms):
    """
    Returns atoms which name matched the name i ref_atoms as well as the once which did not match.
    Matching is done on name, i.e. str(atom)
    TODO speed up with a search tree or hashmap
    """
    ref_atom_names = [str(a) for a in ref_atoms]
    missing_atoms = []
    matching_atoms = []
    # Atoms in inactive not in simu
    for atom in atoms:
        if str(atom) not in ref_atom_names:
            missing_atoms.append(atom)
        else:
            matching_atoms.append(atom)
    return matching_atoms, missing_atoms


def _find_duplicates(atoms):
    atom_names = [str(a) for a in atoms]
    return [a for a in atoms if atom_names.count(str(a)) > 1]
