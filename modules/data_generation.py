from __future__ import absolute_import, division, print_function

import logging
import os
import sys

import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("DataGenerator")


class DataGenerator(object):

    def __init__(self, natoms, nclusters, natoms_per_cluster, nframes_per_cluster,
                 test_model='linear',
                 noise_level=1e-2,
                 noise_natoms=None,
                 displacement=0.1,
                 feature_type='inv-dist',
                 moved_atoms=None):
        """
        Class which generates artificial atoms, puts them into artifical clusters and adds noise to them
        :param natoms: number of atoms
        :param nclusters: number of clusters
        :param nframes_per_cluster:
        :param test_model: 'linear','non-linear','non-linear-random-displacement','non-linear-p-displacement'
        :param noise_level: strength of noise to be added
        :param noise_natoms: number of atoms for constant noise
        :param displacement: length of displacement vector for cluster generation
        :param feature_type: 'inv-dist' to use inversed inter-atomic distances (natoms*(natoms-1)/2 features in total), compact-dist to use as few distances as possible, or anything that starts with 'cartesian' to use atom xyz coordiantes (3*natoms features). Use 'cartesian_rot', 'cartesian_trans' or 'cartesian_rot_trans' to add a random rotation and/or translation to xyz coordaintes
        :param moved_atoms: define which atoms to displace instead of choosing them randomly
        """

        if natoms < nclusters:
            raise Exception("Cannot have more clusters than atoms")
        if natoms_per_cluster is None or len(natoms_per_cluster) != nclusters:
            raise Exception("parameter natoms_per_cluster should be an array of length {}".format(nclusters))
        if moved_atoms is not None and len(moved_atoms) != nclusters:
            raise Exception("parameter moved_atoms should be None or an array of length {}".format(nclusters))
        self.natoms = natoms
        self.nclusters = nclusters
        self.natoms_per_cluster = natoms_per_cluster
        self.nframes_per_cluster = nframes_per_cluster
        self.test_model = test_model
        self.noise_level = noise_level
        self.noise_natoms = noise_natoms
        self.displacement = displacement
        self.feature_type = feature_type
        if self.feature_type == 'inv-dist':
            self.nfeatures = int(self.natoms * (self.natoms - 1) / 2)
        elif self.feature_type.startswith('cartesian'):
            self.nfeatures = 3 * self.natoms
        elif self.feature_type.startswith("compact-dist") and self.natoms > 3:
            self.nfeatures = 4 * (self.natoms - 4) + 6
        else:
            raise Exception("Unsupported feature type {}".format(self.feature_type))
        self.nsamples = self.nframes_per_cluster * self.nclusters
        self._delta = 1e-9
        self.moved_atoms = moved_atoms
        self.moved_atoms_noise = None

    def generate_data(self, xyz_output_dir=None):
        """
        Generate data [ nsamples x nfeatures ] and clusters labels [ nsamples ]
        """
        logger.debug("Selecting atoms for clusters ...")

        if self.moved_atoms is None:
            self.moved_atoms = []

            for cluster_idx in range(self.nclusters):
                # list of atoms to be moved in a selected cluster c
                moved_atoms_c = self._pick_atoms(self.natoms_per_cluster[cluster_idx], self.moved_atoms)
                self.moved_atoms.append(moved_atoms_c)

        if self.noise_natoms is not None:
            logger.debug("Selecting atoms for constant noise ...")
            self.moved_atoms_noise = self._pick_atoms(self.noise_natoms, self.moved_atoms)
        logger.info("Generating frames ...")
        conf0 = self._generate_conformation0()
        labels = np.zeros((self.nsamples, self.nclusters))
        data = np.zeros((self.nsamples, self.nfeatures))

        frame_idx = 0
        self._save_xyz(xyz_output_dir, "conf", conf0, moved_atoms=[y for x in self.moved_atoms for y in x])
        for cluster_idx in range(self.nclusters):

            for f in range(self.nframes_per_cluster):

                labels[frame_idx, cluster_idx] = 1
                conf = np.copy(conf0)

                # Move atoms in each cluster
                for moved_atom_idx, atom_idx in enumerate(self.moved_atoms[cluster_idx]):
                    self._move_an_atom(cluster_idx, conf, moved_atom_idx, atom_idx)

                # Add constant noise
                if self.noise_natoms is not None:
                    for atom_idx in self.moved_atoms_noise:
                        if frame_idx % 3 == 0:  # move noise atoms every 3rd frame
                            conf[atom_idx, :] += [10 * self.displacement, 10 * self.displacement,
                                                  10 * self.displacement]

                # Add random noise
                conf = self._perturb(conf)

                # Generate features
                if self.feature_type == "inv-dist":
                    features = self._to_inv_dist(conf)
                elif self.feature_type.startswith("cartesian"):
                    features = self._to_cartesian(conf)
                elif self.feature_type.startswith("compact-dist"):
                    features = self._to_compact_dist(conf)
                else:
                    raise Exception("Invalid feature type {}".format(self.feature_type))
                self._save_xyz(xyz_output_dir, "cluster{}_frame{}".format(cluster_idx, "0" + str(f) if f < 10 else f),
                               conf,
                               self.moved_atoms[cluster_idx])
                data[frame_idx, :] = features
                frame_idx += 1

        return data, labels

    def _pick_atoms(self, natoms_to_pick, moved_atoms):
        """
        Select atoms to be moved for each cluster
        OR
        Select atoms to be moved for constant noise
        """

        moved_atoms_c = []

        for a in range(natoms_to_pick):

            while True:
                atom_to_move = np.random.randint(self.natoms)
                if (atom_to_move not in moved_atoms_c) and \
                        (atom_to_move not in [y for x in moved_atoms for y in x]):
                    moved_atoms_c.append(atom_to_move)
                    break

        return moved_atoms_c

    def _generate_conformation0(self):

        conf = np.zeros((self.natoms, 3))
        for n in range(self.natoms):
            conf[n] = (np.random.rand(3, ) * 2 - 1)  # Distributed between [-1,1)

        return conf

    def _move_an_atom(self, cluster_idx, conf, moved_atom_idx, atom_idx):
        """
        Move an atom of a cluster
        :param cluster_idx:
        :param conf:
        :param moved_atom_idx: #The index of the moved atom for this cluster
        :param atom_idx:
        :return:
        """
        if self.test_model == 'linear':
            conf[atom_idx, :] += [self.displacement, self.displacement, self.displacement]

        elif self.test_model == 'non-linear':
            if moved_atom_idx == 0:
                conf[atom_idx, :] += [cluster_idx * self.displacement, 0,
                                      self.displacement - cluster_idx * self.displacement]
            else:
                conf[atom_idx, :] = self._move_an_atom_along_circle(cluster_idx, conf, moved_atom_idx, atom_idx)

        elif self.test_model == 'non-linear-random-displacement':
            if moved_atom_idx == 0:
                conf[atom_idx, :] += [cluster_idx * self.displacement + np.random.rand() * self.displacement, \
                                      0 + np.random.rand() * self.displacement, \
                                      self.displacement - cluster_idx * self.displacement + np.random.rand() * self.displacement]  # displacement of the first atom is random
            else:
                conf[atom_idx, :] = self._move_an_atom_along_circle(cluster_idx, conf, moved_atom_idx, atom_idx)

        elif self.test_model == 'non-linear-p-displacement':
            decision = np.random.rand()  # atoms move with [decision] probability
            if decision >= 0.5:
                if moved_atom_idx == 0:
                    conf[atom_idx, :] += [cluster_idx * self.displacement, 0,
                                          self.displacement - cluster_idx * self.displacement]
                else:
                    conf[atom_idx, :] = self._move_an_atom_along_circle(cluster_idx, conf, moved_atom_idx, atom_idx)

    def _move_an_atom_along_circle(self, cluster_idx, conf, moved_atom_idx, atom_idx):
        """
        Move an atom of a cluster along the circle whose center is the previous atom
        First in XY plane
        And next in YZ plane
        """
        if moved_atom_idx > 0:
            previous_atom_idx = self.moved_atoms[cluster_idx][moved_atom_idx - 1]

            radius = np.sqrt(np.sum((conf[atom_idx, 0:2] - conf[previous_atom_idx, 0:2]) ** 2))
            # direction of rotation in xyz plane is defined by atom index
            angle_of_rotation = (-1) ** moved_atom_idx * self.displacement / radius
            conf[atom_idx, 0:2] = conf[previous_atom_idx, 0:2] + \
                                  self._rotate(angle_of_rotation,
                                               conf[atom_idx] - conf[previous_atom_idx],
                                               [0, 1])[0:2]

            radius = np.sqrt(np.sum((conf[atom_idx, 1:3] - conf[previous_atom_idx, 1:3]) ** 2))
            # direction of rotation in yz planed is defined by cluster index
            angle_of_rotation = (-1) ** cluster_idx * self.displacement / radius
            conf[atom_idx, 1:3] = conf[previous_atom_idx, 1:3] + \
                                  self._rotate(angle_of_rotation,
                                               conf[atom_idx] - conf[previous_atom_idx],
                                               [1, 2])[1:3]

        return conf[atom_idx, :]

    def _rotate(self, phi, xyz, dims):

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        xyz = xyz.T  # to work for an N-dim array
        xy = xyz[dims]
        xyz[dims[0]] = (cos_phi * xy[0] + sin_phi * xy[1])
        xyz[dims[1]] = (-sin_phi * xy[0] + cos_phi * xy[1])
        xyz = xyz.T

        return xyz

    def _perturb(self, conf):

        for n in range(self.natoms):
            conf[n] += (np.random.rand(3, ) * 2 - 1) * self.noise_level

        return conf

    def _to_inv_dist(self, conf):

        feats = np.empty((self.nfeatures))
        idx = 0
        for n1, coords1 in enumerate(conf):
            for n2 in range(n1 + 1, self.natoms):
                coords2 = conf[n2]
                feats[idx] = 1 / np.linalg.norm(coords1 - coords2 + self._delta)
                idx += 1

        return feats

    def _to_compact_dist(self, conf):
        if self.natoms < 4:
            return self._to_inv_dist(conf)
        feats = np.empty((self.nfeatures))
        feats[0] = 1 / np.linalg.norm(conf[0] - conf[1] + self._delta)
        feats[1] = 1 / np.linalg.norm(conf[0] - conf[2] + self._delta)
        feats[2] = 1 / np.linalg.norm(conf[0] - conf[3] + self._delta)
        feats[3] = 1 / np.linalg.norm(conf[1] - conf[2] + self._delta)
        feats[4] = 1 / np.linalg.norm(conf[1] - conf[3] + self._delta)
        feats[5] = 1 / np.linalg.norm(conf[2] - conf[3] + self._delta)
        for n in range(4, len(conf)):
            # We need the distances to at least 4 other atoms
            # Here taking the previous 4 atoms in the sequence
            idx = 4 * (n - 4) + 6
            feats[idx] = 1 / np.linalg.norm(conf[n] - conf[n - 4] + self._delta)
            feats[idx + 1] = 1 / np.linalg.norm(conf[n] - conf[n - 3] + self._delta)
            feats[idx + 2] = 1 / np.linalg.norm(conf[n] - conf[n - 2] + self._delta)
            feats[idx + 3] = 1 / np.linalg.norm(conf[n] - conf[n - 1] + self._delta)
        return feats

    def _to_cartesian(self, conf):

        if "_rot" in self.feature_type:
            conf = self._random_rotation(conf)
        if "_trans" in self.feature_type:
            conf = self._random_translation(conf)

        feats = np.empty((self.nfeatures))
        idx = 0
        for n1, coords1 in enumerate(conf):
            feats[idx] = coords1[0]  # x
            idx += 1
            feats[idx] = coords1[1]  # y
            idx += 1
            feats[idx] = coords1[2]  # z
            idx += 1

        return feats

    def _random_rotation(self, xyz):
        """
        Randomly rotate each frame along each axis
        """
        # Random angles between 0 and 2pi
        phi, psi, theta = 2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand(), np.pi * np.random.rand()
        # see http://mathworld.wolfram.com/EulerAngles.html
        xyz = self._rotate(phi, xyz, [0, 1])  # rotate xy plane plane
        xyz = self._rotate(theta, xyz, [1, 2])  # rotate new yz plane
        xyz = self._rotate(psi, xyz, [0, 1])  # rotate new xy plane

        return xyz

    def _random_translation(self, xyz):
        """
        Randomly translate each frame along each axis ; does not support PBC
        """
        [dx, dy, dz] = 5 * (np.random.rand(3) - 0.5)  # random values within box size
        xyz[:, 0] += dx
        xyz[:, 1] += dy
        xyz[:, 2] += dz

        return xyz

    def feature_to_resids(self):

        if self.feature_type == 'inv-dist':
            return None  # TODO fix later; default anyway
        elif self.feature_type.startswith("cartesian"):
            mapping = []
            for a in range(self.natoms):
                mapping.append([a, a])  # x
                mapping.append([a, a])  # y
                mapping.append([a, a])  # z
            return np.array(mapping)
        elif self.feature_type.startswith('compact-dist') and self.natoms > 3:
            mapping = [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [2, 3]
            ]
            for a in range(4, self.natoms):
                mapping.append([a - 4, a])  # x
                mapping.append([a - 3, a])  # x
                mapping.append([a - 2, a])  # y
                mapping.append([a - 1, a])  # z
            return np.array(mapping)
        else:
            raise Exception("Unknown feature type {}".format(self.feature_type))

    def _save_xyz(self, xyz_output_dir, filename, conf, moved_atoms=None, scale=10):
        """

        :param xyz_output_dir:
        :param filename:
        :param conf:
        :param moved_atoms:
        :param scale: multiply atom coordinates with this number - useful for better rendering in e.g. VMD
        :return:
        """
        if xyz_output_dir is None:
            return
        if not os.path.exists(xyz_output_dir):
            os.makedirs(xyz_output_dir)
        filename = "{dir}/{name}.xyz".format(dir=xyz_output_dir, name=filename)
        conf = conf * scale
        with open(filename, 'w') as f:
            f.write("{natoms}\n\n".format(natoms=conf.shape[0]))
            for atom_idx, [x, y, z] in enumerate(conf):
                # The element declares the importance of this atom
                if moved_atoms is not None and atom_idx in moved_atoms:
                    element = "C"
                elif self.moved_atoms_noise is not None and atom_idx in self.moved_atoms_noise:
                    element = "O"
                else:
                    element = "H"
                f.write("{element}\t{x}\t{y}\t{z}\n".format(element=element, x=x, y=y, z=z))
