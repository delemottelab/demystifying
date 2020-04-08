from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import numpy as np
from operator import itemgetter
from biopandas.pdb import PandasPdb
from . import utils
from . import filtering
from . import data_projection as dp

logger = logging.getLogger("postprocessing")


class PostProcessor(object):

    def __init__(self,
                 extractor=None,
                 working_dir=None,
                 rescale_results=True,
                 filter_results=False,
                 feature_to_resids=None,
                 pdb_file=None,
                 accuracy_method='mse',
                 predefined_relevant_residues=None,
                 use_GMM_estimator=True):
        """
        Class which computes all the necessary averages and saves them as fields
        TODO move some functionality from class feature_extractor here
        :param extractor:
        :param feature_importance:
        :param std_feature_importance:
        :param cluster_indices:
        :param working_dir:
        :param feature_to_resids: an array of dimension nfeatures*2 which tells which two residues are involved in a feature
        """
        self.extractor = extractor
        self.feature_importances = extractor.feature_importance
        self.std_feature_importances = extractor.std_feature_importance
        self.supervised = extractor.supervised
        self.cluster_indices = extractor.cluster_indices
        self.nclusters = 1 if extractor.labels is None else extractor.labels.shape[1]
        self.working_dir = working_dir
        if self.working_dir is None:
            self.working_dir = os.getcwd()
        self.pdb_file = pdb_file
        self.predefined_relevant_residues = predefined_relevant_residues
        self.use_GMM_estimator = use_GMM_estimator

        # Rescale and filter results if needed
        self.rescale_results = rescale_results
        if self.feature_importances is not None:
            if rescale_results:
                self.feature_importances, self.std_feature_importances = utils.rescale_feature_importance(
                    self.feature_importances, self.std_feature_importances)
            if filter_results:
                self.feature_importances, self.std_feature_importances = filtering.filter_feature_importance(
                    self.feature_importances, self.std_feature_importances)

            # Put importance and std to 0 for residues pairs which were filtered out during features filtering (they are set as -1 in self.feature_importances and self.std_feature_importances)
            self.indices_filtered = np.where(self.feature_importances[:, 0] == -1)[0]
            self.feature_importances[self.indices_filtered, :] = 0
            self.std_feature_importances[self.indices_filtered, :] = 0
            # Set mapping from features to residues
            self.nfeatures = self.feature_importances.shape[0]
        else:
            self.indices_filtered = np.empty((0, 0))
            self.nfeatures = self.extractor.samples.shape[1]

        if feature_to_resids is None and self.pdb_file is None:
            feature_to_resids = utils.get_default_feature_to_resids(self.nfeatures)
        elif feature_to_resids is None and self.pdb_file is not None:
            feature_to_resids = utils.get_feature_to_resids_from_pdb(self.nfeatures, self.pdb_file)
        self.feature_to_resids = feature_to_resids
        self.accuracy_method = accuracy_method

        # Set average feature importances to None
        self.importance_per_residue_and_cluster = None
        self.std_importance_per_residue_and_cluster = None
        self.importance_per_residue = None
        self.std_importance_per_residue = None

        # Performance metrics
        self.predefined_relevant_residues = predefined_relevant_residues
        self.average_std = None
        if extractor.test_set_errors is not None:
            self.test_set_errors = extractor.test_set_errors.mean()
        else:
            self.test_set_errors = None
        self.data_projector = None
        self.separation_score = None
        self.accuracy = None
        self.accuracy_per_cluster = None
        self._importance_mapped_to_resids = None
        self._std_importance_mapped_to_resids = None

    def average(self):
        """
        Computes average importance per cluster and residue and residue etc.
        Sets the fields importance_per_residue_and_cluster, importance_per_residue
        :return: itself
        """
        self._map_feature_to_resids()
        self._compute_importance_per_residue()

        if self.supervised:
            self._compute_importance_per_residue_and_cluster()

        return self

    def evaluate_performance(self):
        """
        Computes -average of standard deviation (per residue)
                 -projection classification entropy
                 -classification score (for toy model only)
        """
        self._compute_average_std()
        self._compute_projection_classification_entropy()

        if self.predefined_relevant_residues is not None:
            self.compute_accuracy()

        return self

    def get_important_features(self, states=None, sort=True):
        """
        :param states: (optional) the indices of the states
        :param sort: (optional) sort the features by their importance
        :return: np.array of shape (n_features, 2) with entries (feature_index, importance)
        """
        fi = self.feature_importances
        if states is not None and self.supervised:
            fi = fi[:, states]
        fi = fi.sum(axis=1)
        fi, _ = utils.rescale_feature_importance(fi)
        fi = fi.squeeze()
        fi = [(e, i) for (e, i) in enumerate(fi)]
        if sort:
            fi = [(e, i) for (e, i) in sorted(fi, key=itemgetter(1), reverse=True)]
        return np.array(fi)

    def persist(self):
        """
        Save .npy files of the different averages and pdb files with the beta column set to importance
        :return: itself
        """
        directory = self.get_output_dir()

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + "importance_per_residue", self.importance_per_residue)
        np.save(directory + "std_importance_per_residue", self.std_importance_per_residue)
        np.save(directory + "feature_importance", self.feature_importances)
        np.save(directory + "std_feature_importance", self.std_feature_importances)

        if self.importance_per_residue_and_cluster is not None and self.std_importance_per_residue_and_cluster is not None:
            np.save(directory + "importance_per_residue_and_cluster", self.importance_per_residue_and_cluster)
            np.save(directory + "std_importance_per_residue_and_cluster", self.std_importance_per_residue_and_cluster)
        if self.separation_score is not None:
            np.save(directory + 'separation_score', self.separation_score)
        if self.predefined_relevant_residues is not None:
            np.save(directory + "predefined_relevant_residues", self.predefined_relevant_residues)
        if self.accuracy is not None:
            np.save(directory + 'accuracy', self.accuracy)
        if self.accuracy_per_cluster is not None:
            np.save(directory + 'accuracy_per_cluster', self.accuracy_per_cluster)
        if self.test_set_errors is not None:
            np.save(directory + 'test_set_errors', self.test_set_errors)
        if self.feature_to_resids is not None:
            np.save(directory + 'feature_to_resids', self.feature_to_resids)
        if self.pdb_file is not None:
            pdb = PandasPdb()
            pdb.read_pdb(self.pdb_file)
            self._save_to_pdb(pdb, directory + "importance.pdb",
                              self._map_to_correct_residues(self.importance_per_residue))

            if self.importance_per_residue_and_cluster is not None:
                for cluster_idx, importance in enumerate(self.importance_per_residue_and_cluster.T):
                    cluster_name = "cluster_{}".format(cluster_idx) \
                        if self.extractor.label_names is None else \
                        self.extractor.label_names[cluster_idx]
                    self._save_to_pdb(pdb, directory + "{}_importance.pdb".format(cluster_name),
                                      self._map_to_correct_residues(importance))

        return self

    def _load_if_exists(self, filepath):
        if os.path.exists(filepath):
            return np.load(filepath)
        else:
            return None

    def get_output_dir(self):
        return self.working_dir + "/{}/".format(self.extractor.name)

    def load(self):
        """
        Loads files dumped by the 'persist' method
        :return: itself
        """
        directory = self.get_output_dir()

        if not os.path.exists(directory):
            return self

        self.importance_per_residue = np.load(directory + "importance_per_residue.npy")
        self.std_importance_per_residue = np.load(directory + "std_importance_per_residue.npy")
        self.feature_importances = np.load(directory + "feature_importance.npy")
        self.std_feature_importances = np.load(directory + "std_feature_importance.npy")

        self.importance_per_residue_and_cluster = self._load_if_exists(
            directory + "importance_per_residue_and_cluster.npy")
        self.std_importance_per_residue_and_cluster = self._load_if_exists(
            directory + "std_importance_per_residue_and_cluster.npy")
        self.separation_score = self._load_if_exists(directory + "separation_score.npy")
        self.predefined_relevant_residues = self._load_if_exists(directory + "predefined_relevant_residues.npy")
        self.accuracy = self._load_if_exists(directory + "accuracy.npy")
        self.accuracy_per_cluster = self._load_if_exists(directory + "accuracy_per_cluster.npy")
        self.test_set_errors = self._load_if_exists(directory + "test_set_errors.npy")
        if self.feature_to_resids is None:  # Can be useful to override this in postprocesseing
            self.feature_to_resids = self._load_if_exists(directory + "feature_to_resids.npy")

        np.unique(np.asarray(self.feature_to_resids.flatten()))
        return self

    def _map_feature_to_resids(self):
        # Create array of all unique reside numbers
        index_to_resid = self.get_index_to_resid()
        self.nresidues = len(index_to_resid)
        res_id_to_index = {}  # a map pointing back to the index in the array index_to_resid
        for idx, resid in enumerate(index_to_resid):
            res_id_to_index[resid] = idx  # Now we now which residue points to which feature

        _importance_mapped_to_resids = np.zeros((self.nresidues, self.feature_importances.shape[1]))
        _std_importance_mapped_to_resids = np.zeros((self.nresidues, self.feature_importances.shape[1]))
        for feature_idx, rel in enumerate(self.feature_importances):
            corresponding_residues = self.feature_to_resids[feature_idx]
            if isinstance(corresponding_residues, np.number):
                # Object not iterable, i.e. we only have one residue per features
                corresponding_residues = [corresponding_residues]
            for res_seq in corresponding_residues:
                r_idx = res_id_to_index[res_seq]
                _importance_mapped_to_resids[r_idx, :] += rel
                _std_importance_mapped_to_resids[r_idx, :] += self.std_feature_importances[feature_idx, :] ** 2
        _std_importance_mapped_to_resids = np.sqrt(_std_importance_mapped_to_resids)
        self._importance_mapped_to_resids = _importance_mapped_to_resids
        self._std_importance_mapped_to_resids = _std_importance_mapped_to_resids

    def _compute_importance_per_residue(self):

        importance_per_residue = self._importance_mapped_to_resids.mean(axis=1)
        std_importance_per_residue = np.sqrt(np.mean(self._std_importance_mapped_to_resids ** 2, axis=1))

        if self.rescale_results:
            # Adds a second axis to feed to utils.rescale_feature_importance
            importance_per_residue = importance_per_residue.reshape((importance_per_residue.shape[0], 1))
            std_importance_per_residue = std_importance_per_residue.reshape((std_importance_per_residue.shape[0], 1))
            importance_per_residue, std_importance_per_residue = utils.rescale_feature_importance(
                importance_per_residue, std_importance_per_residue)
            importance_per_residue = importance_per_residue[:, 0]
            std_importance_per_residue = std_importance_per_residue[:, 0]

        self.importance_per_residue = importance_per_residue
        self.std_importance_per_residue = std_importance_per_residue

    def _compute_importance_per_residue_and_cluster(self):
        if self.rescale_results:
            self._importance_mapped_to_resids, self._std_importance_mapped_to_resids = utils.rescale_feature_importance(
                self._importance_mapped_to_resids, self._std_importance_mapped_to_resids)

        self.importance_per_residue_and_cluster = self._importance_mapped_to_resids
        self.std_importance_per_residue_and_cluster = self._std_importance_mapped_to_resids

    def _compute_average_std(self):
        """
        Computes average standard deviation
        """
        self.average_std = self.std_importance_per_residue.mean()

        return self

    def _compute_projection_classification_entropy(self):
        """
        Computes separation of clusters in the projected space given by the feature importances
        """
        if self.extractor.labels is None:
            logger.warning("Cannot compute projection classification entropy without labels")
            return
        if self.extractor.mixed_classes:
            logger.warning(
                "Cannot compute projection classification entropy for dataset where not all frames belong to a unique cluster/state.")
            return

        self.data_projector = dp.DataProjector(self.extractor.samples, self.extractor.labels)

        if self.supervised:
            self.data_projector.project(self.feature_importances).score_projection(use_GMM=self.use_GMM_estimator)
        else:
            self.data_projector.project(self.feature_importances)
            self.data_projector.separation_score = np.nan
        # self.separation_score = np.array([self.data_projector.separation_score])
        self.separation_score = self.data_projector.separation_score
        return self

    def compute_accuracy(self):
        """
        Computes accuracy with an normalized MSE based metric
        """
        if self.predefined_relevant_residues is None:
            logger.warn("Cannot compute accuracy without predefined relevant residues")
            return
        relevant_residues_all_clusters = [y for x in self.predefined_relevant_residues for y in x]
        if self.accuracy_method == 'mse':
            self.accuracy = utils.compute_mse_accuracy(self.importance_per_residue,
                                                       relevant_residues=relevant_residues_all_clusters)
        elif self.accuracy_method == 'relevant_fraction':
            self.accuracy = utils.compute_relevant_fraction_accuracy(self.importance_per_residue,
                                                                     relevant_residues=relevant_residues_all_clusters)
        else:
            raise Exception("Invalid accuracy method {}".format(self.accuracy_method))
        if self.supervised:
            self.accuracy_per_cluster = 0
            for i in range(self.nclusters):
                self.accuracy_per_cluster += utils.compute_mse_accuracy(self.importance_per_residue_and_cluster[:, i],
                                                                        relevant_residues=
                                                                        self.predefined_relevant_residues[i])
            self.accuracy_per_cluster /= self.nclusters

    def _map_to_correct_residues(self, importance_per_residue):
        """
        Maps importances to correct residue numbers
        """
        residue_to_importance = {}
        index_to_resid = self.get_index_to_resid()
        for idx, rel in enumerate(importance_per_residue):
            resSeq = index_to_resid[idx]
            residue_to_importance[resSeq] = rel

        return residue_to_importance

    def _save_to_pdb(self, pdb, out_file, residue_to_importance):
        """
        Saves importances into beta column of pdb file
        """
        atom = pdb.df['ATOM']
        missing_residues = []
        for i, line in atom.iterrows():
            resSeq = int(line['residue_number'])
            importance = residue_to_importance.get(resSeq, None)
            if importance is None:
                missing_residues.append(resSeq)
                importance = 0
            atom.at[i, 'b_factor'] = importance
        if len(missing_residues) > 0:
            logger.debug("importance is None for residues %s", [r for r in sorted(set(missing_residues))])
        pdb.to_pdb(path=out_file, records=None, gz=False, append_newline=True)

        return self

    def get_index_to_resid(self):
        return np.unique(np.asarray(self.feature_to_resids.flatten()))


class PerFrameImportancePostProcessor(PostProcessor):

    def __init__(self,
                 per_frame_importance_outfile=None,
                 frame_importances=None,
                 **kwargs):
        PostProcessor.__init__(self, **kwargs)
        self.per_frame_importance_outfile = per_frame_importance_outfile
        self.frame_importances = frame_importances

    def persist(self):
        PostProcessor.persist(self)
        if self.per_frame_importance_outfile is not None and \
                self.frame_importances is not None and self.pdb_file != None:
            with open(self.per_frame_importance_outfile, 'w') as of:
                logger.info("Writing per frame importance to file %s", self.per_frame_importance_outfile)
                self.to_vmd_file(of)

    def to_vmd_file(self, of):
        import mdtraj as md
        """
        writing VMD script, see https://www.ks.uiuc.edu/Research/vmd/mailing_list/vmd-l/5001.html
        :return:
        """
        if self.pdb_file is None:
            raise Exception("PDB file required to write per frame importance")

        # Map the feature to atoms for better performance
        top = md.load(self.pdb_file).top
        feature_to_atoms = []
        residue_to_atoms = {}
        for feature_idx, [res1, res2] in enumerate(self.feature_to_resids):
            atoms1 = residue_to_atoms.get(res1, None)
            if atoms1 is None:
                atoms1 = top.select("protein and resSeq {}".format(res1))
                residue_to_atoms[res1] = atoms1
            atoms2 = residue_to_atoms.get(res2, None)
            if atoms2 is None:
                atoms2 = top.select("protein and resSeq {}".format(res2))
                residue_to_atoms[res2] = atoms2
            feature_to_atoms.append(np.append(atoms1, atoms2))
        ##write to file in minibatches
        for frame_idx, importance in enumerate(self.frame_importances):
            # First normalize importance over features (not same as below)
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)
            # map importance to atom idx
            atom_to_importance = np.zeros((top.n_atoms))
            for feature_idx, atoms in enumerate(feature_to_atoms):
                fi = importance[feature_idx]
                for a in atoms:
                    atom_to_importance[a] += fi
            # Normalize to values between 0 and 1
            atom_to_importance = (atom_to_importance - atom_to_importance.min()) / \
                                 (atom_to_importance.max() - atom_to_importance.min() + 1e-6)
            # Go through atoms in sequential order
            lines = ["#Frame {}\n".format(frame_idx)] + ["{}\n".format(ai) for ai in atom_to_importance]
            of.writelines(lines)
