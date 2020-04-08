# Demystifying

This repository contains code for analyzing molecular simulations data, mainly using machine learning methods.  

# Dependencies
 * Python >= 2.7
 * Scikit-learn with its standard dependencies (numpy, scipy etc.)
 * biopandas (only for postprocessing)
 * MDTraj (only for a preprocessing when writing per frame importance)
 
 
We plan to enable installation via package managers such as conda. For now, include the __demystifying__ module in your pyton path or import it directly in your python project. Below is an example.
# Using the code

## As a standalone library

## Example code

(see demo.py)

```python



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
    "Below we list all features and their importance. Those with highest importance are good candidates for Collective Variables (CVs)")
for feature_index, importance in postprocessor.get_important_features(sort=True):
    if importance < 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        break
    logger.info("Feature %d has importance %.2f. Corresponds to residues %s", feature_index, importance,
                feature_to_resids[int(feature_index)])

```

## Detailed guidelines

#### Pre-processing
* Extract features from your simulation trajectories with e.g. [MDTraj](http://mdtraj.org/) or [MDAnalysis](https://www.mdanalysis.org/). The output should be a 2D numpy array of shape (n_frames, n_features), i.e. the frames are stacked along the first axis and the different features along the second axis. We'll refer to this array as _samples_. 
  * We suggest you start by taking inter residue distances as features ([compute_contacts](http://mdtraj.org/1.9.3/api/generated/mdtraj.compute_contacts.html?highlight=compute%20contacts#mdtraj.compute_contacts) in MDTraj). 

**Optional steps**
* If you can label the frames in your trajectory according to which class/cluster/state it belongs to, then do so.  Assign an index to every class. Either make it a 1D array of shape (n_frames,) where every entry is the corresponding class index, or a 2D array of shape (n_frames, n_classes). We'll call this array _labels_. If a certain frame belongs to a class, then the corresponding entry has a 1. All other entries are 0. 
* Create an array, _feature_to_res_ids_, of shape (n_features, number_of_residues_involved_in_a_feature). For distance based features this will be a (n_features, 2) matrix, where you list the two residues that you measure the distance between.
* Save the numpy arrays to disk for quick loading in future analysis. This can speed things up and makes it easier to share your code without providing access to large trajectories.
  

**Considerations**
* If your simulation frames cannot be properly labelled you should use unsupervised learning. You can easily switch between supervised and unsupervised techniques with demystifying by using different FeatureExtractors.
* We recommend you to perform dimensionality reduction and project your data onto one or two dimensions. See if your data separates into well-defined states. There is limited support in _demystifying_ for this right now, but it's coming soon. For now, we recommend you to try the manifold learning techniques such as PCA, Multi-dimensional Scaling and t-SNE, provided by scikit-learn (https://scikit-learn.org/stable/modules/manifold.html).


**Finding Collective Varibales (CVs) with demystifying**
* See demo code above.
* Demystiyfing uses cross validation to reduce the risk of overfitting. However, for high dimensional or heavily correlated data there is always a risk overfitting. We recommend you comparing different methods and evaluating the CVs with biophyiscal  


## Analyzing biological systems
The biological systems discussed in the paper (the beta2 adrenergic receptor, the voltage sensor domain (VSD) and Calmodulin (CaM)) come with independent run files. These can be used as templates for other systems. 

Input data can be downloaded at [here](https://drive.google.com/drive/folders/19V1mXz7Yu0V_2JZQ8wtgt7aZusAKs2Bb?usp=sharing).

## Benchmarking with a toy model
Start __run_benchmarks.py__ to run the benchmarks discussed in the paper. This can be useful to test different hyperparameter setups as well as to enhance ones understanding of how different methods work.

__run_toy_model__ contains a demo on how to launch single instances of the toy model. This script is currently not maintained.

# Citing this work
Please cite the following paper:

Fleetwood, Oliver, et al. "Molecular insights from conformational ensembles via machine learning." Biophysical Journal (2019). 
[10.1016/j.bpj.2019.12.016](https://doi.org/10.1016/j.bpj.2019.12.016)


The code is also citable with DOI [10.5281/zenodo.3269704](https://doi.org/10.5281/zenodo.3269704).

# Support
Please open an issue or contact oliver.fleetwood (at) gmail.com it you have any questions or comments about the code. 

# Checklist for interpreting molecular simulations with machine learning 

---

1. Identify the problem to investigate

2. Decide if you should use supervised or unsupervised machine learning (or both)

    a. The best choice depends on what data is available and the problem at hand

    b. If you chose unsupervised learning, consider also clustering the simulation frames to label them and perform supervised learning 

3. Select a set of features and scale them

    a. For many processes, protein internal coordinates are adequate. To reduce the number of features, consider filtering distances with a cutoff
    
    b. Consider other features that can be expressed as a function of internal coordinates you suspect to be important for the process of interest (dihedral angles, cavity or pore hydration, ion or ligand binding etc...)

4. Chose a set of ML methods to derive feature importance

    a. To quickly get a clear importance profile with little noise, consider RF or KL for supervised learning. RF may perform better for noisy data.  

    b. For unsupervised learning, consider PCA, which is relatively robust when conducted on internal coordinates

    c. To find all important features, including those requiring nonlinear transformations of input features, also use neural network based approaches such as MLP. This may come at the cost of more peaks in the importance distribution

    d. Decide if you seek the average importance across the entire dataset (all methods), the importance per state (KL, a set of binary RF classifiers or MLP), or the importance per single configuration (MLP, RBM, AE) 

    e. Chose a set of hyperparameters which gives as reasonable trade off between performance and model prediction accuracy

5. Ensure that the selected methods and hyperparameter choice perform well under cross-validation

6. Average the importance per feature over many iterations

7. Check that the distribution of importance has distinguishable peaks

8. To select low-dimensional, interpretable CVs for plotting and enhanced sampling, inspect the top-ranked features 

9. For a holistic view, average the importance per residue or atom and visualize the projection on the 3d system 

10. If necessary, iterate over steps 3-9 with different features, ML methods and hyperparameters 

---
