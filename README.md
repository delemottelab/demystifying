# Demystifying

This repository contains code for analyzing molecular simulations data, mainly using machine learning methods.  

# Dependencies
 * Python >= 2.7
 * Scikit-learn with its standard dependencies (numpy, scipy etc.)
 * biopandas (only for postprocessing)
 * MDTraj (only for a preprocessing when writing per frame importance)
 
 
We are working on upgrading the project to python 3 as well as enabling installation of dependencies via package managers such as conda, pip and similar. 

# Using the code

## As a standalone library
Include the __demystifying__ module in your pyton path or import it directly in your python project. Below is an example.

### Example code

See demo.py for a working example with toy model data.

```python
from demystifying import feature_extraction as fe, visualization
"""
Load your data samples (input features) and labels (cluster indices etc.) here 
"""

# Create a feature extractor. All extractors implement the same methods, but in this demo we use a Random Forest 
extractor = fe.RandomForestFeatureExtractor(samples, labels)
extractor.extract_features()

# Do postprocessing to average the importance per feature into importance per residues
# As well as highlight important residues on a protein structure
postprocessor = extractor.postprocessing(working_dir="output/")
postprocessor.average()
postprocessor.evaluate_performance()
postprocessor.persist()

# Visualize the importance per residue with standard functionality
visualization.visualize([[postprocessor]])

```


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
