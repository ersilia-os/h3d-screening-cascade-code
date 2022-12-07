# h3d-screening-cascade-code
Code for the manuscript entitled "First fully automated AI/ML virtual screening cascade implemented at a drug discovery centre in Africa"

An outline of the code in each python file is provided below:

* auroc_improvement.py: Find the change in AUROC between two models relative to a perfect model (AUROC = 1) and the associated standard error.

* binarizer.py: Classify experimental data points _inclusive of assay cutoff values_ into a binary label, according to whether a large or small experimental readout is the 'active' outcome.

* cross_val.py: Find the mean roc curve for a given number of data folds, as well as the median auroc score and standard error.

* ecfp_descriptor.py: Featurise molecule SMILES into circular Morgan Fingerprints 2048 length with a radius of 3.

* hit_enrichment.py: Calculate the additional active compounds found by using model predictions after having tested different fractions of the dataset.

* model_performance.py: Traditional SKlearn ML performance metrics.

* projections.py: Dimensionality reduction methods (PCA and UMAP) to project high-dimensional molecule descriptions onto two-dimensional space for visualisation.
