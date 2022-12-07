import os
import numpy as np
import math

import Ecfp
from Ecfp_descriptor import Ecfp

from sklearn.decomposition import PCA
import umap

def pca_projection(compounds_df1, compounds_df2):
    #Featurise with circular fingerprints
    ecfp = Ecfp()
    fingerprints1 = ecfp.calc(compounds_df1["SMILES"])
    fingerprints2 = ecfp.calc(compounds_df2["SMILES"])

    #Construct 2D PCA space
    pca = PCA(n_components = 2)
    pca.fit(fingerprints1 + fingerprints2)
    
    #Project each dataset to PCA space
    fps1_pca = pca.transform(fingerprints1)
    fps2_pca = pca.transform(fingerprints2)

    #Get new 2D coords per molecule (row)  
    fps1_pca_trn = np.transpose(fps1_pca)
    fps2_pca_trn = np.transpose(fps2_pca)
    
    return fps1_pca_trn, fps2_pca_trn

def umap_projection(compounds_df1, compounds_df2):
    #Featurise with circular fingerprints
    ecfp = Ecfp()
    fingerprints1 = ecfp.calc(compounds_df1["SMILES"])
    fingerprints2 = ecfp.calc(compounds_df2["SMILES"])

    #Construct 2D PCA space
    umap_proj = umap.UMAP()
    umap_trn.fit(fingerprints1 + fingerprints2)
    
    #Project each dataset to PCA space
    fps1_umap = umap.transform(fingerprints1)
    fps2_umap = umap.transform(fingerprints2)

    #Get new 2D coords per molecule (row)  
    fps1_umap_trn = np.transpose(fps1_umap)
    fps2_umap_trn = np.transpose(fps2_umap)
    
    return fps1_umap_trn, fps2_umap_trn
    
