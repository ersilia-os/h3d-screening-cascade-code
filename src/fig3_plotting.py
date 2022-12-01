import os
import numpy as np
import math
import joblib

import umap 
from sklearn.decomposition import PCA

from stylia import MARKERSIZE_BIG, MARKERSIZE_SMALL, FONTSIZE_BIG, FONTSIZE, LINEWIDTH_THICK
from stylia import NamedColors

named_colors = NamedColors()
GRAY = named_colors.get('gray') #background molecules
PATH_ = "../data/tmp"


def PCA_plot(ax, compound_df, cmap, disease, path_ = PATH_):
    file_path =(os.path.join(path_,"tmp_{0}_pca_data.joblib".format(disease)))
    if os.path.exists(file_path):
        data = joblib.load(file_path)
    else:
        pca_mol = PCA(n_components = 2)
        pca_mol.fit(compound_df["fingerprint"].tolist())
        temp_df = compound_df[compound_df["source"] == "global"]
        data = (temp_df, pca_mol)
        joblib.dump(data, file_path)
    
    temp_df, pca_mol = data[0], data[1]
    
    principalComponents_mol = pca_mol.transform(temp_df["fingerprint"].tolist())
    x_axis = np.transpose(principalComponents_mol)[0]
    y_axis = np.transpose(principalComponents_mol)[1]
    ax.scatter(x_axis, y_axis, color=GRAY, alpha=0.8, s = MARKERSIZE_SMALL/16)
    
    cmap_count = 0
    for s in set(compound_df["source"].tolist()):
        temp_df = compound_df[compound_df["source"] == s]
        principalComponents_mol = pca_mol.transform(temp_df["fingerprint"].tolist())
        x_axis = np.transpose(principalComponents_mol)[0]
        y_axis = np.transpose(principalComponents_mol)[1]
        
        if s != "global":
            ax.scatter(x_axis, y_axis, color=cmap[cmap_count], s = MARKERSIZE_SMALL/8, alpha=0.8)
            cmap_count = cmap_count+1

def UMAP_plot(ax, compound_df, cmap, disease, path_=PATH_):
    file_path = (os.path.join(path_,"tmp_{0}_umap_data.joblib".format(disease)))
    if os.path.exists(file_path):
        data = joblib.load(file_path)
    else:
        if disease == "nf54":
            umap_transformer = umap.UMAP(a=0.01, b=1.8)
        elif disease == "mtb":
            umap_transformer = umap.UMAP(a=0.001, b=3)
        umap_transformer.fit(compound_df["fingerprint"].tolist())
        temp_df = compound_df[compound_df["source"] == "global"]
        data = (temp_df, umap_transformer)
        joblib.dump(data, file_path)
    
    temp_df, umap_transformer = data[0], data[1]
    umap_X = umap_transformer.transform(temp_df["fingerprint"].tolist())
    
    ax.scatter(umap_X.T[0], umap_X.T[1], color = GRAY, alpha=0.8, s = MARKERSIZE_SMALL/16)
    
    cmap_count = 0
    for s in set(compound_df["source"].tolist()):
        temp_df = compound_df[compound_df["source"] == s]
        umap_X = umap_transformer.transform(temp_df["fingerprint"].tolist())
        
        if s != "global":
            ax.scatter(umap_X.T[0], umap_X.T[1], color = cmap[cmap_count], alpha=0.8, s = MARKERSIZE_SMALL/8)
            cmap_count = cmap_count+1

def training_points(ax, scores_avg_dict, cmap, training_points, series):    
    ax.set_xlim(-3, 103)
    ax.set_ylim(0.38, 1.05)

    ax.set_xticks([0, 10, 30, 60, 100])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    x = [int(p) for p in training_points]
    for i, s in enumerate(series):
        ax.plot(x, scores_avg_dict[s][:-1], color=cmap[i], label = s, linewidth=LINEWIDTH_THICK)
        ax.scatter(x, scores_avg_dict[s][:-1], color=cmap[i])

def auroc_improvement(ax, scores_avg_dict, std_devs_dict, cmap, series, folds):
    ax.set_title("", fontsize=FONTSIZE_BIG)
    ax.set_xlim(0, len(series)*1.6)
    ax.set_xticks([])
    ax.axhline(y=0,linewidth=0.5, color='k', dashes=[1,0,1])

    ax2=ax.twinx()
    ax2.grid(False)
    ax2.set_ylim(0.45, 1.05)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.tick_params(width=0.5)
    ax2.set_ylabel("AUROC", fontsize=FONTSIZE)

    loc = 1.0
    spacing = 1
    for i, s in enumerate(series):
        delta = scores_avg_dict[s][-2]-scores_avg_dict[s][-1]
        val = scores_avg_dict[s][-1]
        delta = delta/(1-val)
        ax.bar(loc, delta*100, width=spacing, color=cmap[i])
        error_incl_global = (std_devs_dict[s][-2] / math.sqrt(folds))/scores_avg_dict[s][-2]
        error_excl_global = (std_devs_dict[s][-1] / math.sqrt(folds))/scores_avg_dict[s][-1]
        combined_error = (error_incl_global + error_excl_global)*delta
        ax.errorbar(loc, delta*100, combined_error*100, color='k', capsize=4.0, capthick = 0.5)
        ax2.scatter(loc, scores_avg_dict[s][-2], color=cmap[i], edgecolors='white', marker='o', s=MARKERSIZE_BIG)
        loc += 1.5
