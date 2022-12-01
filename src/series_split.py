from sklearn import metrics
import os
import math
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def PCA_UMAP_preprocess(global_path, series_paths):
    #Load datasets
    background_df = pd.read_csv(global_path + ".csv")
    background_df["source"] = "global"
    series_df = []
    for series in series_paths:
        series_df.append(pd.read_csv(series_paths[series] + ".csv"))
        series_df[-1]['source'] = series
    
    #Annotate series for colouring later
    smiles_background_df = background_df[["CAN_SMILES", "source"]]
    combined_df = smiles_background_df
    for s in series_df:
        smiles_series_df = s[["CAN_SMILES", "source"]]
        combined_df = pd.concat([combined_df,smiles_series_df])
        combined_df.reset_index(level=None, drop=True, inplace=True, col_level=0, col_fill='')
    combined_df.drop_duplicates(subset ="CAN_SMILES", keep = 'last', inplace = True)
    
    fps_combined = []
    for s in combined_df["CAN_SMILES"]:
        mol = Chem.MolFromSmiles(s)
        fps_combined.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))
    
    df = pd.DataFrame(list(zip(fps_combined, combined_df["source"].tolist())), columns=["fingerprint", "source"])
    return df

def get_folds(series, tp, removed, folds, cutoff, path, preds_path):
    fpr_list, tpr_list = [], []
    for fold_num in range(1, folds+1):   
        test = pd.read_csv(os.path.join(path, series, "fold" + str(fold_num), series + "_test_100.csv"))
        X_test = test["SMILES"].tolist()
        
        if "bin" in test.columns:
            y_test = test["bin"]
        else:
            y_temp = test["exp"].tolist()
            y_binary = [1 if n < cutoff else 0 for n in y_temp]   ###Binary cutoff
            y_test = y_binary
        
        preds = pd.read_csv(os.path.join(preds_path, series, "fold" + str(fold_num), series + "_" + str(tp) + "_with_" + str(removed) + "_percent_global_removed.csv"))
           
        fpr, tpr = get_ROC_points(y_test, preds)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
               
    return fpr_list, tpr_list

def get_ROC_points(y, preds):
    clf = None
    for k in preds.keys():
        if "clf_" in k and "bin" not in k:
            clf = k
    
    fpr, tpr, _ = metrics.roc_curve(y, preds[clf])
    return fpr, tpr


def cross_val_data(series, folds, cutoff, training_points, path, preds_path):
    all_scores = {}
    scores_avg_dict = {}
    std_devs_dict = {}

    for s in series:
        scores_list = {}
        scores_avg_list = []
        std_devs_list = []

        for tp in training_points:
            fpr_list, tpr_list = get_folds(s, tp, 0, folds, cutoff, path, preds_path)
            auc = []
            for fpr, tpr in zip(fpr_list, tpr_list):
                auc.append(metrics.auc(fpr, tpr))
            scores_list[tp] = auc
            scores_avg_list.append(round(np.median(auc), 3))
            std_devs_list.append(round(np.std(auc), 3))

        auc = []
        fpr_list, tpr_list = get_folds(s, tp, 100, folds, cutoff, path, preds_path)
        for fpr, tpr in zip(fpr_list, tpr_list):
            auc.append(metrics.auc(fpr, tpr))
        scores_list[str(tp) + "_no_global"] = auc
        scores_avg_list.append(round(np.median(auc), 3))
        std_devs_list.append(round(np.std(auc), 3))

        all_scores[s] = scores_list
        scores_avg_dict[s] = scores_avg_list
        std_devs_dict[s] = std_devs_list
        
    return all_scores, scores_avg_dict, std_devs_dict