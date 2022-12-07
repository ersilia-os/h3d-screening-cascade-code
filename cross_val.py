import os
import numpy as np
import math
from sklearn import metrics

#Calculate the auroc score for each fold, then return 
#the median value with standard deviation and standard error

def auroc_folds(binary_true_list, prediction_score_list):
    auc_list = []
    num_folds = len(binary_true_list)
    
    for i in range(num_folds):
    	auc = metrics.roc_auc_score(binary_true_list[i], prediction_score_list[i])
    	auc_list.append(auc)
    
    median_score = round(np.median(auc_list), 3)
    std_dev = round(np.std(auc_list), 3)
    std_error = round(std_dev/np.sqrt(std_dev), 3)
    
    return median_score, std_dev, std_error
