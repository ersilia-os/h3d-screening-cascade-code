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


#Calculate the mean roc curve and the roc_curves 1 std dev above/below the mean
def roc_folds(binary_true_list, prediction_score_list):
    tprs = [] #list of true positive rates from each fold
    aucs = [] #auroc score for each fold
    folds = len(binary_true_list)
    
    mean_fpr = np.linspace(0,1,100) #Interpolate 100 points in roc curve
    
    for i in range(folds):
        fpr, tpr, _ = metrics.roc_curve(binary_true_list[i], prediction_score_list[i])
        interp_tpr = np.interp(mean_fpr, fpr, tpr) 	#Interpolate roc curve at preset x_coords in mean_fpr
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)   			#Keep interpolated roc curve for averaging
        aucs.append(metrics.roc_auc_score(binary_true_list[i], prediction_score_list[i])) 
    
    #Get average roc curve at each x-coord
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    #Get std dev in roc y-axis for each x-coord
    std_auc = np.std(aucs)
    
    #Calculate 1 std dev difference above and below mean roc curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    return mean_tpr, tprs_upper, tprs_lower
