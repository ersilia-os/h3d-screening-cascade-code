#Model performance metrics based on the SciKit-Learn implementations of the algorithms 

from sklearn import metrics
import numpy as np

def roc(binary_true, prediction_score):
    fpr, tpr, _ = metrics.roc_curve(binary_true, prediction_score)
    return fpr, tpr, _
    
    
def auroc_score(binary_true, prediction_score):
    auc = metrics.roc_auc_score(binary_true, prediction_score)
    return auc

#Calculate the auroc score for each fold, then return the median value 
#and standard deviation and standard error
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

#Compare true labels with binarised predictions
def confusion_plot(binary_true, binary_prediction):
    class_names = ['Inactives (0)', 'Actives (1)']
    confusion = metrics.confusion_matrix(binary_true, binary_prediction, labels = class_names)
    return confusion
    
