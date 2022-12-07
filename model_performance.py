#Model performance metrics based on the SciKit-Learn implementations of the algorithms 

from sklearn import metrics
import numpy as np

def roc(binary_true, prediction_score):
    fpr, tpr, _ = metrics.roc_curve(binary_true, prediction_score)
    return fpr, tpr, _
    
def auroc_score(binary_true, prediction_score):
    auc = metrics.roc_auc_score(binary_true, prediction_score)
    return auc

#Compare true labels with binarised predictions
def confusion_plot(binary_true, binary_prediction):
    class_names = ['Inactives (0)', 'Actives (1)']
    confusion = metrics.confusion_matrix(binary_true, binary_prediction, labels = class_names)
    return confusion
    
