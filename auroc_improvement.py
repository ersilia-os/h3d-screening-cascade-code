import os
import numpy as np
import math
from sklearn import metrics

#Compare the improvement in auroc scores between models with different training sets

#e.g. compare aurocs between a model trained on general training data alone versus a model 
#with additional data from the chemical space of interest

def auroc_improvement(auroc_original, auroc_improv, std_dev_original, std_dev_improv, folds):
    delta = auroc_improv - auroc_original			#auroc gained
    performance_available = 1 - auroc_original		#auroc score that is still possible to achieve
    proportion_increase = delta / performance_available 	#proportion of auroc score gained, out of the auroc still possible to achieve
    
    #relative standard errors for both models
    error_original = (std_dev_original / math.sqrt(folds)) / auroc_original
    error_improv = (std_dev_improv / math.sqrt(folds)) / auroc_improv
    
    #combined relative error in auroc improvement
    combined_error = (error_original + error_improv) * delta

    return proportion_increase, combined_error
