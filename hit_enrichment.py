#Generate hit enrichment curves that describe the rate at which compounds with an #active '1' 
#assay outcome are selected for testing and found active in an experimental assay.

import numpy as np

def hit_enrichment(dataframe, prediction_column="proba1"):
    #First rank compounds by prediction score
    data.sort_values(by=proba_col, axis=0, ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    #Count molecules
    total_mols = len(data)
    active_mols = len(data[data["bin"]==1])
    
    #Calculate cumulative fraction at each subset size of the test set, x-coords in plot 
    frac_tested = np.arange(1,total_mols+1, 1)/total_mols
    
    #3 sets of y-coords:
    #The proportion of active compounds identified by using model predictions to prioritise molecules
    hit_enr = [x/active_mols for x in np.cumsum(data["bin"])]
    
    #The perfect case where all active compounds are selected and measured experimentally first
    actives_first = [x+1 for x in range((active_mols))] 		#cumulative sum of actives
    inactives_last = [active_mols]*(len(data[data["bin"]==0])) 	#for each inactive, the maximum
    cumulative_actives = actives_first + inactives_last 		#cumulative sum of actives at each fraction of the data
    ideal=[x/active_mols for x in cumulative_actives]
    
    #The worst case where compounds are selected randomly and the background activity rate is achieved.
    random =  np.cumsum([active_mols/total_mols]*total_mols) / active_mols
    
    return frac_tested, hit_enr, ideal, random
