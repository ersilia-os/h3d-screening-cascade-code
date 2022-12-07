#Assign binary labels to experimental assay values
#depending on whether a 'high' or 'low' assay readout
#is deemed the active '1' outcome in the assay of interest

def binary_clf(assay, cutoff, dir):
    bin = []
    for value in assay:
    #activity is assigned to be inclusive of the cutoff criteria
        if dir == "high":
                if value >= cutoff:
                    b = 1
                else:
                    b = 0
        elif dir == "low":
            if value <= cutoff:
                b = 1
            else:
                b = 0
        bin += [b]
    return bin
