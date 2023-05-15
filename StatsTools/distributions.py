""" distributions

Provides likelihood functions for distributions and a method to assign the 
most likely distribution to observed numeric data.

discreteOrContinuous used to determine whether data is discrete or continuous,
based on the proportion of unique values
"""

import numpy as np

def discreteOrContinuous(data : np.array, modifier = 0.2):
    discreteModifier = modifier
    buckets = np.unique(data)
    

def geometricLH(data : np.array) -> float:
    """
    Geometric Likelihood Function
    """
    
    