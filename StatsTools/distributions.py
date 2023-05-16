""" distributions

Provides likelihood functions for distributions and a method to assign the 
most likely distribution to observed numeric data.

discreteData used to determine whether data is discrete or not.
"""

import numpy as np
from scipy import special as sp
import sys

def discreteData(data : np.array, discreteProportion = .5) -> tuple[bool,tuple[np.array,np.array]]:
    """
    discreteData tests if the data is integral, non-negative, and 
    proportion of classes less than 'discreteProportion' to determine if discrete
    >>> discreteData(np.array([1,2,3,3,3,3]))
    (True, (array([1, 2, 3]), array([1, 1, 4], dtype=int64)))
    """
    # Integer and Non-negative
    if data.dtype != "int32" or np.min(data) < 0 :
        return (False, None)
    # Proportion test
    buckets = np.unique(data, return_counts=True)
    if len(buckets) > discreteProportion*len(data):
        return (False, None)
    else:
        return (True, buckets)
    

def geometricLH(data : np.array, buckets : tuple[np.array,np.array]) -> float:
    """
    Geometric Likelihood Function

    Accepts an np.array of data, and its buckets: (unique_values,counts)
    
    Assumes buckets are non-negative integers
    """
    n = len(data)
    # Begin at zero
    if buckets[0] == 0:
        p_hat = 1/np.mean(data)
        expectations = n*p_hat*(1-p_hat)**(buckets[0]-1)
    # Begin above zero
    else:
        p_hat = 1/(1+np.mean(data))
        expectations = n*p_hat*(1-p_hat)**buckets[0]
    
    
    sd_hat = ((1-p_hat)/p_hat**2)**0.5
    relativeLoss = np.sum((expectations-buckets[1])**2)/(sd_hat/n**0.5)
    return relativeLoss

def poissonLH(data : np.array, buckets : np.array) -> float:
    """
    Poisson Likelihood Function

    Assumes buckets are non-negative integers
    """
    n = len(data)
    lambda_hat = np.mean(data)
    expectations = n*(lambda_hat**buckets[1]*np.exp(-lambda_hat))/sp.factorial(buckets[1])
    sd_hat = lambda_hat**0.5
    relativeLoss = np.sum((expectations-buckets[1])**2)/(sd_hat/n**0.5)
    return relativeLoss



def multinomialLH(data : np.array, buckets = np.array) -> float:
    """
    Multinomial Likelihood Function
    """
    n = len(data)
    p_hat = buckets[1]/n
    expectations = sp.factorial(n)
    return False


    
    