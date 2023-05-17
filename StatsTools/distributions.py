""" distributions

Provides likelihood functions for distributions and a method to assign the 
most likely distribution to observed numeric data.

discreteData used to determine whether data is discrete or not.
"""


import numpy as np
from scipy import special as sp
from scipy import stats
import sys


def discreteData(data : np.array, discreteProportion = .5) -> tuple[bool,str,float]:
    """
    discreteData tests if the data is integral, non-negative, and 
    proportion of classes less than 'discreteProportion' to determine if discrete

    returns a tuple of (whether_discrete,discrete_family,loss) where
    discrete_family=loss=None if not discrete

    """
    # Integer and Non-negative
    if data.dtype != "int32" or np.min(data) < 0 :
        return (False, None, None)
    buckets = np.unique(data, return_counts=True)
    # Proportion test
    if len(buckets) > discreteProportion*len(data):
        return (False, None, None)
    else:
        # Test discrete families
        families = ["geometric","poisson"]
        familyLoss = []

        # Geoemtric
        geoLoss = geometricLoss(data,buckets)
        familyLoss.append(geoLoss)
        
        # Poisson
        poiLoss = poissonLoss(data, buckets)
        familyLoss.append(poiLoss)

        minIndex = np.argmin(familyLoss)
        return (True, families[minIndex], familyLoss[minIndex])
   
def ctsData(data : np.array) -> tuple[str,float]:
    """
    Handles Continuous Data 

    Returns tuple of distribution family with lowest Loss,
    and corresponding loss.
    """
    sortedData = np.sort(data)
    families = ["exponential","gaussian"]
    familyLoss = []

    # Exponential
    expLoss = sys.float_info.max
    if sortedData[0]>=0:
        expLoss = exponentialLoss(sortedData)
    familyLoss.append(expLoss)
    # Gaussian 
    gauLoss = gaussianLoss(sortedData)
    familyLoss.append(gauLoss)

    # Find minimum
    minIndex = np.argmin(familyLoss)
    return (families[minIndex],familyLoss[minIndex])

def distributionOptimizer(data : np.array, discreteProportion = .5) -> tuple[str,float]:
    """
    Selects optimal distribution based on standardized loss

    Returns tuple ("Distribution_Family",Loss)
    """
    
    # Discrete Test
    discrete = discreteData(data, discreteProportion)
    if discrete[0] == False:
        return ctsData(data)
    else:
        return discrete

def geometricLoss(data : np.array, buckets : tuple[np.array,np.array]) -> float:
    """
    Geometric Loss Function

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


def poissonLoss(data : np.array, buckets : np.array) -> float:
    """
    Poisson Loss Function

    Assumes buckets are non-negative integers
    """
    n = len(data)
    lambda_hat = np.mean(data)
    expectations = n*(lambda_hat**buckets[1]*np.exp(-lambda_hat))/sp.factorial(buckets[1])
    sd_hat = lambda_hat**0.5
    relativeLoss = np.sum((expectations-buckets[1])**2)/(sd_hat/n**0.5)
    return relativeLoss


def gaussianLoss(data : np.array) -> float:
    """
    Gaussian Loss Function

    Assumes data sorted.
    """
    n = len(data)
    mu_hat = np.mean(data);
    sd_hat = 1/n*np.sum((data-mu_hat)**2)
    # Goodness-of-fit test : 
    # Squared difference between data and quantile at theoretical
    percentiles = np.append(np.linspace(1/n,1-1/n,n-1),1-1/n**2)
    norm_dist = stats.norm(mu_hat,sd_hat)
    relativeLoss = np.sum((data-norm_dist.pdf(percentiles))**2)/(sd_hat/n**0.5)
    return relativeLoss


def exponentialLoss(data : np.array) -> float:
    """
    Exponential Loss Function

    Assumes data sorted.
    """
    n = len(data)
    lambda_hat = 1/np.mean(data)
    sd_hat = 1/lambda_hat
    # Goodness-of-fit test :
    percentiles = np.append(np.linspace(1/n,1-1/n,n-1),1-1/n**2)
    exp_dist = stats.expon(lambda_hat,sd_hat)
    relativeLoss = np.sum((data-exp_dist.pdf(percentiles))**2)/(sd_hat/n**0.5)
    return relativeLoss


    
    