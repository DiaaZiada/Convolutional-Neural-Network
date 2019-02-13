#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 00:02:23 2019

@author: diaa
"""
import numpy as np




def compute_cost(AL, Y,parameters=None,lambd=None):
    """
    Function:
        Implement the cost function
        
    Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
        parameters -- list of model Ws&Bs
        lambd -- regularization hyperparameter, scalar
        
    Returns:
        cost -- cross-entropy cost
    """
    m = Y.shape[1]
    assert not(np.nan in AL)
    assert not(np.inf in AL)
    assert not(np.inf in AL*-1)
    logprobs = AL* np.multiply(np.log(AL),Y) + (AL-1) * np.multiply(np.log(1 - AL), 1 - Y)

    cost = 1./m * np.nansum(logprobs)
    
    L2_regularization_cost = 0

    if lambd:
        L = len(parameters) // 2
        for l in range(1, L+1):
            L2_regularization_cost += np.sum(np.square(parameters['FC']["W"+str(l)])) 
        L2_regularization_cost *= lambd / (2. * m)

    cost +=  L2_regularization_cost

    
    assert(cost.shape == ())
    
    return cost
