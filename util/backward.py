#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 00:00:52 2019

@author: diaa
"""
import numpy as np
from util.func import * 

def linear_activation_backward(dA, cache, activation,lambd=None):
    """
    Function:
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        lambd -- regularization hyperparameter, scalar
        
    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)

    return dA_prev, dW, db



def L_model_backward(AL, Y, caches,lambd=None,keep_prob=None,Ds=None):
    """
    Function:
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        lambd -- regularization hyperparameter, scalar
        keep_prob -- probability of keeping a neuron active during drop-out, scalar.
        Ds -- list of dropout neurons
        
    Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
    """
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    dAL = np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)
    dAL = np.nan_to_num(dAL)

    current_cache = caches[-1]

    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
  
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA = grads["dA" + str(l + 2)]

        if keep_prob:
            D = Ds['D'+str(l+1)]

            dA *= D
            dA /= keep_prob
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, current_cache, activation="relu",lambd=lambd)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads, dA_prev_temp


