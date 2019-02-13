#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 00:00:42 2019

@author: diaa
"""
import numpy as np
from util.func import *

def linear_activation_forward(A_prev, W, b, activation):
    """
    Function:
        Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
    """
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache =  (linear_cache, activation_cache)

    return A, cache



def L_model_forward(X, parameters,keep_prob):
    """
    Function:
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", ...
        keep_prob -- probability of keeping a neuron active during drop-out, scalar
                   
    Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        Ds -- list of dropout neurons 
    """

   
    np.random.seed(1)

    caches = []
    Ds = {}
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        if keep_prob:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = D < keep_prob
            A *= D
            A /= keep_prob
            Ds["D"+str(l)] = D
        caches.append(cache)
        
    AL, cache= linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    
    caches.append(cache)
    return AL, caches, Ds

