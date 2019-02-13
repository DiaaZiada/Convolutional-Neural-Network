#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 23:56:51 2019

@author: diaa
"""


import numpy as np


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    assert Z.shape == (W.shape[0],A.shape[1])
    return Z, (A, W, b)

def sigmoid(Z):
    """
    Function:
        Implement the sigmoid activation function in numpy
    
    Arguments:
        Z -- Output of the linear layer, of any shape
    
    Returns:
        A -- output of sigmoid(Z), same shape as Z
        cache -- Z
        
    Raise:
        shape error if A shape not equal Z shape
   
    """
    A = 1. / (1. + np.exp(-Z))
    assert A.shape == Z.shape
    
    return A, Z

def relu(Z):
    """
    Function:
        Implement the RELU activation function in numpy

    Arguments:
        Z -- Output of the linear layer, of any shape
        
    Returns:
        A -- Output of relu(Z), same shape as Z
        cache -- Z 
        
    Raise:
        shape error if A shape not equal Z shape
    
    """
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    return A, Z



def linear_backward(dZ, cache, lambd=None):
    """
    Function:
        Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        lambd -- regularization hyperparameter, scalar

    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b

    Raise:
        shape error if dA_prev shape not equal A_prev shape
        shape error if dW_prev shape not equal W shape
        shape error if db_prev shape not equal b shape
        
    """
    
    
    A_prev, W, b = cache
    
    m = A_prev.shape[1]
    
    dA_prev = np.dot(W.T, dZ)
    
    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    if lambd:
        dW += (lambd * W) / m

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db




def sigmoid_backward(dA, cache):
    """
    Function:
        Implement derivative of sigmoid activations .

    Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ -- Gradient of the cost with respect to Z

    Raise:
        shape error if dZ shape not equal Z shape
    """
   
    Z = cache
    
    s = 1. / (1. + np.exp(-Z))
    
    dZ =  dA * s * (1. - s)
    
    assert dZ.shape == dA.shape 
    
    return dZ








def relu_backward(dA, cache):
    """
    Function:
        Implement derivative of relu activations
    
    Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently
    
    Returns:
        dZ -- Gradient of the cost with respect to Z
    
    Raise:
        shape error if dZ shape not equal Z shape
    """
    
    Z = cache
    
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    
    return dZ
