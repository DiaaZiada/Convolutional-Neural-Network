#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:33:22 2019

@author: diaa
"""
import numpy as np

def pool_forward(A_prev, hparameters, mode="max"):
    """
    Function:
     Implements the forward pass of the pooling layer
    
    Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              

    
    for i in range(m):
        
        for h in range(n_H):
            
            for w in range(n_W):
            
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    
    cache = (A_prev, hparameters)
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache


def create_mask_from_window(x):
    """
    Function:
        Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
        x -- Array of shape (f, f)
        
    Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = x == np.max(x)
    
    return mask

def distribute_value(dz, shape):    
    """
    Function:
        Distributes the input value in the matrix of dimension shape
    
    Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_H, n_W) = shape
    
    average = dz / (n_H * n_W)
    
    a = np.ones(shape) * average
    
    return a


def pool_backward(dA, cache, mode = "max"):
    """
    Function:
        Implement the backward propagation for a convolution function
    
    Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
                   dW -- gradient of the cost with respect to the weights of the conv layer (W)
                   numpy array of shape (f, f, n_C_prev, n_C)
                   db -- gradient of the cost with respect to the biases of the conv layer (b)
                   numpy array of shape (1, 1, 1, n_C)
    """
    
    
    (A_prev, hparameters) = cache
    
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                      
       
        a_prev = A_prev[i]
        
        for h in range(n_H):                   
        
            for w in range(n_W):               
            
                for c in range(n_C):           

                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    if mode == "max":

                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":

                        da = dA[i, h, w, c]

                        shape = (f, f)

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                        
    
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev