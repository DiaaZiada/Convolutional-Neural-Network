#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:35:50 2019

@author: diaa
"""
import numpy as np
from .forward import L_model_forward
from .conv import *
from .pool import *

def predict(X,layers_parameters):
    """
    Function:
        This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
    
    Display:
        predictions for the given dataset X
    """
    
    for p in list(layers_parameters.keys()):
        
        if 'conv' in p:
            A, _ = conv_forward(A, layers_parameters[p]['W'], layers_parameters[p]['b'], layers_parameters[p]['hparameters'])
        elif 'pool' in p:
            A, _ = pool_forward(A, layers_parameters[p]['hparameters'], mode=layers_parameters[p]['mode'])
        elif 'FC' == p:        
            A = A.reshape(-1,A.shape[0])
            Y_, _, _ = L_model_forward(A, layers_parameters[p],keep_prob)
    for y_ in range(0, Y_.shape[1]):
        print("class precitions is: {}".format(np.argmax(y_)))     
        
def accuracy_precitor(X, Y, layers_parameters):
    """
    Function:
        This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
        X -- data set of examples 
        Y -- data set of labels
        parameters -- parameters of the trained model
    
    Returns:
    p -- accuracy of the model
    """
    m = X.shape[0]
    prob = np.zeros((1,m), dtype = np.int)
    
    A = X
    for p in list(layers_parameters.keys()):
        
        if 'conv' in p:
            A, _ = conv_forward(A, layers_parameters[p]['W'], layers_parameters[p]['b'], layers_parameters[p]['hparameters'])
        elif 'pool' in p:
            A, _ = pool_forward(A, layers_parameters[p]['hparameters'], mode=layers_parameters[p]['mode'])
        elif 'FC' == p:        
            A = A.reshape(-1,A.shape[0])
            Y_, _, _ = L_model_forward(A, layers_parameters[p],None)    
    for i in range(0, Y_.shape[0]):
        if Y_[0,i] > 0.5:
            prob[0,i] = 1
        else:
            prob[0,i] = 0
    print("Accuracy: "  + str(np.mean((prob[0,:] == Y[0,:]))))
    return prob


