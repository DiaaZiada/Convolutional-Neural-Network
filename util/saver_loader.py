#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 18:38:44 2019

@author: diaa
"""
import numpy as np

def saver(parameters, path):
    """
    Function:
        Save trained model parameters
    
    Arguments:
        parameters -- parameters of the trained model
        path -- path to save model
    """
    np.save(path,parameters)    

def loader(path):
    """
    Function:
        load trained model parameters
    
    Arguments:
        path -- path of  trained model parameters
    """
    return np.load(path)