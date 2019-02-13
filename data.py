#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 00:15:41 2019

@author: diaa
"""

from util.encoding import *

def train_data_loader(path=None):
   X_train = None
   Y_train = None
   X_test = None
   Y_test = None
   n_classes = None
  
   ### Write your logic here ###
   
   
   Y_train = one_hot_encode(Y_train, n_classes)
   Y_test = one_hot_encode(Y_test, n_classes)
   
   return X_train, Y_train, X_test, Y_test

def predict_data_loader(path=None):
    X = None
    
    ### Write your logic here ###
    
    return X
