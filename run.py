#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 13:15:07 2019

@author: diaa
"""

import argparse

from data import *

from util.initializer import *
from util.forward import *
from util.cost import *
from util.backward import *
from util.conv import *
from util.pool import *
from util.encoding import *
from util.updater import *
from util.saver_loader import *
from util.prediction import *

def L_layer_model(X, Y, layers, initializer=0, n_epochs = 3000, learning_rate = 0.001, lambd=None, keep_prob=None,beta1=None, beta2=None, epsilon=None):

    layers_parameters, shape = cnn_initialize_parameters(layers, initializer)               
    ll = layers_parameters
    v = None
    s = None
    if beta1:
        v = initialize_beta(layers_parameters['FC'])
    if beta2:    
        s = initialize_beta(layers_parameters['FC'])


    for e in range(1,n_epochs+1):

        grads = {}   
        caches = []
    
        A = X
        A = X
        for p in list(layers_parameters.keys()):
            
            if 'conv' in p:
                A, cache = conv_forward(A, layers_parameters[p]['W'], layers_parameters[p]['b'], layers_parameters[p]['hparameters'])
                caches.append(cache)
            elif 'pool' in p:
                A, cache = pool_forward(A, layers_parameters[p]['hparameters'], mode=layers_parameters[p]['mode'])
                caches.append(cache)
            elif 'FC' == p:        
                A = A.reshape(-1,A.shape[0])
                AL, cache, Ds = L_model_forward(A, layers_parameters[p],keep_prob)
                caches.append(cache)
      

        for p in reversed(list(layers_parameters.keys())):
            
            if 'FC' == p:
                cache = caches.pop()
                grad, dA_prev = L_model_backward(AL, Y, cache,lambd,keep_prob,Ds)
                grads[p] = grad
                dA_prev = dA_prev.reshape(-1,*shape)
            
            elif 'pool' in p:
                cache = caches.pop()
                dA_prev = pool_backward(dA_prev, cache, mode = layers_parameters[p]['mode'])
            
            elif 'conv' in p:
                cache = caches.pop()
                dA_prev, dW, db = conv_backward(dA_prev, cache)
                grads[p] ={'dW':dW,'db':db}
        
        layers_parameters, v, s = update(layers_parameters=layers_parameters, grads=grads, learning_rate=learning_rate,v=v,beta1=beta1,s=s,beta2=beta2,t=e,epsilon=epsilon)
        cost = compute_cost(AL, Y,parameters=layers_parameters,lambd=lambd)
        if e % (n_epochs/10) == 0:
            print ("Cost after iteration {}: {}" .format(e, cost))
   
    return layers_parameters


def manager():
    
    
        
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--train', type=bool,default=False)
    parser.add_argument('--predict', type=bool,default=False)
    
    parser.add_argument('--image_shape', type=int, nargs='+')
    parser.add_argument('--conv', type=str)
    parser.add_argument('--fc', type=int, nargs='+')
    
    
    parser.add_argument('--initializer', type=int)
    
    parser.add_argument('--n_epochs', type=int)
   
    
    
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lambd', type=float,default=None)
    parser.add_argument('--keep_prob', type=float,default=None)
    parser.add_argument('--beta1', type=float,default=None)
    parser.add_argument('--beta2', type=float,default=None)
    parser.add_argument('--epsilon', type=float,default=None)
    
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--saving_path', type=str)
    parser.add_argument('--loading_path', type=str)

    parser.add_argument('--print_accuracy', type=bool, default=True)

    


    args = parser.parse_args()
    train = args.train
    predict = args.predict
    image_shape = args.image_shape
    conv = args.conv
    fc = args.fc
    
    initializer = args.initializer
    
    n_epochs = args.n_epochs
    
    learning_rate = args.learning_rate
    lambd = args.lambd
    keep_prob = args.keep_prob
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.epsilon
    
    
    if epsilon is None:
        epsilon = 1e-8
        
    data_path = args.data_path    
    saving_path = args.saving_path
    loading_path = args.loading_path
    
    print_accuracy = args.print_accuracy
    
    
    layers = {}
    layers ['image_shape'] = image_shape
    for a in conv.split('=>'):
        s = a.strip().split(' ')
        if 'conv'in s[0]:
            layers[s[0]]= {'W':(int(s[1]),int(s[2])),'hparameters':(int(s[3]),int(s[4]))}
        elif 'pool' in s[0]:
            layers[s[0]] = {'hparameters':(int(s[1]),int(s[2])),'mode':s[3]} 
            
    layers['FC']=fc

    
    if train:
        X_train,Y_train,X_test,Y_test = train_data_loader(data_path)

        parameters = L_layer_model(X=X_test, Y=Y_test, layers=layers, initializer=initializer, n_epochs=n_epochs, learning_rate=learning_rate, lambd=lambd, keep_prob=keep_prob, beta1=beta1, beta2=beta2,epsilon=epsilon)
        
        if print_accuracy:  
            accuracy_precitor(X_test, Y_test, parameters)

        if saving_path:
            saver(parameters, saving_path)
            
    elif predict:
        parameters = loader(loading_path) 
        X = predict_data_loader(data_path)
        predict(X,parameters)

manager()
#!python run.py --train True --image_shape 64 64 3 --conv "conv1 2 8 2 0 => pool1 2 2 max => conv2 2 16 2 0 => pool2 2 2 max " --fc 500 100 50 6 --n_epochs 5 --learning_rate 0.3