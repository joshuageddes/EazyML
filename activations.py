import numpy as np
from eazyml import *

def sigmoid(layer):
    
    return 1/(1+np.exp(-layer.Z))

def relu(layer):
    
    return layer.Z * (layer.Z > 0)

def tanh(layer):
    return np.tanh(layer.Z)


def dcost(layer, y):
    A = layer.A
    return - (np.divide(y, A) - np.divide(1 - y, 1 - A))


def dsigmoid(layer):
    return layer.A * (1-layer.A)

def drelu(layer):
    return 1 * (layer.A > 0)


def dtanh(layer):
    return 1 - layer.A*layer.A

def dactivation(layer):
    if layer.activation == "relu":
        return drelu(layer)
    elif layer.activation == "sigmoid":
        return dsigmoid(layer)
    elif layer.activation == "tanh":
        return dtanh(layer)