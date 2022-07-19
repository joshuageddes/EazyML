import numpy as np
from eazyml import *

def init_params(model, n_init):

        n_prev = n_init
        for layer in model.layers:

            layer.weights = np.random.randn(layer.neurons, n_prev)*0.01
            layer.bias = np.zeros((layer.neurons, 1))  
            n_prev = layer.neurons


def forward_propagate(model, x):



    A_prev = x

    for layer in model.layers:

        layer.Z = np.dot(layer.weights, A_prev) + layer.bias

        if layer.activation == "sigmoid":
            layer.A = sigmoid(layer)

        elif layer.activation == "relu":
            layer.A = relu(layer)
        
        elif layer.activation == "tanh":
            layer.A = tanh(layer)
            
        A_prev = layer.A


def calculate_cost(model, y):
    
    examples = y.shape[1]
    A = model.layers[-1].A

    

    loss = -(y*np.log(A) + (1-y)*np.log(1-A))
    cost = np.squeeze(np.sum(loss)/examples)
    model.cost = cost

    print("Cost: " + str(cost))

def backward_propagate(model, x, y):

    

    final_layer = model.layers[-1]
    examples = final_layer.A.shape[1]

    dA = dcost(final_layer, y)

   

    for i in reversed(range(1, len(model.layers))):
        layer = model.layers[i]

        

        dZ = dA * dactivation(layer)
        layer.dW = np.dot(dZ, model.layers[i-1].A.T)/examples
        layer.db = np.sum(dZ, axis=1, keepdims=True)/examples
        dA = np.dot(layer.weights.T, dZ)
    
    #special case for first layer

    dZ = dA * dactivation(model.layers[0])
    model.layers[0].dW = np.dot(dZ, x.T)/examples
    model.layers[0].db = np.sum(dZ, axis=1, keepdims=True)/examples


def update(layer, learning_rate):
    layer.weights = layer.weights - layer.dW*learning_rate
    layer.bias = layer.bias - layer.db*learning_rate