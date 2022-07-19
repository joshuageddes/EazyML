import numpy as np
from eazyml import *

class Model():

    def __init__(self):
        self.layers = []
        self.cost = None
        self.accuracy = None

    def add_layer(self, neurons, activation):
        self.layers.append(Layer(neurons, activation))
    
    def train(self, x_train, y_train, epochs=50, learning_rate=0.01):

        x_train = np.array(x_train).T
        y_train = np.array(y_train).T
        

        n_init = x_train.shape[0]
        init_params(self, n_init)

     
        for i in range(epochs):
            print("Epoch: " + str(i+1))
            forward_propagate(self, x_train)
            calculate_cost(self, y_train)
            backward_propagate(self, x_train, y_train)


            for layer in self.layers:
                update(layer, learning_rate)
    
    def test(self, x_test, y_test, threshold = 0.5):
        x_test = np.array(x_test).T
        y_test = np.array(y_test).T
        forward_propagate(self, x_test)

        

        layer = self.layers[-1]
        y = layer.A

        

    
        

        accuracy_matrix = (y > (y_test-threshold)) & (y < (y_test+threshold))

        accuracy = np.count_nonzero(accuracy_matrix) / accuracy_matrix.shape[1]
        print("Test accuracy is " + str(accuracy*100) + "%")
        self.accuracy = accuracy

    def layers(self):
        return self.layers
    
    def cost(self):
        return self.cost
    
    def accuracy(self):
        return self.accuracy