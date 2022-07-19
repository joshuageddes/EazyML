class Layer():

    def __init__(self, neurons, activation):

        self.neurons = neurons
        self.activation = activation
        self.weights = None
        self.bias = None
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None
    
    def neurons(self):
        return self.neurons
    
    def activation(self):
        return self.activation
    
    def weights(self):
        return self.weights

    def bias(self):
        return self.bias

    def Z(self):
        return self.Z
    
    def A(self):
        return self.A
    
    def dW(self):
        return self.dW
    
    def db(self):
        return self.db