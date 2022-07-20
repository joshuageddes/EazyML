import numpy as np
from eazyml import *

class Model():
    """
    Creates a model object.
    Build, train, and test the model to observe loss and accuracy.
    """

    def __init__(self):
        self.layers = []
        self.cost = None
        self.accuracy = None

    def add_layer(self, neurons, activation):

        """
        Creates and adds a Layer object to the Model.layers list.
        
        :param neurons: The number of neurons in the layer.
        :type neurons: int

        :param activation: The activation of the layer. Currently support sigmoid, relu, and tanh.
        :type activation: str
        """
        self.layers.append(Layer(neurons, activation))
    
    def train(self, x_train, y_train, epochs=50, learning_rate=0.01):
        """
        Trains the model on a labelled dataset.
        
        :param x_train: The training input.
        :type x_train: list of shape (examples, features)

        :param y_train: The training labels.
        :type y_train: list of shape (examples, neurons in last layer)

        :param epochs: The number of epochs to train the model on. Default of 50.
        :type epochs: int

        :param learning_rate: The learning rate of the model. Default of 0.01.
        :type learning_rate: float
        """

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

        """
        Tests the model with an independent labelled testset.
        
        :param x_test: The testing input.
        :type x_test: list of shape (examples, features)

        :param y_test: The testing labels.
        :type y_test: list of shape (examples, neurons in last layer)

        :param threshold: The numerical tolerance for a prediction to be considered equal to a label. Default of 0.5 (reccomended for sigmoid final layer).
        :type threshold: float
        """

        x_test = np.array(x_test).T
        y_test = np.array(y_test).T
        forward_propagate(self, x_test)

        

        layer = self.layers[-1]
        y = layer.A

        

    
        

        accuracy_matrix = (y > (y_test-threshold)) & (y < (y_test+threshold))

        accuracy = (np.count_nonzero(accuracy_matrix) / accuracy_matrix.shape[1])*100
        print("Test accuracy is " + str(accuracy) + "%")
        self.accuracy = accuracy

    def layers(self):
        """
        Returns the list of layers in the model.

        :return: The list of Layer objects.
        :rtype: list
        """
        return self.layers
    
    def cost(self):
        """
        Returns the most recently computed model cost.

        :return: The cost.
        :rtype: float
        """
        return self.cost
    
    def accuracy(self):
        """
        Returns the most recently computed model accuracy.

        :return: The accuracy.
        :rtype: float
        """
        return self.accuracy
        