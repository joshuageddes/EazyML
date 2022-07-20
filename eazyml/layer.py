class Layer():
    """
    Creates a Layer object.

    :param neurons: The number of neurons in the layer.
    :type neurons: int

    :param activation: The activation of the layer. Currently support sigmoid, relu, and tanh.
    :type activation: str
    """

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
        """
        Returns the number of neurons in the layer.

        :return: The number of neurons.
        :rtype: int
        """
        return self.neurons
    
    def activation(self):
        """
        Returns the activation of the layer.

        :return: The activation type.
        :rtype: str
        """
        return self.activation
    
    def weights(self):
        """
        Returns the current weight values of the layer.

        :return: The weight matrix.
        :rtype: NumPy array of shape (neurons in current layer, neurons in previous layer)
        """
        return self.weights

    def bias(self):
        """
        Returns the current bias values of the layer.

        :return: The bias matrix.
        :rtype: NumPy array of shape (neurons in current layer, 1)
        """
        return self.bias

    def Z(self):
        return self.Z
    
    def A(self):
        """
        Returns the most recently computed activation of the layer.

        :return: The activation matrix.
        :rtype: NumPy array of shape (neurons in current layer, examples)
        """
        return self.A
    
    def dW(self):
        return self.dW
    
    def db(self):
        return self.db