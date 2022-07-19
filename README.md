# EazyML
A python library for building, training, and testing simple machine learning models.

### Installation
```
pip install EazyML
```

### Get started
Build and test a simple logistic regression model

```Python
from EazyML import *

# Instantiate a Model object
model = Model()

#Add a sigmoid layer to the model with one neuron
model.add_layer(1, "sigmoid")

#Train the model with labeled training data
model.train(x_train, y_train, epochs=1, learning_rate=0.005)

#Test the model with labeled test data
model.test(x_train, y_train, threshold=0.5)
```