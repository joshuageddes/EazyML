from eazyml import *

model = Model()
model.add_layer(1, "sigmoid")
model.train([[1,2],[2,1]], [[0],[1]], 1, 0.00001)
model.test([[1,2],[2,1]], [[0],[1]], 0.5)






