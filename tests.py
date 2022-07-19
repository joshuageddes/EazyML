import unittest
from eazyml import *

class ModelTests(unittest.TestCase):

    def test_sigmoid(self):
        layer = Layer(1, "sigmoid")
        layer.Z = np.array([0,2])
        result = sigmoid(layer)
        self.assertTrue((result==np.array([0.5, 0.8807970779778823])).all())
    
    def test_single_layer(self):
        model = Model()
        model.add_layer(1, "sigmoid")
        x = np.array([[1,-2,-1],[3,0.5,-3.2]])
        model.layers[0].weights = np.array([[1,2]])
        model.layers[0].bias = 1.5
        forward_propagate(model, x)
        
        self.assertTrue((model.layers[0].A==np.array([[0.9997965730219448, 0.6224593312018546, 0.002731960763011059]])).all())

        y = np.array([[1,1,0]])
        calculate_cost(model, y)
        self.assertEqual(model.cost,0.15900537707692405)

        backward_propagate(model, x, y)

        
        
        self.assertTrue((model.layers[0].dW == np.array([[0.25071531661840823,-0.06604096325829124]])).all())
        self.assertTrue((model.layers[0].db == np.array([[-0.12500404500439652]])).all())

        for layer in model.layers:
            update(layer, 0.009)
        
        self.assertTrue((model.layers[0].weights == np.array([[0.9977435621504344, 2.000594368669325]])).all())
        self.assertTrue((model.layers[0].bias==np.array([[1.5011250364050395]])).all())
    
    def test_multi_layer(self):
        model = Model()
        model.add_layer(4, "tanh")
        model.add_layer(1, "sigmoid")
        model.layers[0].weights = np.array([[-0.00416758,-0.00056267],[-0.02136196,0.01640271],[-0.01793436,-0.00841747],[0.00502881,-0.01245288]])
        model.layers[0].bias = np.array([[1.74481176], [-0.7612069 ], [0.3190391 ], [-0.24937038]])
        model.layers[1].weights = np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]])
        model.layers[1].bias = np.array([[-1.3]])
        x = np.array([[1.62434536,-0.61175641,-0.52817175],[-1.07296862,0.86540763,-2.3015387]])
        y = np.array([[1,0,0]])

        forward_propagate(model, x)
        self.assertTrue((model.layers[1].A == np.array([[0.212926557568532, 0.21274673185662651, 0.21295975569317957]])).all())

        calculate_cost(model, y)
        self.assertEqual(model.cost, 0.6751630453563474)

        y = np.array([[1,0,1]])
        model.layers[0].A = np.array([[-0.00616578,  0.0020626,   0.00349619],
        [-0.05225116,  0.02725659, -0.02646251],
        [-0.02009721,  0.0036869,   0.02883756],
        [ 0.02152675, -0.01385234,  0.02599885]])
        model.layers[1].A = np.array([[0.5002307,0.49985831,0.50023963]])
        backward_propagate(model, x, y)

        self.assertTrue((model.layers[0].db == np.array([[0.0017620133365788525],[0.0015099480049211977],[-0.0009173633897839889],[-0.0038142178737840815]])).all())
        

if __name__ == '__main__':
    unittest.main()


















#should be [[ 0.00078841  0.01765429 -0.00084166 -0.01022527]], db = [[-0.16655712]]

#should be [[ 0.00301023 -0.00747267][ 0.00257968 -0.00641288] [-0.00156892  0.003893  ][-0.00652037  0.01618243]]
#db should be[[ 0.00176201] [ 0.00150995] [-0.00091736] [-0.00381422]]




