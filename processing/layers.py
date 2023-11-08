import numpy as np

class Layer:
    def __init__(self, input: int =10, output: int = 10):
        self.weights = np.random.randn(output, input) 
        self.bias = np.random.randn(output)

    def forward(self, X):
        pass
    
    # derivative of activation function
    def _dy(self, X, y):
        pass

    # y = forward(X)
    def Jacobian(self, X, y):
        dy = self._dy(X, y)
        return np.multiply(self.weights, dy[:, np.newaxis])

    # get external gradient (used in backward pass)
    def external_gradient(self, X, y, egrad):
        jacob = self.Jacobian(X, y)
        return jacob.T @ egrad
    
    # get directional derivative (update weights) 
    def _Wderivative(self, X, y, egrad):
        dy = self._dy(X, y)
        M = np.outer(dy.T, X)
        return np.multiply(M, egrad[:, np.newaxis])

    # get directional derivative (update weights) 
    def _bderivative(self, X, y, egrad):
        dy = self._dy(X, y)
        return dy * egrad
    
    def gradient(self, X, y, egrad):
        dw = self._Wderivative(X, y, egrad)
        db = self._bderivative(X, y, egrad)
        return (dw, db)


class SigmoidLayer(Layer):
    def __init__(self, input: int =10, output: int = 10):
        super().__init__(input, output)

    def forward(self, X):
        y = self.weights @ X
        y = y + self.bias  

        condlist = [y < 0, y>= 0]
        choicelist = [np.e ** y / (1 + np.e ** y), 1 / (1 + np.e ** -y)]
        y = np.select(condlist, choicelist)
        return y
    
    def _dy(self, X, y):
        return y * (1 - y)
    

class ReLULayer(Layer): 
    def __init__(self, input: int =10, output: int = 10, alpha:float=1.0):
        super().__init__(input, output)
        self.alpha = alpha

    def forward(self, X):
        y = self.weights @ X.T
        y = y + self.bias
        y = np.maximum(y, 0)
        return y * self.alpha
    
    def _dy(self, X, y):
        return (y >= 0).astype('float64') * self.alpha