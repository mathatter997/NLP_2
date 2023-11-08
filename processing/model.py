from processing.layers import *
from typing import List
import numpy as np

class NeuralNet:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.depth = len(layers)
        self.egrads = [0 for _ in range(len(layers))]
        self.values = [0 for _ in range(len(layers) + 1)]
    
    def forward(self, X):
        self.values[0] = X
        for i, layer in enumerate(self.layers):
            self.values[i + 1] = layer.forward((self.values[i]))

    def backward(self, egrad):
        
        for i in range(self.depth - 1, -1, -1):
            X = self.values[i]
            y = self.values[i + 1]
            self.egrads[i] = egrad
            egrad = self.layers[i].external_gradient(X, y, egrad)

    # loss is computed in batch update using categorical cross-entropy
    # lamb is for l2 regularization
    def batch_update(self, Xs, ys, eta:float =0.01, lamb:float = 0.01):
        wgrad_avg = [0 for _ in range(self.depth)]
        bgrad_avg = [0 for _ in range(self.depth)]
        for i in range(len(Xs)):
            X, y = Xs[i], ys[i]
            self.forward(X)
            pred = self.values[-1]
            if pred == y: val = 0
            elif pred == 0: val = 10
            elif pred == 1: val = -10
            else: val = (y / pred - (1 - y) / (1 - pred))[0] # unpack (1,) np.array
            if val < -10: val = -10
            elif val > 10: val = 10
            dloss = np.array([val], dtype='float64') # re-pack

            self.backward(dloss)
            for k in range(self.depth):
                dw, db = self.layers[k].gradient(self.values[k], self.values[k+1], self.egrads[k])
                wgrad_avg[k] += (dw - lamb * self.layers[k].weights ** 2)
                bgrad_avg[k] += (db - lamb * self.layers[k].bias ** 2)
        
        wgrad_avg = [wgrad_avg[k]/len(Xs) - lamb * self.layers[k].weights ** 2 for k in range(self.depth)]
        bgrad_avg = [bgrad_avg[k]/len(Xs) - lamb * self.layers[k].bias ** 2 for k in range(self.depth)]
        for i in range(self.depth):
            self.layers[i].weights += eta * wgrad_avg[i]
            self.layers[i].bias += eta * bgrad_avg[i]
     
    def batch_predict(self, Xs):
        pred = [0 for _ in range(len(Xs))]
        for i in range(len(Xs)):
            X = Xs[i]
            self.forward(X)
            pred[i] = self.values[-1][0]
        return pred






