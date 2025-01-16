import numpy as np

class OPS:
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(x, axis = 1, keepdims = True)