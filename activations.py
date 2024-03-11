### Activation Functions

import numpy as np
from scipy.special import softmax

class Sigmoid:
    def forward(self, z):
        self.input = z
        self.output = 1/( 1 + np.exp( self.input*(-1) ) )
        return self.output

    def backward(self):
        der = self.output*(1-self.output)
        return der
        #backward pass of Sigmoid (the structure is the same as the relu activation, just the formulas are different)
        #USE that grad_sigmoid = sigmoid * (1-sigmoid)
    
class Softmax:
    def forward(self, z):
        self.input = z
        self.output = softmax(z, axis = 0) #softmax for each column
        return self.output

    def backward(self, d_output):
        p = self.output
        
        pp = np.einsum('ji, ki -> jki', p, p)
        
        diag = np.einsum('ji, kj -> jki',p, np.eye(p.shape[0]))
        
        return np.einsum('ij, jki -> ik', d_output, diag-pp)
    
class relu:
    """ReLu activation function"""
    def forward(self, z):
        self.input = z
        return np.maximum(0, self.input)
    
    def backward(self):
        derivative = (self.input > 0)
        return derivative
    
class identity:
    def forward(self, z):
        self.input = z
        return z
    
    def backward(self):
        return np.ones_like(self.input)
