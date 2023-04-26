import numpy as np

class Dense:
    def __init__(self, input_shape, amount_neurs, act_func, use_bias=True):
        self.input_shape = input_shape
        self.amount_neurs = amount_neurs
        self.act_func = act_func
        self.use_bias = use_bias

        self.weights = np.random.randn(input_shape, amount_neurs)
        self.bias = 0 if use_bias else None

    def __call__(self, inp):
        a = np.dot(inp, self.weights) + self.bias
        z = self.act_func(a)

        return a, z
    
    def grad(self, inp):
        d_inp = self.weights.T
        d_weights = inp
        d_bias = 1 or None

        return (d_inp, d_weights, d_bias)
    
    def back_prop(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        if self.use_bias:
            self.bias -= lr * grad_b
