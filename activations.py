import numpy as np

class ReLU:
    def __call__(self, inp):
        return np.maximum(0, inp)

    def grad(self, inp):
        d_inp = np.zeros(len(inp))

        # deriv[inp < 0] = 0
        d_inp[inp > 0] = 1

        return d_inp
    

class Sigmoid:
    def __call__(self, inp):
        return 1 / (1 + np.exp(-inp))

    def grad(self, inp):
        sig = self(inp)
        d_inp = sig * (1 - sig)

        return d_inp


class Linear:
    def __call__(self, inp):
        return inp
    
    def grad(self, _):
        return 1
