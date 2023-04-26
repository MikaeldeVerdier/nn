from matplotlib import pyplot as plt

import config
from activations import ReLU, Linear
from layers import Dense
from losses import MSE
from optimizers import SGD

class Model:
    def __init__(self, input_shape, output_shape):
        self.layers = []

        for amount_neurons in config.HIDDEN_DENSES:
            self.layers.append(Dense(input_shape, amount_neurons, ReLU()))
            input_shape = amount_neurons
        self.layers.append(Dense(input_shape, output_shape, Linear()))

        self.loss = MSE()
        self.optimizer = SGD(config.LEARNING_RATE)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)[1]

        return x

    def fit(self, x, y):
        return self.optimizer(self, x, y)

    def plot_loss(self, history):
        plt.plot(history)
        plt.savefig("loss", dpi=200)
