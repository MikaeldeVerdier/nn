import numpy as np
import random

from model import Model

x = [np.array([1, 2, 3])]
y = [np.array([3, 2, 1])]

model = Model(len(x[0]), len(y[0]))

pack = list(zip(*(x, y)))

history = []
for _ in range(10000):
    sample = random.sample(pack, 1)[0]
    history.append(model.fit(*sample))

model.plot_loss(history)

prediction = model(sample[0])
