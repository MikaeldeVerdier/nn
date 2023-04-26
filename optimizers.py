import numpy as np

class SGD:
    def __init__(self, lr):
        self.lr = lr
    
    def __call__(self, model, x, y):
        z = x

        act_func_grads = []
        layer_grads = []
        for layer in model.layers:
            a, z = layer(z)

            act_func_grads.append(layer.act_func.grad(z))
            layer_grads.append(layer.grad(a))

        loss = model.loss(z, y)
        loss_grad = model.loss.grad(z, y)[0]

        grad = loss_grad
        for i, layer in reversed(list(enumerate(model.layers))):
            grad *= act_func_grads[i]

            a_grad = layer_grads[i][0]
            w_grad = layer_grads[i][1]
            b_grad = layer_grads[i][2]
            layer.back_prop(grad * w_grad, grad * b_grad, self.lr)

            grad = np.dot(grad, a_grad)

        return loss
