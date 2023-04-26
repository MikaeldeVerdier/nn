class MSE:
    def __call__(self, y_pred, y_true):
        return 1/len(y_true) * sum((y_pred - y_true) ** 2)
    
    def grad(self, y_pred, y_true):
        d_y_pred = 1/len(y_true) * 2 * (y_pred - y_true)
        d_y_true = 1/len(y_true) * 2 * (y_pred - y_true)

        return (d_y_pred, d_y_true)


class MAE:
    def __call__(self, y_pred, y_true):
        return 1/len(y_true) * sum(abs(y_pred - y_true))
    
    def grad(self, y_pred, y_true):
        d_y_pred = 1/len(y_true) * abs(y_pred - y_true)
        d_y_true = 1/len(y_true) * abs(y_pred - y_true)

        return (d_y_pred, d_y_true)
