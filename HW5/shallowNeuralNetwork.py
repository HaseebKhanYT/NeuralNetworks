import numpy as np


class Activation:
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(z):
        s = Activation.sigmoid(z)
        return s * (1.0 - s)


class Neuron:
    def __init__(self, n_inputs, activation='sigmoid', rng=None):
        scale = 1.0 / np.sqrt(n_inputs)
        if rng is None:
            self.w = np.random.randn(n_inputs) * scale
        else:
            self.w = rng.normal(0.0, scale, size=(n_inputs,))
        self.b = 0.0
        self.act_name = activation

        # caches for backprop
        self.last_x = None
        self.last_z = None
        self.last_a = None

    def _act(self, z):
        if self.act_name == 'sigmoid':
            return Activation.sigmoid(z)
        return z  # linear

    def _act_deriv(self, z):
        if self.act_name == 'sigmoid':
            return Activation.sigmoid_deriv(z)
        return 1.0  # linear

    def forward(self, x):
        self.last_x = x
        self.last_z = float(np.dot(self.w, x) + self.b)
        self.last_a = self._act(self.last_z)
        return self.last_a

    def backward(self, dL_da):
        d_act = self._act_deriv(self.last_z)
        dL_dz = dL_da * d_act
        grad_w = dL_dz * self.last_x
        grad_b = dL_dz
        dL_dx = self.w * dL_dz
        return dL_dx, grad_w, grad_b


class Layer:
    def __init__(self, n_inputs, n_neurons, activation='sigmoid', rng=None):
        self.neurons = [Neuron(n_inputs, activation, rng)
                        for _ in range(n_neurons)]
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.last_input = x
        out = np.array([n.forward(x) for n in self.neurons])
        self.last_output = out
        return out

    def backward(self, dL_dA_vec):
        n_inputs = self.neurons[0].w.shape[0]
        dL_dx_total = np.zeros(n_inputs)
        grads_w = []
        grads_b = []
        for j, neuron in enumerate(self.neurons):
            dL_dx, grad_w, grad_b = neuron.backward(dL_dA_vec[j])
            dL_dx_total += dL_dx
            grads_w.append(grad_w)
            grads_b.append(grad_b)
        return np.array(dL_dx_total), np.array(grads_w), np.array(grads_b)


class Parameters:
    def __init__(self, lr=0.1, epochs=5000, hidden_size=4, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.seed = seed


class Model:
    def __init__(self, n_features, params: Parameters):
        rng = np.random.default_rng(params.seed)
        self.hidden = Layer(n_features, params.hidden_size,
                            activation='sigmoid', rng=rng)
        self.output = Layer(params.hidden_size, 1,
                            activation='sigmoid', rng=rng)

    def forward(self, x):
        h = self.hidden.forward(x)
        y_hat = self.output.forward(h)[0]
        return h, y_hat

    def predict(self, X):
        return np.array([self.forward(x)[1] for x in X])


class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
        return 0.5 * (y_pred - y_true) ** 2

    @staticmethod
    def mse_deriv(y_true, y_pred):
        return (y_pred - y_true)


class ForwardProp:
    @staticmethod
    def run(model: Model, x):
        return model.forward(x)


class BackProp:
    @staticmethod
    def run(model: Model, y_true, caches):
        h, y_hat = caches
        # Output layer delta (using MSE): dL/da_out
        dL_da_out = LossFunction.mse_deriv(y_true, y_hat)
        # Backprop through output layer (vector of one)
        dL_dh, grads_w_out, grads_b_out = model.output.backward(
            np.array([dL_da_out]))
        # Backprop through hidden layer
        dL_dx, grads_w_hidden, grads_b_hidden = model.hidden.backward(dL_dh)
        grads = {
            'hidden_w': grads_w_hidden,
            'hidden_b': grads_b_hidden,
            'out_w': grads_w_out,
            'out_b': grads_b_out
        }
        return grads


class GradDescent:
    @staticmethod
    def step(model: Model, grads, lr):
        # Update output layer
        for j, neuron in enumerate(model.output.neurons):
            neuron.w -= lr * grads['out_w'][j]
            neuron.b -= lr * grads['out_b'][j]
        # Update hidden layer
        for j, neuron in enumerate(model.hidden.neurons):
            neuron.w -= lr * grads['hidden_w'][j]
            neuron.b -= lr * grads['hidden_b'][j]


class Training:
    @staticmethod
    def fit(model: Model, X, y, params: Parameters, verbose=False):
        losses = []
        for epoch in range(params.epochs):
            total_loss = 0.0
            # Simple SGD over samples
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]
                h, y_hat = ForwardProp.run(model, x_i)
                total_loss += LossFunction.mse(y_i, y_hat)
                grads = BackProp.run(model, y_i, (h, y_hat))
                GradDescent.step(model, grads, params.lr)
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % max(1, params.epochs // 10) == 0:
                print(f"Epoch {epoch+1}, loss={avg_loss:.4f}")
        return losses


if __name__ == "__main__":
    # Tiny XOR demo
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])

    params = Parameters(lr=0.5, epochs=5000, hidden_size=4, seed=0)
    model = Model(n_features=2, params=params)
    losses = Training.fit(model, X, y, params, verbose=True)

    preds = model.predict(X)
    print("Predictions:", np.round(preds, 3))
