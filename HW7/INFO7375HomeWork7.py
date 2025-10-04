
import time
import random
import numpy as np
import matplotlib.pyplot as plt


class Activation:
    @staticmethod
    def linear(Z):
        A = Z
        return A

    @staticmethod
    def relu(Z):
        A = np.maximum(0, Z)
        return A

    @staticmethod
    def sigmoid(Z):
        A = 1/(1+np.exp(-Z))
        return A

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(z):
        expZ = np.exp(z)
        return expZ/(np.sum(expZ, 0))

    @staticmethod
    def derivative_relu(Z):
        return np.array(Z > 0, dtype='float')

    @staticmethod
    def derivative_tanh(x):
        return (1 - np.power(x, 2))


class Neuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = None
        self.bias = None
        self.output = None
        self.input = None


class Layer:
    def __init__(self, layer_dims, layer_index):
        self.layer_dims = layer_dims
        self.layer_index = layer_index
        self.neurons = [Neuron(layer_dims[layer_index-1])
                        for _ in range(layer_dims[layer_index])]


class Parameters:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        S = len(self.layer_dims)
        for s in range(1, S):
            self.parameters['W' + str(s)] = np.random.randn(self.layer_dims[s],
                                                            self.layer_dims[s-1]) / np.sqrt(self.layer_dims[s-1])
            self.parameters['b' + str(s)] = np.zeros((self.layer_dims[s], 1))

    def get_parameters(self):
        return self.parameters


class LossFunction:
    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        if Y.shape[0] == 1:
            cost = (1./m) * (-np.dot(Y, np.log(AL).T) -
                             np.dot(1-Y, np.log(1-AL).T))
        else:
            cost = -(1./m) * np.sum(Y * np.log(AL))
        cost = np.squeeze(cost)
        return cost


class ForwardProp:
    def __init__(self, parameters, activation):
        self.parameters = parameters
        self.activation = activation

    def forward_propagation(self, X):
        forward_cache = {}
        L = len(self.parameters) // 2
        forward_cache['A0'] = X

        # Hidden Layers
        for l in range(1, L):
            forward_cache['Z' + str(l)] = self.parameters['W' + str(l)].dot(
                forward_cache['A' + str(l-1)]) + self.parameters['b' + str(l)]

            if self.activation == 'tanh':
                forward_cache['A' +
                              str(l)] = Activation.tanh(forward_cache['Z' + str(l)])
            elif self.activation == 'relu':
                forward_cache['A' +
                              str(l)] = Activation.relu(forward_cache['Z' + str(l)])
            elif self.activation == 'sigmoid':
                forward_cache['A' +
                              str(l)] = Activation.sigmoid(forward_cache['Z' + str(l)])
            else:
                raise ValueError(f"Unknown activation: {self.activation}")

        # Output Layer
        forward_cache['Z' + str(L)] = self.parameters['W' + str(L)].dot(
            forward_cache['A' + str(L-1)]) + self.parameters['b' + str(L)]

        if forward_cache['Z' + str(L)].shape[0] == 1:
            forward_cache['A' +
                          str(L)] = Activation.sigmoid(forward_cache['Z' + str(L)])
        else:
            forward_cache['A' +
                          str(L)] = Activation.softmax(forward_cache['Z' + str(L)])

        return forward_cache['A' + str(L)], forward_cache


class BackProp:
    def __init__(self, parameters, activation):
        self.parameters = parameters
        self.activation = activation

    def backward_propagation(self, AL, Y, forward_cache):
        grads = {}
        L = len(self.parameters)//2
        m = AL.shape[1]

        grads["dZ" + str(L)] = AL - Y
        grads["dW" + str(L)] = 1./m * np.dot(grads["dZ" + str(L)],
                                             forward_cache['A' + str(L-1)].T)
        grads["db" + str(L)] = 1./m * \
            np.sum(grads["dZ" + str(L)], axis=1, keepdims=True)

        for l in reversed(range(1, L)):
            if self.activation == 'tanh':
                grads["dZ" + str(l)] = np.dot(self.parameters['W' + str(l+1)].T,
                                              grads["dZ" + str(l+1)])*Activation.derivative_tanh(forward_cache['A' + str(l)])
            else:
                grads["dZ" + str(l)] = np.dot(self.parameters['W' + str(l+1)].T,
                                              grads["dZ" + str(l+1)])*Activation.derivative_relu(forward_cache['A' + str(l)])

            grads["dW" + str(l)] = 1./m * np.dot(grads["dZ" + str(l)],
                                                 forward_cache['A' + str(l-1)].T)
            grads["db" + str(l)] = 1./m * \
                np.sum(grads["dZ" + str(l)], axis=1, keepdims=True)

        return grads


class GradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
                self.learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
                self.learning_rate * grads["db" + str(l+1)]
        return parameters


class Model:
    def __init__(self, layer_dims, learning_rate=0.03, activation='relu'):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.activation = activation
        self.parameters = Parameters(layer_dims)
        self.forward_prop = ForwardProp(
            self.parameters.get_parameters(), activation)
        self.back_prop = BackProp(self.parameters.get_parameters(), activation)
        self.grad_descent = GradDescent(learning_rate)

    def predict(self, X, y):
        m = X.shape[1]
        y_pred, caches = self.forward_prop.forward_propagation(X)

        if y.shape[0] == 1:
            y_pred = np.array(y_pred > 0.5, dtype='float')
        else:
            y = np.argmax(y, 0)
            y_pred = np.argmax(y_pred, 0)

        return np.round(np.sum((y_pred == y)/m), 2)

    def train(self, X, Y, num_iterations=3000, X_train=None, Y_train=None, X_test=None, Y_test=None):
        np.random.seed(1)
        costs = []
        parameters = self.parameters.get_parameters()

        for i in range(0, num_iterations):
            AL, forward_cache = self.forward_prop.forward_propagation(X)
            cost = LossFunction.compute_cost(AL, Y)
            grads = self.back_prop.backward_propagation(AL, Y, forward_cache)
            parameters = self.grad_descent.update_parameters(parameters, grads)

            if i % (num_iterations/10) == 0:
                train_acc = self.predict(
                    X_train, Y_train) if X_train is not None else 0
                test_acc = self.predict(
                    X_test, Y_test) if X_test is not None else 0
                print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:{}".format(
                    i, np.round(cost, 2), train_acc, test_acc))

            if i % 10 == 0:
                print("==", end='')

        return parameters


class Training:
    def __init__(self, model):
        self.model = model

    def train_model(self, X, Y, num_iterations=3000, X_train=None, Y_train=None, X_test=None, Y_test=None):
        return self.model.train(X, Y, num_iterations, X_train, Y_train, X_test, Y_test)


# Dataset
X_train = np.loadtxt('dataset/cat_train_x.csv', delimiter=',')/255.0
Y_train = np.loadtxt('dataset/cat_train_y.csv',
                     delimiter=',').reshape(1, X_train.shape[1])
X_test = np.loadtxt('dataset/cat_test_x.csv', delimiter=',')/255.0
Y_test = np.loadtxt('dataset/cat_test_y.csv',
                    delimiter=',').reshape(1, X_test.shape[1])


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

index = random.randrange(0, X_train.shape[1])
plt.imshow(X_train[:, index].reshape(64, 64, 3))
plt.show()


# Initialize Parameters
layer_dims = [X_train.shape[0], 100, 200, Y_train.shape[0]]
params = Parameters(layer_dims).get_parameters()

for s in range(1, len(layer_dims)):
    print("Shape of W" + str(s) + ":", params['W' + str(s)].shape)
    print("Shape of B" + str(s) + ":", params['b' + str(s)].shape, "\n")

# Test forward propagation
forward_prop = ForwardProp(params, 'relu')
aL, forw_cache = forward_prop.forward_propagation(X_train)

for l in range(len(params)//2 + 1):
    print("Shape of A" + str(l) + " :", forw_cache['A' + str(l)].shape)

# Test backward propagation
back_prop = BackProp(params, 'relu')
grads = back_prop.backward_propagation(
    forw_cache["A" + str(3)], Y_train, forw_cache)

for l in reversed(range(1, len(grads)//3 + 1)):
    print("Shape of dZ" + str(l) + " :", grads['dZ' + str(l)].shape)
    print("Shape of dW" + str(l) + " :", grads['dW' + str(l)].shape)
    print("Shape of dB" + str(l) + " :", grads['db' + str(l)].shape, "\n")

# Train model
layers_dims = [X_train.shape[0], 20, 7, 5, Y_train.shape[0]]  # 4-layer model
lr = 0.0075
iters = 2500

model = Model(layers_dims, learning_rate=lr, activation='relu')
training = Training(model)
parameters = training.train_model(X_train, Y_train, num_iterations=iters,
                                  X_train=X_train, Y_train=Y_train,
                                  X_test=X_test, Y_test=Y_test)
