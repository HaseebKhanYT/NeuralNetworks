
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


class Dropout:
    @staticmethod
    def apply_dropout(A, keep_prob, training=True):
        """Apply dropout to activations"""
        if training and keep_prob < 1:
            mask = (np.random.rand(*A.shape) < keep_prob).astype(np.float32)
            A = A * mask / keep_prob
            return A, mask
        return A, None

    @staticmethod
    def backward_dropout(dA, mask, keep_prob):
        """Apply dropout mask during backpropagation"""
        if mask is not None:
            dA = dA * mask / keep_prob
        return dA


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
    def __init__(self, parameters, activation, keep_probs=None):
        self.parameters = parameters
        self.activation = activation
        # keep_probs is a list of keep probabilities for each layer
        # e.g., [1, 0.8, 0.7, 1] means no dropout for input and output layers
        self.keep_probs = keep_probs

    def forward_propagation(self, X, training=True):
        forward_cache = {}
        dropout_cache = {}  # Store dropout masks
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

            # Apply dropout to hidden layer
            if self.keep_probs and l < len(self.keep_probs):
                forward_cache['A' + str(l)], dropout_cache['D' + str(l)] = \
                    Dropout.apply_dropout(forward_cache['A' + str(l)],
                                          self.keep_probs[l],
                                          training)

        # Output Layer (no dropout)
        forward_cache['Z' + str(L)] = self.parameters['W' + str(L)].dot(
            forward_cache['A' + str(L-1)]) + self.parameters['b' + str(L)]

        if forward_cache['Z' + str(L)].shape[0] == 1:
            forward_cache['A' +
                          str(L)] = Activation.sigmoid(forward_cache['Z' + str(L)])
        else:
            forward_cache['A' +
                          str(L)] = Activation.softmax(forward_cache['Z' + str(L)])

        # Add dropout cache to forward cache
        forward_cache.update(dropout_cache)

        return forward_cache['A' + str(L)], forward_cache


class BackProp:
    def __init__(self, parameters, activation, keep_probs=None):
        self.parameters = parameters
        self.activation = activation
        self.keep_probs = keep_probs

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
            # First compute dA
            dA = np.dot(
                self.parameters['W' + str(l+1)].T, grads["dZ" + str(l+1)])

            # Apply dropout if it was used in forward pass
            if self.keep_probs and l < len(self.keep_probs) and 'D' + str(l) in forward_cache:
                dA = Dropout.backward_dropout(
                    dA, forward_cache['D' + str(l)], self.keep_probs[l])

            # Then compute dZ through activation derivative
            if self.activation == 'tanh':
                grads["dZ" + str(l)] = dA * \
                    Activation.derivative_tanh(forward_cache['A' + str(l)])
            elif self.activation == 'sigmoid':
                grads["dZ" + str(l)] = dA * (forward_cache['A' + str(l)]
                                             * (1 - forward_cache['A' + str(l)]))
            else:  # relu
                grads["dZ" + str(l)] = dA * \
                    Activation.derivative_relu(forward_cache['A' + str(l)])

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

    def __init__(self, layer_dims, learning_rate=0.03, activation='relu', keep_probs=None, batch_size=32):
        self.batch_size = batch_size
        self.input_mean = None
        self.input_std = None
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.activation = activation
        self.keep_probs = keep_probs
        self.parameters = Parameters(layer_dims)
        self.forward_prop = ForwardProp(
            self.parameters.get_parameters(), activation, keep_probs)
        self.back_prop = BackProp(
            self.parameters.get_parameters(), activation, keep_probs)
        self.grad_descent = GradDescent(learning_rate)

    def create_mini_batches(self, X, Y):
        """Create mini-batches from X and Y"""
        m = X.shape[1]
        mini_batches = []

        # Shuffle the data
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        # Calculate number of complete mini-batches
        num_complete_minibatches = m // self.batch_size

        # Create complete mini-batches
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k *
                                      self.batch_size: (k+1) * self.batch_size]
            mini_batch_Y = shuffled_Y[:, k *
                                      self.batch_size: (k+1) * self.batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handle the last mini-batch if there are remaining samples
        if m % self.batch_size != 0:
            mini_batch_X = shuffled_X[:,
                                      num_complete_minibatches * self.batch_size:]
            mini_batch_Y = shuffled_Y[:,
                                      num_complete_minibatches * self.batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def normalize_input(self, X, fit=False):
        if fit:
            # Calculate and store mean/std from training data
            self.input_mean = np.mean(X, axis=1, keepdims=True)
            # Add epsilon to avoid division by zero
            self.input_std = np.std(X, axis=1, keepdims=True) + 1e-8

        # Apply normalization
        X_normalized = (X - self.input_mean) / self.input_std
        return X_normalized

    def predict(self, X, y):
        m = X.shape[1]

        X = self.normalize_input(X, fit=False)
        # Use training=False for prediction (no dropout)
        y_pred, caches = self.forward_prop.forward_propagation(
            X, training=False)

        if y.shape[0] == 1:
            # Binary classification
            y_pred = np.array(y_pred > 0.5, dtype='float')
            accuracy = np.sum(y_pred == y) / m  # Use original y here
        else:
            # Multi-class classification
            y_copy = np.argmax(y, 0)  # Convert one-hot to class indices
            y_pred = np.argmax(y_pred, 0)
            accuracy = np.sum(y_pred == y_copy) / m  # Use y_copy here

        return np.round(accuracy, 2)

    def train(self, X, Y, num_iterations=3000, X_train=None, Y_train=None, X_test=None, Y_test=None):
        np.random.seed(1)
        costs = []
        parameters = self.parameters.get_parameters()

        X = self.normalize_input(X, fit=True)  # Fit on training data
        if X_train is not None:
            X_train = self.normalize_input(X_train, fit=False)
        if X_test is not None:
            X_test = self.normalize_input(X_test, fit=False)

        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batch

            # Forward propagation on mini-batch
            AL, forward_cache = self.forward_prop.forward_propagation(
                mini_batch_X, training=True)

            # Compute cost for this mini-batch
            mini_batch_cost = LossFunction.compute_cost(AL, mini_batch_Y)
            epoch_cost += mini_batch_cost * \
                mini_batch_X.shape[1]  # Weight by batch size

            # Backward propagation on mini-batch
            grads = self.back_prop.backward_propagation(
                AL, mini_batch_Y, forward_cache)

            # Update parameters after each mini-batch
            parameters = self.grad_descent.update_parameters(parameters, grads)

        # Calculate average cost for the epoch
        epoch_cost = epoch_cost / X.shape[1]

        # Print progress
        if i % (num_iterations/10) == 0:
            train_acc = self.predict(
                X_train, Y_train) if X_train is not None else 0
            test_acc = self.predict(
                X_test, Y_test) if X_test is not None else 0
            print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:{}".format(
                i, np.round(epoch_cost, 2), train_acc, test_acc))

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

# Train model with dropout
layers_dims = [X_train.shape[0], 16, 8, Y_train.shape[0]]  # 4-layer model
lr = 0.0075
iters = 2500

# Define dropout probabilities for each layer
# [input_layer, hidden1, hidden2, output_layer]
# No dropout on input (1.0) and output (1.0) layers
keep_probs = [1.0, 0.8, 0.7, 1.0]  # 20% dropout on first hidden, 30% on second

model = Model(layers_dims, learning_rate=lr,
              activation='relu', keep_probs=keep_probs,
              batch_size=32)
training = Training(model)
parameters = training.train_model(X_train, Y_train, num_iterations=iters,
                                  X_train=X_train, Y_train=Y_train,
                                  X_test=X_test, Y_test=Y_test)
