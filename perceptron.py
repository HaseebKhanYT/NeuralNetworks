"""
Perceptron Neural Network Implementation

This module implements a single-layer perceptron neural network for image classification.
The perceptron uses sigmoid activation function and backpropagation for training.
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, List, Optional


class Perceptron:
    """
    A single-layer perceptron neural network implementation.

    The perceptron is a fundamental building block of neural networks that can learn
    to classify linearly separable data. It uses sigmoid activation and gradient descent
    for weight updates.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (input_size, 1)
        bias (np.ndarray): Bias term of shape (1,)
        learning_rate (float): Learning rate for gradient descent updates
    """

    def __init__(self, input_size: int, learning_rate: float = 0.01) -> None:
        """
        Initialize the perceptron with random weights and bias.

        Args:
            input_size (int): Number of input features (e.g., flattened image pixels)
            learning_rate (float): Learning rate for weight updates (default: 0.01)
        """
        # Initialize weights with small random values (Xavier initialization)
        self.weights = np.random.randn(input_size, 1) * 0.1
        # Initialize bias to zero
        self.bias = np.zeros((1,))
        self.learning_rate = learning_rate

        # Store input size for validation
        self.input_size = input_size

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        The sigmoid function maps any real number to a value between 0 and 1,
        making it suitable for binary classification problems.

        Args:
            x (np.ndarray): Input values

        Returns:
            np.ndarray: Sigmoid activation values between 0 and 1
        """
        # Clip x to prevent overflow in exp(-x)
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid function.

        Used in backpropagation to compute gradients. The derivative of sigmoid(x)
        is sigmoid(x) * (1 - sigmoid(x)).

        Args:
            x (np.ndarray): Sigmoid output values

        Returns:
            np.ndarray: Derivative values
        """
        return x * (1 - x)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the perceptron.

        Computes the weighted sum of inputs plus bias, then applies sigmoid activation.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, input_size)

        Returns:
            np.ndarray: Output predictions of shape (batch_size, 1)
        """
        # Validate input shape
        if inputs.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {inputs.shape[1]}")

        # Compute weighted sum: inputs @ weights + bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply sigmoid activation
        output = self.sigmoid(weighted_sum)
        return output

    def backward(self, inputs: np.ndarray, targets: np.ndarray, outputs: np.ndarray) -> None:
        """
        Backward propagation to update weights and bias.

        Computes gradients using the chain rule and updates parameters using
        gradient descent.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, input_size)
            targets (np.ndarray): Target values of shape (batch_size, 1)
            outputs (np.ndarray): Predicted outputs of shape (batch_size, 1)
        """
        # Compute prediction error
        error = targets - outputs

        # Compute gradient of sigmoid activation
        gradient = self.sigmoid_derivative(outputs)

        # Compute weight gradients using chain rule
        weight_gradients = np.dot(inputs.T, error * gradient)
        bias_gradients = np.sum(error * gradient, axis=0, keepdims=True)

        # Update weights and bias using gradient descent
        self.weights += self.learning_rate * weight_gradients
        self.bias += self.learning_rate * bias_gradients.flatten()

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int,
              verbose: bool = True) -> List[float]:
        """
        Train the perceptron on the given dataset.

        Performs multiple epochs of training, updating weights after each sample
        using stochastic gradient descent.

        Args:
            inputs (np.ndarray): Training input data
            targets (np.ndarray): Training target labels
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress

        Returns:
            List[float]: Training loss history
        """
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Train on each sample individually (stochastic gradient descent)
            for i in range(len(inputs)):
                # Get single sample and target
                input_sample = inputs[i:i+1]  # Keep 2D shape
                target_sample = targets[i:i+1]

                # Forward pass
                output = self.forward(input_sample)

                # Compute loss (mean squared error)
                loss = np.mean((target_sample - output) ** 2)
                epoch_loss += loss

                # Backward pass
                self.backward(input_sample, target_sample, output)

            # Average loss for this epoch
            avg_loss = epoch_loss / len(inputs)
            loss_history.append(avg_loss)

            # Print progress every 100 epochs
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return loss_history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            inputs (np.ndarray): Input data to predict on

        Returns:
            np.ndarray: Predicted outputs
        """
        return self.forward(inputs)

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the perceptron on test data.

        Args:
            inputs (np.ndarray): Test input data
            targets (np.ndarray): Test target labels

        Returns:
            Tuple[float, float]: (accuracy, average_loss)
        """
        predictions = self.predict(inputs)

        # Convert predictions to binary (threshold at 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        binary_targets = (targets > 0.5).astype(int)

        # Calculate accuracy
        accuracy = np.mean(binary_predictions == binary_targets)

        # Calculate average loss
        loss = np.mean((targets - predictions) ** 2)

        return accuracy, loss


def load_test_images(directory_path: str, image_size: Tuple[int, int] = (20, 20)) -> np.ndarray:
    """
    Load and preprocess test images from a directory.

    Args:
        directory_path (str): Path to directory containing test images
        image_size (Tuple[int, int]): Target size for resizing images (width, height)

    Returns:
        np.ndarray: Preprocessed image array of shape (num_images, height, width)
    """
    test_images = []

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(
            f"Warning: Directory {directory_path} does not exist. Using generated data instead.")
        return generate_test_data()

    # Look for images with pattern img_*.jpg
    for i in range(10):  # Try to load up to 10 images
        image_path = os.path.join(directory_path, f"img_{i}.jpg")

        if os.path.exists(image_path):
            try:
                # Load and preprocess image
                image = Image.open(image_path)
                # Convert to grayscale and resize
                image = image.convert("L").resize(image_size)
                # Normalize pixel values to [0, 1]
                image_array = np.array(image) / 255.0
                test_images.append(image_array)
                print(f"Loaded image: {image_path}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")

    if not test_images:
        print("No test images found. Using generated data instead.")
        return generate_test_data()

    return np.array(test_images)


def generate_training_data(num_samples: int = 10, image_size: Tuple[int, int] = (20, 20)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for demonstration.

    Args:
        num_samples (int): Number of training samples to generate
        image_size (Tuple[int, int]): Size of generated images (height, width)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (images, labels) where images are of shape
                                      (num_samples, height, width) and labels are
                                      one-hot encoded of shape (num_samples, 1)
    """
    np.random.seed(42)  # For reproducible results

    # Generate random images
    images = np.random.random(size=(num_samples, *image_size))

    # Generate binary labels (first class = 1, others = 0)
    labels = np.zeros((num_samples, 1))
    labels[0] = 1  # First image is positive class

    return images, labels


def generate_test_data(num_samples: int = 5, image_size: Tuple[int, int] = (20, 20)) -> np.ndarray:
    """
    Generate synthetic test data for evaluation.

    Args:
        num_samples (int): Number of test samples to generate
        image_size (Tuple[int, int]): Size of generated images (height, width)

    Returns:
        np.ndarray: Test images of shape (num_samples, height, width)
    """
    np.random.seed(123)  # Different seed for test data
    return np.random.random(size=(num_samples, *image_size))


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Flatten image arrays from 3D to 2D for neural network input.

    Args:
        images (np.ndarray): Image array of shape (num_images, height, width)

    Returns:
        np.ndarray: Flattened array of shape (num_images, height * width)
    """
    return images.reshape(len(images), -1)


def main():
    """
    Main function to demonstrate perceptron training and testing.
    """
    print("=== Perceptron Neural Network Demo ===\n")

    # Generate training data
    print("Generating training data...")
    images, labels = generate_training_data(
        num_samples=10, image_size=(20, 20))
    print(f"Training data shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Flatten images for neural network input
    flattened_images = flatten_images(images)
    print(f"Flattened training data shape: {flattened_images.shape}")

    # Initialize perceptron
    input_size = flattened_images.shape[1]  # 20 * 20 = 400
    perceptron = Perceptron(input_size=input_size, learning_rate=0.1)
    print(f"Initialized perceptron with input size: {input_size}\n")

    # Train the perceptron
    print("Training perceptron...")
    loss_history = perceptron.train(
        flattened_images, labels, epochs=1000, verbose=True)
    print(f"Training completed. Final loss: {loss_history[-1]:.6f}\n")

    # Load or generate test images
    test_images_directory = "/Users/zafar/Desktop/Northeastern University/Neural Networks/Test images"
    print(f"Loading test images from: {test_images_directory}")
    test_images = load_test_images(test_images_directory, image_size=(20, 20))
    print(f"Test images shape: {test_images.shape}\n")

    # Flatten test images
    flattened_test_images = flatten_images(test_images)

    # Make predictions on test data
    print("Making predictions on test data...")
    for i in range(len(flattened_test_images)):
        test_input = flattened_test_images[i:i+1]  # Keep 2D shape
        prediction = perceptron.predict(test_input)
        confidence = prediction[0, 0]

        # Interpret prediction
        predicted_class = "Positive" if confidence > 0.5 else "Negative"
        print(f"Test Image {i + 1}: Prediction = {predicted_class} "
              f"(confidence: {confidence:.4f})")

    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_accuracy, train_loss = perceptron.evaluate(flattened_images, labels)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training Loss: {train_loss:.6f}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
