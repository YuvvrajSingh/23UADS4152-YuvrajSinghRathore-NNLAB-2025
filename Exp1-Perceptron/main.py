# Objective: Implement the Perceptron Learning Algorithm using NumPy in Python.Evaluate the performance of a single-layer perceptron for NAND and XOR truth tables.

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """Initialize perceptron with random weights, learning rate, and number of epochs."""
        self.weights = np.random.randn(input_size + 1)  # Extra weight for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Step activation function: Returns 1 if x is non-negative, otherwise 0."""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """Compute the perceptron output for a given input."""
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        """Train the perceptron using the perceptron learning rule."""
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias term
                y_pred = self.activation(np.dot(self.weights, x_i))
                self.weights += self.learning_rate * (y[i] - y_pred) * x_i  # Update rule


# Define input dataset (truth table inputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define expected outputs
Y_nand = np.array([1, 1, 1, 0])  # NAND truth table
Y_xor = np.array([0, 1, 1, 0])   # XOR truth table

# Train perceptron for NAND function
perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X, Y_nand)

# Test perceptron for NAND function
print("NAND Perceptron Output:")
for x in X:
    print(f"Input: {x}, Output: {perceptron_nand.predict(x)}")

# Train perceptron for XOR function
perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X, Y_xor)

# Test perceptron for XOR function
print("XOR Perceptron Output:")
for x in X:
    print(f"Input: {x}, Output: {perceptron_xor.predict(x)}")

"""
Explanation:

Perceptron Class:
- Initializes weights (including a bias) randomly.
- Uses a simple step function for activation.
- Updates weights using the perceptron learning rule.

Training Process:
- Runs through the dataset multiple times (epochs).
- Adjusts weights when a wrong prediction is made.

Results:
- NAND function: The perceptron successfully learns and correctly classifies the input.
- XOR function: The perceptron fails because XOR is not linearly separable.
"""
