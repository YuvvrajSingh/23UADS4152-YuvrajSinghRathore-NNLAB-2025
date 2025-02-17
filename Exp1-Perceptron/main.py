# Objective: Implement the Perceptron Learning Algorithm using NumPy in Python.Evaluate the performance of a single-layer perceptron for NAND and XOR truth tables.

import numpy as np 

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """Initialize the perceptron with random weights, learning rate, and training cycles (epochs)."""
        self.weights = np.random.randn(input_size + 1)  
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Step function: If x >= 0, return 1, otherwise return 0."""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """Make a prediction based on input x."""
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))  

    def train(self, X, y):
        """Train the perceptron by adjusting weights when predictions are wrong."""
        for _ in range(self.epochs):  # Repeat multiple times to improve accuracy
            for i in range(len(X)):  # Go through each training example
                x_i = np.insert(X[i], 0, 1)  # Add bias term
                y_pred = self.activation(np.dot(self.weights, x_i))  # Make a prediction
                self.weights += self.learning_rate * (y[i] - y_pred) * x_i  # Update weights

# Define input dataset (Truth Table inputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # All possible input combinations

# Define expected outputs
Y_nand = np.array([1, 1, 1, 0])  # Expected output for NAND gate
Y_xor = np.array([0, 1, 1, 0])   # Expected output for XOR gate

# Train and test perceptron for NAND function
perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X, Y_nand)
print("NAND Perceptron Output:")
for x in X:
    print(f"Input: {x}, Output: {perceptron_nand.predict(x)}")

# Train and test perceptron for XOR function
perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X, Y_xor)
print("XOR Perceptron Output:")
for x in X:
    print(f"Input: {x}, Output: {perceptron_xor.predict(x)}")

"""
Explanation:

- The Perceptron is a simple artificial neuron that makes decisions based on inputs.
- It learns by adjusting weights when its predictions are wrong.
- The NAND function is successfully learned because it is linearly separable.
- The XOR function fails because a single-layer perceptron cannot solve non-linearly separable problems.
"""
