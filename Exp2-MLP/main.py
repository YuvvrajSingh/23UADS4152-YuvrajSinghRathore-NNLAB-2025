# Objective - to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        """Initialize the multi-layer perceptron (MLP) with random weights and biases."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases for input-hidden and hidden-output layers
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid for backpropagation."""
        return x * (1 - x)

    def forward(self, inputs):
        """Forward propagation: compute hidden and output layer activations."""
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, inputs, expected_output, actual_output):
        """Backpropagation: Adjust weights based on error."""
        output_error = expected_output - actual_output
        output_delta = output_error * self.sigmoid_derivative(actual_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, inputs, expected_output):
        """Train the MLP using forward and backward propagation."""
        for _ in range(self.epochs):
            actual_output = self.forward(inputs)
            self.backward(inputs, expected_output, actual_output)

    def predict(self, inputs):
        """Make predictions using the trained model."""
        return self.forward(inputs)

# XOR input and expected output
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([[0], [1], [1], [0]])  # Expected output

# Initialize and train the MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.train(xor_inputs, xor_labels)

# Test the trained MLP on XOR inputs
print("Testing MLP on XOR function:")
for inputs in xor_inputs:
    predicted_output = mlp.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {predicted_output.round().astype(int)}")

"""
Explanation:

1. **MLP Class**:
   - Initializes weights and biases for two layers.
   - Uses a **sigmoid activation function** for non-linearity.
   - Implements **forward propagation** to compute predictions.
   - Uses **backpropagation** to adjust weights and minimize error.

2. **Training Process**:
   - Runs **10,000 epochs** for learning.
   - Uses **gradient descent** to update weights.

3. **XOR Learning**:
   - Since XOR is not linearly separable, a **single-layer perceptron fails**.
   - But this **multi-layer perceptron (MLP) learns XOR correctly** using the hidden layer.
"""
