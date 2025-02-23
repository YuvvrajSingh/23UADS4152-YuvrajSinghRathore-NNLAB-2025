import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input data and ensure consistent dtype
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

# Flatten images to 1D vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to one-hot encoding
y_train = np.eye(10, dtype=np.float32)[y_train]
y_test = np.eye(10, dtype=np.float32)[y_test]

# Define network parameters
n_input = 784
n_hidden1 = 128
n_hidden2 = 64
n_output = 10
learning_rate = 0.01
epochs = 10
batch_size = 100

# Define weights and biases with explicit float32 type
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1, dtype=tf.float32))

W1 = init_weights([n_input, n_hidden1])
b1 = tf.Variable(tf.zeros([n_hidden1], dtype=tf.float32))

W2 = init_weights([n_hidden1, n_hidden2])
b2 = tf.Variable(tf.zeros([n_hidden2], dtype=tf.float32))

W3 = init_weights([n_hidden2, n_output])
b3 = tf.Variable(tf.zeros([n_output], dtype=tf.float32))

# Define the model using TensorFlow 2.x functions
class NeuralNetwork(tf.Module):
    def __init__(self):
        super().__init__()
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3

    def __call__(self, X):
        X = tf.cast(X, tf.float32)  # Ensure input is float32
        layer1 = tf.nn.relu(tf.add(tf.matmul(X, self.W1), self.b1))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.W2), self.b2))
        output_layer = tf.add(tf.matmul(layer2, self.W3), self.b3)
        return output_layer

# Instantiate the model
model = NeuralNetwork()

# Define loss function and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# Training loop using GradientTape
for epoch in range(epochs):
    total_batches = len(x_train) // batch_size
    avg_loss = 0

    for i in range(total_batches):
        batch_x = x_train[i * batch_size: (i + 1) * batch_size]
        batch_y = y_train[i * batch_size: (i + 1) * batch_size]

        with tf.GradientTape() as tape:
            logits = model(batch_x)
            loss = loss_fn(batch_y, logits)

        grads = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
        optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]))

        avg_loss += loss.numpy() / total_batches

    # Compute accuracy
    train_logits = model(x_train)
    train_predictions = tf.argmax(train_logits, axis=1)
    train_labels = tf.argmax(y_train, axis=1)
    train_acc = tf.reduce_mean(tf.cast(tf.equal(train_predictions, train_labels), tf.float32))

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Training Accuracy: {train_acc.numpy():.4f}")

# Evaluate model on test set
test_logits = model(x_test)
test_predictions = tf.argmax(test_logits, axis=1)
test_labels = tf.argmax(y_test, axis=1)
test_acc = tf.reduce_mean(tf.cast(tf.equal(test_predictions, test_labels), tf.float32))

print(f"Test Accuracy: {test_acc.numpy():.4f}")

# Function to test the model on a single image
def test_single_image(image_index):
    test_image = x_test[image_index].reshape(1, 784)
    actual_label = np.argmax(y_test[image_index])

    predicted_logits = model(test_image)
    predicted_probabilities = tf.nn.softmax(predicted_logits)
    predicted_label = np.argmax(predicted_probabilities.numpy())

    # Display the image
    plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

    print(f"Predicted probabilities: {predicted_probabilities.numpy()}")

# Test a single random image
test_single_image(image_index=np.random.randint(0, len(x_test)))
