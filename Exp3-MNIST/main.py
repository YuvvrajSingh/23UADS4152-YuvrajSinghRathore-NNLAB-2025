import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Step 1: Load MNIST data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Flatten the 28x28 images into a 1D array of size 784 (28*28)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Convert the labels to one-hot encoded vectors
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]


# Step 2: Define the model using tf.Variable
class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.W1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))
        self.W2 = tf.Variable(tf.random.normal([128, 64], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([64]))
        self.W3 = tf.Variable(tf.random.normal([64, 10], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([10]))

    def forward(self, x):
        x = tf.matmul(x, self.W1) + self.b1
        x = tf.nn.sigmoid(x)
        x = tf.matmul(x, self.W2) + self.b2
        x = tf.nn.sigmoid(x)
        x = tf.matmul(x, self.W3) + self.b3
        return tf.nn.softmax(x)


# Step 3: Instantiate the model
model = NeuralNetwork()

# Step 4: Define the loss function and optimizer
loss_fn = tf.nn.softmax_cross_entropy_with_logits
optimizer = tf.optimizers.Adam(learning_rate=0.01)


# Step 5: Training function
def train_step(model, x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model.forward(x_batch)
        loss = tf.reduce_mean(loss_fn(y_batch, logits))
    grads = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
    optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]))
    return loss


# Step 6: Evaluate the model on test data
def evaluate(model, X_test, y_test):
    predictions = model.forward(X_test)
    accuracy = np.mean(np.argmax(predictions.numpy(), axis=1) == np.argmax(y_test, axis=1))
    return accuracy


# Step 7: Training loop
num_epochs = 10
batch_size = 64
num_batches = X_train.shape[0] // batch_size
loss_history = []

for epoch in range(num_epochs):
    avg_cost = 0.0
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        start = batch * batch_size
        end = (batch + 1) * batch_size
        batch_X = X_train[start:end]
        batch_Y = y_train_one_hot[start:end]

        loss = train_step(model, batch_X, batch_Y)
        avg_cost += loss.numpy() / num_batches

        progress_bar.set_postfix(loss=avg_cost)

    loss_history.append(avg_cost)
    print(f"Epoch {epoch + 1}, Cost: {avg_cost:.4f}")

# Step 8: Evaluate the model on the test data
test_accuracy = evaluate(model, X_test, y_test_one_hot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 9: Visualize loss curve
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Step 10: Visualize predictions
predictions = model.forward(X_test[:5])
predicted_classes = np.argmax(predictions.numpy(), axis=1)

for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}, Actual: {y_test[i]}")
    plt.show()