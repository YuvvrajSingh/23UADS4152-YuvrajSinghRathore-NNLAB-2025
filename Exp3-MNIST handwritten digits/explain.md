## Objective

To implement a **three-layer neural network** using the **TensorFlow** library (**without Keras**) for classifying the **MNIST handwritten digits dataset**, showcasing the **feed-forward** and **back-propagation** approaches.

---

## Description of the Model

This neural network consists of:

- **Input Layer:** 784 neurons (flattened 28x28 images).
- **Hidden Layer 1:** 128 neurons with **ReLU** activation.
- **Hidden Layer 2:** 64 neurons with **ReLU** activation.
- **Output Layer:** 10 neurons (digit classes 0–9).
- **Feed-Forward:** Passes input through layers to generate predictions.
- **Back-Propagation:** Optimizes weights using gradient descent to minimize loss.

---

## Description of Code

1. **Data Preparation:**

   - Loads MNIST dataset.
   - Normalizes pixel values to [0,1].
   - Flattens images and applies **one-hot encoding** to labels.

2. **Network Parameters:**

   - Defines architecture, learning rate (0.01), epochs (10), and batch size (100).
   - Initializes weights with random values and biases with zeros.

3. **Neural Network Class (`NeuralNetwork`):**

   - **Feed-Forward:**
     - Layer 1: `ReLU(W1 * X + b1)`
     - Layer 2: `ReLU(W2 * L1 + b2)`
     - Output: `W3 * L2 + b3` (logits).
   - **Back-Propagation:**
     - Uses `tf.GradientTape` for automatic differentiation.
     - Optimizes with **SGD** (Stochastic Gradient Descent).

4. **Training Loop:**

   - For each epoch:
     - Processes batches, computes loss, and updates weights.
     - Calculates and displays training accuracy per epoch.

5. **Model Evaluation:**

   - Predicts on test data and computes **test accuracy**.

6. **Single Image Prediction:**
   - Predicts a random test image’s digit.
   - Displays the image with **actual** and **predicted** labels.
   - Shows **probability distribution** over all classes.

---

## Performance Evaluation

- **Training Accuracy:** Gradually improves over epochs, showing effective learning.
- **Test Accuracy:** Achieves competitive performance, validating generalization.
- **Prediction Visualization:** Correctly predicts and visualizes single test samples.

---

## Limitations

- **No Backpropagation Customization:** Relies solely on `tf.GradientTape`.
- **Static Hyperparameters:** Fixed learning rate and architecture.
- **No Regularization:** Lacks dropout or L2 regularization for better generalization.

---

## Scope for Improvement

- **Add Dropout Layers:** To prevent overfitting.
- **Advanced Optimizers:** Use **Adam** or **RMSProp** for faster convergence.
- **Dynamic Learning Rate:** Implement learning rate decay for optimization.
- **Deeper Architectures:** Introduce more hidden layers for complex feature extraction.
- **Backpropagation Customization:** Implement manual gradient calculations for deeper understanding.
