## Objective

To implement the **Perceptron Learning Algorithm** using **NumPy** in Python and evaluate the performance of a **single-layer perceptron** on the **NAND** and **XOR** truth tables.

---

## Description of the Model

A **perceptron** is a fundamental building block of neural networks that classifies inputs into two categories based on a weighted sum. The learning process adjusts weights using a simple rule until convergence.

**Key Components:**

- **Step Activation Function:** Outputs 1 if input ≥ 0, otherwise 0.
- **Weight & Bias Update Rule:** Adjusted based on prediction errors.
- **Training Process:** Iterative updates to minimize classification errors.
- **Prediction Function:** Computes output for new inputs using trained weights.

---

## Description of Code

1. **Initialization:**

   - Randomly initializes weights (including bias).
   - Defines learning rate and number of training epochs.

2. **Activation Function:**

   - Implements a **step function** for binary classification.

3. **Prediction Method:**

   - Adds bias term and computes output using the activation function.

4. **Training Process:**

   - Iterates through the dataset.
   - Updates weights using the formula:  
     \[
     w \leftarrow w + \alpha \times (y*{\text{true}} - y*{\text{pred}}) \times x
     \]
   - Continues for a fixed number of epochs to ensure convergence if possible.

5. **Evaluation:**
   - Tests the trained perceptron on **NAND** and **XOR** truth tables.
   - Displays predicted outputs for each input combination.

---

## Performance Evaluation

### NAND Gate (Linearly Separable):

- **Expected Output:** `[1, 1, 1, 0]`
- **Predicted Output:** Matches expected outputs.
- **Result:** ✅ **Successfully learned NAND**, confirming the perceptron’s ability to handle linearly separable problems.

### XOR Gate (Non-Linearly Separable):

- **Expected Output:** `[0, 1, 1, 0]`
- **Predicted Output:** Fails to match expected outputs.
- **Result:** ❌ **Failed to learn XOR**, highlighting that a single-layer perceptron cannot solve non-linearly separable problems.

---

## Limitations

- **Inability to Solve Non-Linear Problems:**
  - Fails on XOR due to lack of hidden layers.
- **Rigid Activation Function:**
  - The step function prevents gradient-based optimization techniques.
- **No Early Stopping Mechanism:**
  - Continues training even after potential convergence.

---

## Scope for Improvement

- **Multi-Layer Perceptron (MLP):**
  - Introduce a hidden layer to handle non-linear separability (e.g., XOR).
