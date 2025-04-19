## **Experiment**: WAP to train and evaluate a convolutional neural network using Keras Library to classify MNIST fashion dataset. Demonstrate the effect of filter size, regularization, batch size and optimization algorithm on model performance.

---

## **Description of the Model**

This CNN model is made up of:

- **Two convolutional layers** (filter sizes: 3×3 or 5×5) to extract features from the image.
- **MaxPooling layers** to reduce dimensionality and focus on important features.
- **A fully connected layer** to combine features and make predictions.
- **A softmax layer** at the end to classify the image into one of the 10 fashion categories.
- **L2 regularization** is applied to avoid overfitting.
- We used both **Adam** and **SGD** optimizers to compare how each performs with different settings.

---

## **Description of Code**

- **Data Preprocessing**: The Fashion MNIST dataset is loaded and normalized. Each grayscale image is reshaped into 28×28×1 format.
- **Model Construction**: The `create_model` function builds a CNN with two convolutional layers, followed by max pooling, dense layers, and softmax for classification.
- **Regularization & Optimizer**: L2 regularization is applied to convolution and dense layers. Models are compiled with either Adam or SGD optimizers.
- **Training**: The model is trained over 10 epochs across different hyperparameter combinations (filter size, regularization strength, batch size, optimizer).
- **Evaluation**: Each model's validation accuracy is recorded and visualized.

---
