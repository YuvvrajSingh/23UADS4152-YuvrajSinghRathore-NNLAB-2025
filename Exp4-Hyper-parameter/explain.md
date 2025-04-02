<h1> Object 4 </h1> 
<h4>WAP to evaluate the performance of implemented three-layer neural network with
variations in activation functions, size of hidden layer, learning rate, batch size and
number of epochs.</h4>
<hr>

<h3>Description of the model:-</h3>

1. **Training Combinations**: The model is trained with varying batch sizes (1, 10, 100) and epochs (10, 50, 100).
2. **Data Preprocessing**: Pixel values are scaled to [0, 1], images are reshaped to 784-dimensional vectors, and labels are one-hot encoded.
3. **Input Layer**: The input consists of 784 neurons, one for each pixel of the 28x28 image.<

4. **Hidden Layer**: 256 neurons with Xavier initialization and ReLU activation function.
5. **Output Layer**: 10 neurons, one for each possible digit, with raw logits.
6. **Loss Function**: Softmax cross-entropy loss is used to compare the logits with one-hot encoded labels.
7. **Optimizer**: Adam optimizer with a learning rate of 0.1.
8. **Training Process**: Trains over 50 epochs with a batch size of 10, supporting pause and resume via Ctrl+C. Loss is averaged per batch and displayed using a tqdm progress bar.
9. **Performance Evaluation**: Accuracy is computed after each epoch using the test set and is compared to the true labels.
10. **Visualization & Metrics**: Loss curve, accuracy curve, and confusion matrix are used for performance analysis. Training time and test accuracy are calculated after training.
<hr>
<h3>Description of the code :-</h3>
<ol>

<li><b>Data Loading and Preprocessing:</b></li> 
<ul>
<li>Loading MNIST: The MNIST dataset, which contains 28x28 grayscale images of handwritten digits (0-9), is loaded using "tf.keras.datasets.mnist". The dataset is divided into training and test sets.</li>
<li>Normalization: The pixel values of the images are normalized to a range between 0 and 1 by dividing each pixel by 255.0.</li>
<li>Reshaping: The images are reshaped from 28x28 2D arrays into 1D arrays of size 784 (28 * 28).</li>
<li>One-Hot Encoding: The target labels (digits) are converted into one-hot encoded vectors, where each digit is represented as a vector of length 10 (with 1 at the index corresponding to the digit and 0 elsewhere).</li>
</ul>
<br><hr><br>

<li><b>Model Definition:</b></li>
<ul>
<li>A custom "NeuralNetwork" class is defined to represent the neural network model.</li>
<li>The model consists of two layers:<br>
a.) Layer 1: A fully connected layer (784 input units to 256 hidden units), initialized using Xavier (Glorot) initialization.<br>
b.) Layer 2: Layer 2: A fully connected layer (256 hidden units to 10 output units), representing the 10 possible digits.</li>
<li>The activation function used in the hidden layer is ReLU (tf.nn.relu).</li>
<li>The raw logits (unscaled output values) are returned by the model's forward pass (no softmax activation is applied here).</li>
</ul>
<br><hr><br>

<li><b>Loss Function and Optimizer:</b></li>
<ul>
<li>Loss Function: "softmax_cross_entropy_with_logits" is used as the loss function. This computes the cross-entropy between the true one-hot encoded labels and the raw logits.</li>
<li>Optimizer: The Adam optimizer is used for gradient descent optimization, with a learning rate of 0.1.</li>
</ul>
<br><hr><br>

<li><b>Training Function:</b></li>
<ul>
<li>"train_step" performs a forward pass through the network, calculates the loss, computes gradients using "tf.GradientTape", and updates the model's weights using the Adam optimizer.</li>
</ul>
<br><hr><br>

<li><b>Evaluation Function:</b></li>
<ul><li>"evaluate" computes the model's accuracy on the test set. The raw logits are passed through a softmax activation to obtain probabilities, and the predicted class is determined by selecting the class with the highest probability. Accuracy is computed as the proportion of correct predictions.</li>
</ul>
<br><hr><br>

<li><b>Pause Training Functionality:</b></li>
<ul>
<li>A signal handler for "SIGINT" (typically triggered by pressing Ctrl+C) is set up to pause the training process. When the signal is caught, the training pauses, and the user is prompted to press Enter to resume.</li>
</ul>
<br><hr><br>

<li><b>Training Loop:</b></li>
<ul>
<li>The training loop iterates over the dataset for "num_epochs" (50 epochs in this case). In each epoch:<br>
a.) The training data is divided into batches of size "batch_size"(10).<br>
b.) The model is trained in each batch, and the average loss for the epoch is computed.<br>
c.) The test set is evaluated at the end of each epoch, and the accuracy is stored for later visualization.<br></li>
<li>The training process also includes a progress bar using "tqdm" to show the current progress within each epoch.</li>
</ul>
<br><hr><br>

<li><b>Training Time:</b></li>
<ul><li>The total training time is calculated by measuring the time before and after the training loop.</li></ul>
<br><hr><br>

<li><b>Visualization:</b></li>
<ul><li>Loss Curve: A plot of the loss over epochs is generated to visualize the training progress and convergence.</li>
<li>Accuracy Curve: A plot of the accuracy over epochs is also generated to track the model's performance on the test set during training.</li></ul>
<br><hr><br>

<li><b>Confusion Matrix:</b></li><ul>
<li>After training, the model is evaluated on the test set, and predictions are made.</li>
<li>A confusion matrix is generated using the true labels and predicted labels to visualize the classification performance. This matrix is then plotted using a heatmap to show how well the model is performing across different classes.</li>
</ul>
<br><hr><br>

</ol>
<h3> My Comments</h3>

I noticed something interesting about how batch size and the number of epochs affect test accuracy. The best accuracy I got—33.89%—happened when both the batch size and epochs were set to 10. But when I pushed the batch size down to just 1 and cranked the epochs up to 100, the accuracy dropped to its lowest point: 9.58%.

So, simply increasing the number of epochs while reducing the batch size doesn’t guarantee better accuracy. Instead, there's an optimal balance that needs to be found rather than assuming "more training = better results."

<hr>
