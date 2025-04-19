## **Experiment 7**: WAP to retrain a pretrained imagenet model to classify a medical image dataset(HAM10000 Dataset).

---

## **Model Description**

In this experiment, I used **DenseNet121**, a convolutional neural network originally trained on ImageNet. The idea is to leverage its ability to extract general image features and fine-tune it to recognize specific patterns in skin lesion images.

DenseNet is known for its “dense connections,” where each layer receives input from all previous layers, which helps with gradient flow and feature reuse.

---

## **Code Description**

The code performs the following steps:

1. **Data Preparation**

   - Reads the metadata CSV to label images.
   - Copies images from the source folder and sorts them into subfolders per class.
   - Uses `ImageDataGenerator` for preprocessing and splitting into train/validation/test sets.

2. **Model Creation**

   - Loads pretrained DenseNet121 (without top layers).
   - Adds custom fully connected layers on top for 7-class classification.
   - Freezes the base model initially, then unfreezes for fine-tuning.

3. **Training**

   - Trained in two phases:
     1. Initial training with frozen base model for 5 epochs.
     2. Fine-tuning with all layers trainable for 10 more epochs.

4. **Evaluation**
   - Accuracy and loss tracked during training.
   - Predictions made on test set.
   - Confusion matrix, classification report, and plots generated.

---
