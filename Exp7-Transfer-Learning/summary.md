## Output:

```
Found 4000 validated image filenames belonging to 7 classes.
Found 1000 validated image filenames belonging to 7 classes.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
29084464/29084464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 125s 787ms/step - accuracy: 0.6518 - loss: 1.1712 - val_accuracy: 0.6990 - val_loss: 0.8929
Epoch 2/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 77s 614ms/step - accuracy: 0.7163 - loss: 0.8168 - val_accuracy: 0.7230 - val_loss: 0.7976
Epoch 3/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 82s 611ms/step - accuracy: 0.7306 - loss: 0.7280 - val_accuracy: 0.7140 - val_loss: 0.7733
Epoch 4/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 83s 617ms/step - accuracy: 0.7445 - loss: 0.6979 - val_accuracy: 0.7340 - val_loss: 0.7478
Epoch 5/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 77s 618ms/step - accuracy: 0.7517 - loss: 0.6761 - val_accuracy: 0.7420 - val_loss: 0.7226
Epoch 1/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 264s 840ms/step - accuracy: 0.6953 - loss: 0.8959 - val_accuracy: 0.7370 - val_loss: 0.7620
Epoch 2/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 86s 687ms/step - accuracy: 0.7569 - loss: 0.6785 - val_accuracy: 0.7530 - val_loss: 0.6911
Epoch 3/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 88s 701ms/step - accuracy: 0.7925 - loss: 0.5809 - val_accuracy: 0.7820 - val_loss: 0.6083
Epoch 4/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 87s 696ms/step - accuracy: 0.8248 - loss: 0.5081 - val_accuracy: 0.8030 - val_loss: 0.5754
Epoch 5/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 87s 692ms/step - accuracy: 0.8350 - loss: 0.4707 - val_accuracy: 0.7970 - val_loss: 0.5896
Epoch 6/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 88s 703ms/step - accuracy: 0.8451 - loss: 0.4269 - val_accuracy: 0.7860 - val_loss: 0.5873
Epoch 7/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 88s 706ms/step - accuracy: 0.8628 - loss: 0.3983 - val_accuracy: 0.7790 - val_loss: 0.6408
Epoch 8/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 86s 688ms/step - accuracy: 0.8817 - loss: 0.3472 - val_accuracy: 0.7740 - val_loss: 0.6838
Epoch 9/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 89s 712ms/step - accuracy: 0.8777 - loss: 0.3420 - val_accuracy: 0.7720 - val_loss: 0.6874
Epoch 10/10
125/125 ━━━━━━━━━━━━━━━━━━━━ 141s 698ms/step - accuracy: 0.8865 - loss: 0.3203 - val_accuracy: 0.7950 - val_loss: 0.6297
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
```

---

## **Performance Evaluation**

### **Accuracy**

- Final **test accuracy**: 80%
- Validation accuracy peaked at **~80.3%**.

### **Confusion Matrix**

- The model performs well on the **‘nv’ (nevus)** class, which is the most common in the dataset.
- Classes like **‘akiec’ (Actinic keratoses)** and **‘df’ (Dermatofibroma)** had much lower recall, which means the model often misclassified these.

![Confusion Matrix](attachment:/mnt/data/download.png)

### **Training Curves**

- **Training Accuracy** kept improving with epochs, showing the model was learning.
- **Validation Accuracy** plateaued and slightly declined after epoch 8–10 in the second training phase, suggesting slight overfitting.

![Training and Validation Curves](<attachment:/mnt/data/download%20(1).png>)

### **Classification Report (Test Set)**

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| akiec | 0.88      | 0.19   | 0.31     |
| bcc   | 0.69      | 0.68   | 0.69     |
| bkl   | 0.49      | 0.88   | 0.63     |
| df    | 1.00      | 0.36   | 0.53     |
| mel   | 0.48      | 0.46   | 0.47     |
| nv    | 0.94      | 0.87   | 0.91     |
| vasc  | 0.65      | 0.85   | 0.73     |

- **Macro F1-Score**: 0.61
- **Weighted F1-Score**: 0.80

---

## **My Comments**

1. **Class Imbalance**: The model struggles with underrepresented classes like 'akiec' and 'df'. Balancing the dataset (via augmentation or oversampling) could help.

2. **Model Bias**: It performs well on 'nv' (the majority class) but poorly on rarer classes. This shows the model might be biased toward the more frequent types.

3. **Overfitting Signs**: Validation accuracy stops improving in later epochs. Regularization techniques like dropout, early stopping, or data augmentation could help further.

---
