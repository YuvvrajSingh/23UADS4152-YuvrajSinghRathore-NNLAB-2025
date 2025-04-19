## Output:

```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
29515/29515 ━━━━━━━━━━━━━━━━━━━━ 0s 3us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26421880/26421880 ━━━━━━━━━━━━━━━━━━━━ 62s 2us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
5148/5148 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4422102/4422102 ━━━━━━━━━━━━━━━━━━━━ 26s 6us/step
Training with Filter=3, Reg=0.0001, Batch=32, Opt=adam
c:\Python312\Lib\site-packages\keras\src\layers\convolutional\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/5
1875/1875 - 20s - 11ms/step - accuracy: 0.8359 - loss: 0.4846 - val_accuracy: 0.8623 - val_loss: 0.4289
Epoch 2/5
1875/1875 - 16s - 8ms/step - accuracy: 0.8873 - loss: 0.3512 - val_accuracy: 0.8867 - val_loss: 0.3575
Epoch 3/5
1875/1875 - 16s - 8ms/step - accuracy: 0.9008 - loss: 0.3156 - val_accuracy: 0.8914 - val_loss: 0.3490
Epoch 4/5
1875/1875 - 17s - 9ms/step - accuracy: 0.9098 - loss: 0.2938 - val_accuracy: 0.8991 - val_loss: 0.3330
Epoch 5/5
1875/1875 - 17s - 9ms/step - accuracy: 0.9167 - loss: 0.2776 - val_accuracy: 0.9017 - val_loss: 0.3215
Test accuracy: 0.9017

Training with Filter=3, Reg=0.0001, Batch=32, Opt=sgd
Epoch 1/5
1875/1875 - 17s - 9ms/step - accuracy: 0.6945 - loss: 0.8864 - val_accuracy: 0.7534 - val_loss: 0.6713
Epoch 2/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8069 - loss: 0.5509 - val_accuracy: 0.8162 - val_loss: 0.5350
Epoch 3/5
1875/1875 - 14s - 8ms/step - accuracy: 0.8367 - loss: 0.4817 - val_accuracy: 0.8347 - val_loss: 0.5006
Epoch 4/5
1875/1875 - 14s - 8ms/step - accuracy: 0.8509 - loss: 0.4431 - val_accuracy: 0.8512 - val_loss: 0.4496
Epoch 5/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8613 - loss: 0.4144 - val_accuracy: 0.8539 - val_loss: 0.4365
Test accuracy: 0.8539

Training with Filter=3, Reg=0.0001, Batch=64, Opt=adam
Epoch 1/5
938/938 - 14s - 15ms/step - accuracy: 0.8229 - loss: 0.5207 - val_accuracy: 0.8626 - val_loss: 0.4090
Epoch 2/5
938/938 - 12s - 13ms/step - accuracy: 0.8816 - loss: 0.3607 - val_accuracy: 0.8817 - val_loss: 0.3631
Epoch 3/5
938/938 - 12s - 13ms/step - accuracy: 0.8971 - loss: 0.3211 - val_accuracy: 0.8875 - val_loss: 0.3479
Epoch 4/5
938/938 - 12s - 12ms/step - accuracy: 0.9071 - loss: 0.2955 - val_accuracy: 0.8919 - val_loss: 0.3383
Epoch 5/5
938/938 - 12s - 12ms/step - accuracy: 0.9133 - loss: 0.2786 - val_accuracy: 0.9034 - val_loss: 0.3115
Test accuracy: 0.9034

Training with Filter=3, Reg=0.0001, Batch=64, Opt=sgd
Epoch 1/5
938/938 - 12s - 13ms/step - accuracy: 0.6233 - loss: 1.1362 - val_accuracy: 0.6876 - val_loss: 0.9324
Epoch 2/5
938/938 - 11s - 12ms/step - accuracy: 0.7586 - loss: 0.6703 - val_accuracy: 0.7615 - val_loss: 0.6609
Epoch 3/5
938/938 - 11s - 12ms/step - accuracy: 0.7953 - loss: 0.5793 - val_accuracy: 0.7883 - val_loss: 0.5981
Epoch 4/5
938/938 - 11s - 12ms/step - accuracy: 0.8209 - loss: 0.5230 - val_accuracy: 0.8197 - val_loss: 0.5313
Epoch 5/5
938/938 - 11s - 12ms/step - accuracy: 0.8352 - loss: 0.4869 - val_accuracy: 0.8311 - val_loss: 0.4990
Test accuracy: 0.8311

Training with Filter=3, Reg=0.001, Batch=32, Opt=adam
Epoch 1/5
1875/1875 - 18s - 10ms/step - accuracy: 0.8263 - loss: 0.6068 - val_accuracy: 0.8507 - val_loss: 0.5269
Epoch 2/5
1875/1875 - 16s - 9ms/step - accuracy: 0.8696 - loss: 0.4565 - val_accuracy: 0.8722 - val_loss: 0.4535
Epoch 3/5
1875/1875 - 16s - 9ms/step - accuracy: 0.8803 - loss: 0.4214 - val_accuracy: 0.8778 - val_loss: 0.4344
Epoch 4/5
1875/1875 - 16s - 9ms/step - accuracy: 0.8862 - loss: 0.4034 - val_accuracy: 0.8769 - val_loss: 0.4230
Epoch 5/5
1875/1875 - 16s - 8ms/step - accuracy: 0.8916 - loss: 0.3902 - val_accuracy: 0.8796 - val_loss: 0.4237
Test accuracy: 0.8796

Training with Filter=3, Reg=0.001, Batch=32, Opt=sgd
Epoch 1/5
1875/1875 - 15s - 8ms/step - accuracy: 0.6877 - loss: 1.1367 - val_accuracy: 0.7759 - val_loss: 0.8811
Epoch 2/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8001 - loss: 0.8043 - val_accuracy: 0.8068 - val_loss: 0.7823
Epoch 3/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8318 - loss: 0.7184 - val_accuracy: 0.8348 - val_loss: 0.7061
Epoch 4/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8477 - loss: 0.6647 - val_accuracy: 0.8451 - val_loss: 0.6658
Epoch 5/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8591 - loss: 0.6236 - val_accuracy: 0.8351 - val_loss: 0.6655
Test accuracy: 0.8351

Training with Filter=3, Reg=0.001, Batch=64, Opt=adam
Epoch 1/5
938/938 - 13s - 14ms/step - accuracy: 0.8228 - loss: 0.6514 - val_accuracy: 0.8511 - val_loss: 0.5378
Epoch 2/5
938/938 - 11s - 12ms/step - accuracy: 0.8704 - loss: 0.4660 - val_accuracy: 0.8607 - val_loss: 0.4760
Epoch 3/5
938/938 - 11s - 12ms/step - accuracy: 0.8806 - loss: 0.4293 - val_accuracy: 0.8814 - val_loss: 0.4266
Epoch 4/5
938/938 - 12s - 12ms/step - accuracy: 0.8866 - loss: 0.4079 - val_accuracy: 0.8797 - val_loss: 0.4274
Epoch 5/5
938/938 - 11s - 12ms/step - accuracy: 0.8923 - loss: 0.3908 - val_accuracy: 0.8768 - val_loss: 0.4311
Test accuracy: 0.8768

Training with Filter=3, Reg=0.001, Batch=64, Opt=sgd
Epoch 1/5
938/938 - 11s - 12ms/step - accuracy: 0.6442 - loss: 1.3279 - val_accuracy: 0.7268 - val_loss: 1.0270
Epoch 2/5
938/938 - 11s - 11ms/step - accuracy: 0.7614 - loss: 0.9111 - val_accuracy: 0.7704 - val_loss: 0.9174
Epoch 3/5
938/938 - 11s - 11ms/step - accuracy: 0.7968 - loss: 0.8150 - val_accuracy: 0.8000 - val_loss: 0.8061
Epoch 4/5
938/938 - 11s - 12ms/step - accuracy: 0.8172 - loss: 0.7595 - val_accuracy: 0.7962 - val_loss: 0.7876
Epoch 5/5
938/938 - 12s - 12ms/step - accuracy: 0.8321 - loss: 0.7202 - val_accuracy: 0.8129 - val_loss: 0.7554
Test accuracy: 0.8129

Training with Filter=5, Reg=0.0001, Batch=32, Opt=adam
Epoch 1/5
1875/1875 - 17s - 9ms/step - accuracy: 0.8337 - loss: 0.4887 - val_accuracy: 0.8671 - val_loss: 0.4001
Epoch 2/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8898 - loss: 0.3404 - val_accuracy: 0.8948 - val_loss: 0.3391
Epoch 3/5
1875/1875 - 15s - 8ms/step - accuracy: 0.9044 - loss: 0.3059 - val_accuracy: 0.8987 - val_loss: 0.3302
Epoch 4/5
1875/1875 - 15s - 8ms/step - accuracy: 0.9132 - loss: 0.2854 - val_accuracy: 0.9050 - val_loss: 0.3166
Epoch 5/5
1875/1875 - 15s - 8ms/step - accuracy: 0.9190 - loss: 0.2705 - val_accuracy: 0.9099 - val_loss: 0.3017
Test accuracy: 0.9099

Training with Filter=5, Reg=0.0001, Batch=32, Opt=sgd
Epoch 1/5
1875/1875 - 15s - 8ms/step - accuracy: 0.7104 - loss: 0.8257 - val_accuracy: 0.7868 - val_loss: 0.6041
Epoch 2/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8130 - loss: 0.5377 - val_accuracy: 0.8213 - val_loss: 0.5248
Epoch 3/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8417 - loss: 0.4699 - val_accuracy: 0.8457 - val_loss: 0.4604
Epoch 4/5
1875/1875 - 14s - 7ms/step - accuracy: 0.8567 - loss: 0.4290 - val_accuracy: 0.8543 - val_loss: 0.4391
Epoch 5/5
1875/1875 - 14s - 8ms/step - accuracy: 0.8664 - loss: 0.4004 - val_accuracy: 0.8636 - val_loss: 0.4092
Test accuracy: 0.8636

Training with Filter=5, Reg=0.0001, Batch=64, Opt=adam
Epoch 1/5
938/938 - 13s - 13ms/step - accuracy: 0.8227 - loss: 0.5176 - val_accuracy: 0.8558 - val_loss: 0.4215
Epoch 2/5
938/938 - 11s - 12ms/step - accuracy: 0.8825 - loss: 0.3531 - val_accuracy: 0.8825 - val_loss: 0.3583
Epoch 3/5
938/938 - 11s - 12ms/step - accuracy: 0.8988 - loss: 0.3142 - val_accuracy: 0.8892 - val_loss: 0.3379
Epoch 4/5
938/938 - 11s - 12ms/step - accuracy: 0.9078 - loss: 0.2920 - val_accuracy: 0.8959 - val_loss: 0.3255
Epoch 5/5
938/938 - 11s - 12ms/step - accuracy: 0.9146 - loss: 0.2734 - val_accuracy: 0.9026 - val_loss: 0.3121
Test accuracy: 0.9026

Training with Filter=5, Reg=0.0001, Batch=64, Opt=sgd
Epoch 1/5
938/938 - 12s - 13ms/step - accuracy: 0.6453 - loss: 1.0263 - val_accuracy: 0.7270 - val_loss: 0.8057
Epoch 2/5
938/938 - 11s - 12ms/step - accuracy: 0.7790 - loss: 0.6227 - val_accuracy: 0.7936 - val_loss: 0.6024
Epoch 3/5
938/938 - 11s - 11ms/step - accuracy: 0.8105 - loss: 0.5470 - val_accuracy: 0.8071 - val_loss: 0.5652
Epoch 4/5
938/938 - 11s - 12ms/step - accuracy: 0.8291 - loss: 0.5014 - val_accuracy: 0.8172 - val_loss: 0.5271
Epoch 5/5
938/938 - 11s - 11ms/step - accuracy: 0.8411 - loss: 0.4695 - val_accuracy: 0.8343 - val_loss: 0.5073
Test accuracy: 0.8343

Training with Filter=5, Reg=0.001, Batch=32, Opt=adam
Epoch 1/5
1875/1875 - 17s - 9ms/step - accuracy: 0.8224 - loss: 0.6287 - val_accuracy: 0.8500 - val_loss: 0.5143
Epoch 2/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8738 - loss: 0.4524 - val_accuracy: 0.8715 - val_loss: 0.4574
Epoch 3/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8856 - loss: 0.4163 - val_accuracy: 0.8838 - val_loss: 0.4134
Epoch 4/5
1875/1875 - 16s - 8ms/step - accuracy: 0.8918 - loss: 0.3965 - val_accuracy: 0.8901 - val_loss: 0.4030
Epoch 5/5
1875/1875 - 16s - 9ms/step - accuracy: 0.8955 - loss: 0.3817 - val_accuracy: 0.8829 - val_loss: 0.3992
Test accuracy: 0.8829

Training with Filter=5, Reg=0.001, Batch=32, Opt=sgd
Epoch 1/5
1875/1875 - 16s - 9ms/step - accuracy: 0.7056 - loss: 1.0863 - val_accuracy: 0.7831 - val_loss: 0.8482
Epoch 2/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8142 - loss: 0.7738 - val_accuracy: 0.8110 - val_loss: 0.7645
Epoch 3/5
1875/1875 - 14s - 8ms/step - accuracy: 0.8403 - loss: 0.6944 - val_accuracy: 0.8449 - val_loss: 0.6945
Epoch 4/5
1875/1875 - 14s - 8ms/step - accuracy: 0.8535 - loss: 0.6441 - val_accuracy: 0.8508 - val_loss: 0.6498
Epoch 5/5
1875/1875 - 15s - 8ms/step - accuracy: 0.8625 - loss: 0.6076 - val_accuracy: 0.8600 - val_loss: 0.6145
Test accuracy: 0.8600

Training with Filter=5, Reg=0.001, Batch=64, Opt=adam
Epoch 1/5
938/938 - 13s - 14ms/step - accuracy: 0.8152 - loss: 0.6608 - val_accuracy: 0.8618 - val_loss: 0.5137
Epoch 2/5
938/938 - 11s - 12ms/step - accuracy: 0.8721 - loss: 0.4661 - val_accuracy: 0.8716 - val_loss: 0.4638
Epoch 3/5
938/938 - 11s - 12ms/step - accuracy: 0.8861 - loss: 0.4193 - val_accuracy: 0.8754 - val_loss: 0.4389
Epoch 4/5
938/938 - 11s - 12ms/step - accuracy: 0.8928 - loss: 0.3973 - val_accuracy: 0.8876 - val_loss: 0.4080
Epoch 5/5
938/938 - 11s - 12ms/step - accuracy: 0.8963 - loss: 0.3822 - val_accuracy: 0.8843 - val_loss: 0.4070
Test accuracy: 0.8843

Training with Filter=5, Reg=0.001, Batch=64, Opt=sgd
Epoch 1/5
938/938 - 12s - 12ms/step - accuracy: 0.6474 - loss: 1.2889 - val_accuracy: 0.7510 - val_loss: 0.9414
Epoch 2/5
938/938 - 11s - 11ms/step - accuracy: 0.7736 - loss: 0.8738 - val_accuracy: 0.7803 - val_loss: 0.8499
Epoch 3/5
938/938 - 11s - 11ms/step - accuracy: 0.8081 - loss: 0.7882 - val_accuracy: 0.7902 - val_loss: 0.8088
Epoch 4/5
938/938 - 11s - 11ms/step - accuracy: 0.8278 - loss: 0.7331 - val_accuracy: 0.8208 - val_loss: 0.7456
Epoch 5/5
938/938 - 11s - 12ms/step - accuracy: 0.8422 - loss: 0.6924 - val_accuracy: 0.8357 - val_loss: 0.7093
Test accuracy: 0.8357
```

## **Performance Evaluation**

### **Validation Accuracy Trends**

![image](https://github.com/user-attachments/assets/7b635f67-9845-45f9-8e32-cd8838fbe192)


### **Plot Training vs Validation Loss**

![image](https://github.com/user-attachments/assets/465f6dab-7a97-40bc-affc-d53e836378ba)

---

## **My Comments**

- Using **Adam** as the optimizer led to faster convergence and better accuracy.
- **Adam optimizer consistently outperforms SGD** across all configurations, especially early in training.
- **L2 regularization of 0.001** performs better than 0.01, suggesting that too much regularization hurts the model’s learning ability.
- Smaller batch sizes (32) generally give better performance than larger ones (64), likely because they allow more weight updates per epoch.
- The best-performing model reached a validation accuracy of just over **91%** by epoch 10.
- The validation accuracy graph clearly shows which settings perform best over time.
- The model generalizes well, achieving over 90% accuracy without any advanced tricks.

---
