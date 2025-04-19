## Output:

```
Original shape: (145063, 551)
Epoch 1/500 - Loss: 1.1863
Epoch 10/500 - Loss: 1.1297
Epoch 20/500 - Loss: 1.1057
Epoch 30/500 - Loss: 1.0906
Epoch 40/500 - Loss: 1.0787
Epoch 50/500 - Loss: 1.0641
Epoch 60/500 - Loss: 1.0539
Epoch 70/500 - Loss: 1.0443
Epoch 80/500 - Loss: 1.0350
Epoch 90/500 - Loss: 1.0276
Epoch 100/500 - Loss: 1.0216
Epoch 110/500 - Loss: 1.0150
Epoch 120/500 - Loss: 1.0102
Epoch 130/500 - Loss: 1.0069
Epoch 140/500 - Loss: 1.0013
Epoch 150/500 - Loss: 0.9957
Epoch 160/500 - Loss: 0.9920
Epoch 170/500 - Loss: 0.9786
Epoch 180/500 - Loss: 0.9650
Epoch 190/500 - Loss: 0.9584
Epoch 200/500 - Loss: 0.9048
Epoch 210/500 - Loss: 0.8471
Epoch 220/500 - Loss: 0.9257
Epoch 230/500 - Loss: 0.8118
Epoch 240/500 - Loss: 0.7594
Epoch 250/500 - Loss: 0.7171
Epoch 260/500 - Loss: 0.6855
Epoch 270/500 - Loss: 0.6548
Epoch 280/500 - Loss: 0.6361
Epoch 290/500 - Loss: 0.6064
Epoch 300/500 - Loss: 0.5823
Epoch 310/500 - Loss: 0.7299
Epoch 320/500 - Loss: 1.0329
Epoch 330/500 - Loss: 1.0217
Epoch 340/500 - Loss: 1.0110
Epoch 350/500 - Loss: 1.0079
Epoch 360/500 - Loss: 1.0043
Epoch 370/500 - Loss: 1.0006
Epoch 380/500 - Loss: 0.9967
Epoch 390/500 - Loss: 0.9917
Epoch 400/500 - Loss: 0.9839
Epoch 410/500 - Loss: 0.9662
Epoch 420/500 - Loss: 0.6587
Epoch 430/500 - Loss: 0.5781
Epoch 440/500 - Loss: 0.5493
Epoch 450/500 - Loss: 0.5255
Epoch 460/500 - Loss: 0.5048
Epoch 470/500 - Loss: 0.4888
Epoch 480/500 - Loss: 0.4748
Epoch 490/500 - Loss: 0.4619
Epoch 500/500 - Loss: 0.4499

📊 Performance Metrics:
✅ RMSE = 0.7939
✅ MSE  = 0.6302
✅ MAE  = 0.4223
✅ R²   = -0.0757
```

## Plots and Observations

### Plot 1: Actual vs. Predicted (First 100 Points)
![image](https://github.com/user-attachments/assets/e52ac6e6-51b5-4d0f-865c-f79b3cca136d)
---

### Plot 2: Training Loss Curve
![image](https://github.com/user-attachments/assets/025b5964-7b19-4b55-b8ca-f04411a1b944)

---

## Comments:

- In Plot 1: Actual vs. Predicted (First 100 Points), the predicted line is very flat, showing little variation. This indicates underfitting, the model seems to act like a moving average, capturing long-term trends but missing out on sharp changes. This may be due to the shallow architecture or the model being biased by noise in the data.
- The mid-training spikes in the loss curve are unusual and could indicate problems with training setup or data handling.
- The model lacks the ability to react dynamically to changes, behaving more like a rolling mean than a time series predictor.
