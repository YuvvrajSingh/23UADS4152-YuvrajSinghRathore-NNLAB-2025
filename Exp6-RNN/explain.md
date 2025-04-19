# Experiment 6: WAP to train and evaluate a Recurrent Neural Network using PyTorch Library to predict the next value in a sample time series dataset.

---

## Model Description

We used a simple RNN for sequence-to-one regression to predict the next day's traffic based on prior data.

- **Input size**: 1 (daily views)
- **Hidden size**: 64
- **Layers**: 1 RNN layer + 1 Linear output layer

The loss function was Mean Squared Error (MSE), and the optimizer was Adam.

---

## Code Description

1. **Preprocessing**:

   - Selected 1000 rows for simplicity
   - Normalized the data
   - Created 30-day sequences to predict the next day's views

2. **Model Definition**:

   - Defined a simple RNN model with one hidden layer and a linear output layer

3. **Training**:

   - Trained for 500 epochs
   - Recorded training loss at each epoch

4. **Evaluation**:
   - Evaluated using MSE, RMSE, MAE, and R²
   - Plotted training loss and actual vs. predicted values

---

## Performance Evaluation

### Metrics:

| Metric | Value   |
| ------ | ------- |
| MSE    | 0.6302  |
| RMSE   | 0.7939  |
| MAE    | 0.4223  |
| R²     | -0.0757 |

The negative R² suggests that the model underperforms compared to predicting the mean value. This is likely due to the noisy, volatile nature of web traffic data.

---
