import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('/content/train_1.csv')
print("Original shape:", df.shape)

df = df.fillna(0)

raw_series = df.drop(columns=['Page']).values


series = raw_series[0] 
series = (series - np.mean(series)) / np.std(series) 


def create_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = create_sequences(series, SEQ_LEN)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)


class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)        
        out = out[:, -1, :]          
        return self.fc(out).squeeze()


model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 500

losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())  # Save loss
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")



model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()
    actual = y_test.numpy()

mse = mean_squared_error(actual, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)

print("\n📊 Performance Metrics:")
print(f"✅ MSE  = {mse:.4f}")
print(f"✅ RMSE = {rmse:.4f}")
print(f"✅ MAE  = {mae:.4f}")
print(f"✅ R²   = {r2:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(actual[:100], label='Actual')
plt.plot(preds[:100], label='Predicted')
plt.title('Next-Day Prediction on Web Traffic Time Series')
plt.xlabel('Day')
plt.ylabel('Normalized Views')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("📉 Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
plt.plot(targets[:100], label="Actual", marker='o')
plt.plot(predictions[:100], label="Predicted", marker='x')
plt.title("🔮 Predicted vs. Actual (First 100 points)")
plt.xlabel("Sample Index")
plt.ylabel("Views (Normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
