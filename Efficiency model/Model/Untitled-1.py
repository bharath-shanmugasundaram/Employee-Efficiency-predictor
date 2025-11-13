import numpy as np
from module import X, Y
import torch as t
import torch.nn as nn
import torch.optim as optim
import sklearn.model_selection as sk

# ---- Train-test split ----
X_train, X_test, Y_train, Y_test = sk.train_test_split(X, Y, test_size=0.33, random_state=42)

# ---- Normalize using training stats ----
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# ---- Model ----
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)

model = NeuralNet(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---- Training ----
for epoch in range(2000):
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        with t.no_grad():
            y_class = (t.sigmoid(y_pred) >= 0.5).float()
            acc = (y_class.eq(Y_train).sum().item() / Y_train.shape[0]) * 100
        print(f"Epoch [{epoch+1}/2000] | Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")

# ---- Evaluation ----
with t.no_grad():
    y_pred_test = model(X_test)
    y_class_test = (t.sigmoid(y_pred_test) >= 0.5).float()
    acc_test = (y_class_test.eq(Y_test).sum().item() / Y_test.shape[0]) * 100
    print(f"\nTest Accuracy: {acc_test:.2f}%")

# ---- Save checkpoint (model + normalization stats) ----
t.save({
    "model_state": model.state_dict(),
    "mean": X_mean,
    "std": X_std
}, "model.pth")

print("Model saved successfully ✅")
