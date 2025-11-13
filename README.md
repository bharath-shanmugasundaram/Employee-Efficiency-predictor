# 🧠 Neural Network Classifier — PyTorch + Flask Deployment

This project demonstrates how to **train a neural network using PyTorch** and **deploy it using Flask** for real-time predictions. It includes **data preprocessing, model training, saving, and an API-based prediction service**.

---

## 📘 Overview

- Built using **PyTorch** for training a 3-layer neural network.
- Deployed with a **Flask web app** that serves predictions through a REST API.
- Model predicts a **binary classification output (0 or 1)** based on input features.
- Includes **data normalization** and **checkpoint saving** for consistent inference.

---

## 🧩 Architecture

The model is a simple Fully Connected Neural Network (FCNN):

| Layer | Type | Neurons | Activation |
| :---: | :---: | :---: | :---: |
| **1** | Linear | Input → 16 | ReLU |
| **2** | Linear | 16 → 8 | ReLU |
| **3** | Linear | 8 → 1 | Sigmoid (applied during inference) |

---

## 📂 Project Structure

NeuralNet-Flask/ │ ├── module.py # Contains X, Y data (preprocessed or loaded) ├── train.py # Training script (PyTorch) ├── app.py # Flask backend serving predictions ├── model.pth # Saved trained model and normalization stats ├── templates/ │ └── index.html # Web interface for input (optional) └── README.md # Project documentation

---

## ⚙️ Training Script — `train.py`

The script trains a binary classifier neural network using `sklearn` for data splitting and `PyTorch` for modeling.

### Key Steps:

1.  **Load and Split Data**
    ```python
    X, xtest, Y, ytrain = sk.train_test_split(X, Y, test_size=0.33, random_state=42)
    ```
2.  **Normalize Inputs**
    The mean and standard deviation are calculated on the training set and saved.
    ```python
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std
    ```
3.  **Model Definition**
    The `nn.Sequential` block defines the network structure:
    ```python
    class NeuralNet(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
    ```
4.  **Training Loop**
    Uses `BCEWithLogitsLoss` and an optimizer (e.g., Adam or SGD).
    ```python
    for i in range(2000):
        y_pred = model(X)
        loss = criterion(y_pred, Y)
        # ... backpropagation steps ...
    ```
5.  **Save Model and Normalization Stats**
    The checkpoint includes the model state and the normalization constants necessary for consistent deployment.
    ```python
    t.save({
        "model_state": model.state_dict(),
        "mean": X_mean,
        "std": X_std
    }, "model.pth")
    ```

---

## 🚀 Flask Backend — `app.py`

The Flask app loads the trained PyTorch model and provides a **REST API** endpoint for predictions.

### Endpoints

| Endpoint | Method | Description |
| :---: | :---: | :--- |
| `/` | `GET` | Returns the web interface (`index.html`). |
| `/predict` | `POST` | Accepts JSON input containing feature values and returns prediction and probability. |

### Example Request:

```json
{
  "features": [0.23, -0.17, 0.56, 1.22, -0.85, 0.77, 0.14, 0.66]
}
