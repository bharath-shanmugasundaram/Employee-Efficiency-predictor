from flask import Flask, request, jsonify, render_template
import torch as t
import torch.nn as nn

# ---- Model (same as training) ----
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

app = Flask(__name__)

# ---- Load checkpoint ----
checkpoint = t.load("model.pth", map_location="cpu")
input_size = 8
model = NeuralNet(input_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()
X_mean, X_std = checkpoint["mean"], checkpoint["std"]

# ---- Routes ----
@app.route("/")
def home():
    return render_template("index.html", input_size=input_size)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", None)
    if features is None or len(features) != input_size:
        return jsonify({"error": f"Expected {input_size} features"}), 400

    x = t.tensor([features], dtype=t.float32)
    x = (x - X_mean) / X_std

    with t.no_grad():
        output = model(x)
        prob = t.sigmoid(output).item()
        pred_class = int(prob >= 0.5)

    return jsonify({"probability": prob, "prediction": pred_class})

if __name__ == "__main__":
    app.run(debug=True)
