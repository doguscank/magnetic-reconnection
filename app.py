from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import numpy as np
import io
import torch
import torch.nn as nn

app = Flask(__name__)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        prediction_type = request.form["prediction_type"]
        inputs = request.form.getlist("input_field")
        vectors = [[float(j) for j in i.split(",")] for i in inputs]


        if prediction_type == "classification":
            vectors = np.array(vectors).reshape(1, 30, 7)

            model = LSTMRegressor(
                input_size=7, hidden_size=50, num_layers=2, output_size=2
            )
            model.load_state_dict(torch.load("weights/single_cls_20_epoch_77_42.pth"))
            model.eval()

            vectors = torch.tensor(vectors, dtype=torch.float)

            pred = model(vectors)
            pred = torch.softmax(pred, dim=1)
            return f"Result: {pred.detach().numpy()}"
        elif prediction_type == "forecasting":
            vectors = np.array(vectors).reshape(1, 24, 10)

            model = LSTMRegressor(
                input_size=10, hidden_size=50, num_layers=2, output_size=10
            )
            model.load_state_dict(torch.load("weights/30_epoch_forecast.pth"))
            model.eval()

            vectors = torch.tensor(vectors, dtype=torch.float)

            pred = model(vectors)
            return f"Result: {pred.detach().numpy()}"

    return render_template("prediction.html")


@app.route("/education")
def education():
    return render_template("education.html")


if __name__ == "__main__":
    app.run(debug=True)
