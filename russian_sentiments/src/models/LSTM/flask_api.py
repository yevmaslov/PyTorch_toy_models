import flask
import time
import torch
import torch.nn as nn
from flask import Flask
from flask import request
from model import RNN
from inference import load_vocab_model, preprocess


app = Flask(__name__)

DEVICE = "cpu"
STOI, MODEL = load_vocab_model()

MODEL.load_state_dict(torch.load(
 'lstm_model.pt', map_location=torch.device(DEVICE)
 ))
MODEL.to(DEVICE)
MODEL.eval()


def sentence_prediction(sentence):
    sentence_prep, length = preprocess(sentence, STOI)
    prediction = torch.sigmoid(MODEL(sentence_prep, length))

    label = 'not toxic'
    if prediction > 0.5:
        label = 'toxic'

    return label


@app.route("/predict", methods=["GET"])
def predict():
    sentence = request.args.get("sentence")
    label = sentence_prediction(sentence)
    response = {"label": str(label)}
    return flask.jsonify(response)


if __name__ == '__main__':
    app.run()
