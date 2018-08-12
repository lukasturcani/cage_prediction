from flask import Flask
import os
from glob import iglob
import pickle


def predict(model_name):
    ...


def load_models():



def create_app():
    app = Flask(__name__, instance_relative_config=True)

    if not os.path.exists(app.instance_path):
        os.mkdir(app.instance_path)

    app.route('/predict/<model_name>', methods=['POST'])(predict)

    return app
