from flask import Flask, current_app, request, url_for, redirect
import os
from glob import iglob
import pickle
import rdkit.Chem.AllChem as rdkit


def bb_count_fingerprint(mols, bits, radius):
    """

    """

    full_fp = []
    for mol in mols:
        rdkit.GetSSSR(mol)
        mol.UpdatePropertyCache(strict=False)
        info = {}
        fp = rdkit.GetMorganFingerprintAsBitVect(mol,
                                                 radius,
                                                 bits,
                                                 bitInfo=info)
        fp = list(fp)
        for bit, activators in info.items():
            fp[bit] = len(activators)
        full_fp.extend(fp)
    full_fp.append(1)
    return full_fp


def fingerprint():
    bb = rdkit.AddHs(rdkit.MolFromSmiles(request.form['bb']))
    lk = rdkit.AddHs(rdkit.MolFromSmiles(request.form['lk']))
    return [bb_count_fingerprint((bb, lk), 512, 8)]


def predict(model_name):
    ans = current_app.models[model_name].predict(fingerprint())
    return str(ans[0])


def load_models():
    models = {}
    for model_path in iglob('backend/models/*.pkl'):
        name, _ = os.path.splitext(os.path.basename(model_path))
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
    return models


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    if not os.path.exists(app.instance_path):
        os.mkdir(app.instance_path)

    app.models = load_models()

    @app.route('/')
    def root():
        return redirect(url_for('static', filename='index.html'))

    app.route('/predict/<model_name>', methods=['POST'])(predict)

    return app
