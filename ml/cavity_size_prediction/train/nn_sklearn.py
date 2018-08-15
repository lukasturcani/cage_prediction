from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import (cross_val_score,
                                     train_test_split)
import numpy as np
import pymongo
import pickle


def get_fp(match, radius, bits, fp_tags):
    struct = next(s for s in match['structures'] if
                  s['calc_params'] == {'software': 'stk'})

    for fp in struct['fingerprints']:
        if (fp['radius'] == radius and
            fp['bits'] == bits and
           all(tag in fp['type'] for tag in fp_tags)):
            return fp['fp']

    raise RuntimeError('Fingerprint not found.')


def get_target(match):

    calc_params = {
                        'software': 'schrodinger2017-4',
                        'max_iter': 5000,
                        'md': {
                               'confs': 50,
                               'temp': 700,
                               'sim_time': 2000,
                               'time_step': 1.0,
                               'force_field': 16,
                               'max_iter': 2500,
                               'eq_time': 100,
                               'gradient': 0.05,
                               'timeout': None
                        },
                        'force_field': 16,
                        'restricted': 'both',
                        'timeout': None,
                        'gradient': 0.05
    }

    for struct in match['structures']:
        if struct['calc_params'] == calc_params:
            return struct['cavity_size']

    raise RuntimeError('Cavity size not found.')


def load_data(db, radius, bits, fp_tags, labeller, query):
    fingerprints, topologies, targets = [], [], []
    for match in db.find(query):
        target = get_target(match)
        targets.append(target)
        topologies.append(match['topology']['class'])
        fingerprints.append(get_fp(match, radius, bits, fp_tags))

    topologies = LabelBinarizer().fit_transform(topologies)
    fingerprints = np.concatenate((fingerprints, topologies), axis=1)
    fingerprints, targets = shuffle(fingerprints, targets)
    return np.array(fingerprints), np.array(targets)


def train(db, radius, bits, fp_tags, labeller, cv):
    """

    """

    query = {'tags': 'amine2aldehyde3', 'topology.class': 'FourPlusSix'}

    fingerprints, targets = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=query)
    print('dataset size:', len(targets))
    reg = MLPRegressor(hidden_layer_sizes=(1000, 100),
                       batch_size=32)
    scores = cross_val_score(reg, fingerprints, targets, cv=cv, n_jobs=-1)

    split = train_test_split(fingerprints,
                             targets,
                             test_size=1/cv)
    fp_train, fp_test, targets_train, targets_test = split

    reg.fit(fp_train, targets_train)

    with open('random_forest.pkl', 'wb') as f:
        pickle.dump(reg, f)

    expected = targets_test
    predicted = reg.predict(fp_test)
    mae = metrics.mean_absolute_error(expected, predicted)
    return scores.mean(), mae


def main():
    np.random.seed(42)
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    cv_score, mae = train(
                                          db=client.small.cages,
                                          radius=2,
                                          bits=512,
                                          fp_tags=['bb'],
                                          labeller='pywindow_plus',
                                          cv=5)
    print(cv_score, mae, sep='\n\n')


if __name__ == '__main__':
    main()
