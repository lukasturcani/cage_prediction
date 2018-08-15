from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import (cross_validate,
                                     cross_val_predict)
from sklearn.metrics import (make_scorer,
                             mean_squared_error,
                             mean_absolute_error,
                             r2_score)
import numpy as np
import pymongo
from pprint import pprint


def save_r2_data(reg, fingerprints, targets, cv):
    np.random.seed(2)
    y_predict = cross_val_predict(reg,
                                  fingerprints,
                                  targets,
                                  cv=cv,
                                  n_jobs=-1)

    targets.dump('y_true.np')
    y_predict.dump('y_pred.np')


def get_fp(match, radius, bits, fp_tags):
    struct = next(s for s in match['structures'] if
                  s['calc_params'] == {'software': 'stk'})

    return next(fp['fp'] for fp in struct['fingerprints'] if
                fp['radius'] == radius and fp['bits'] == bits and
                all(tag in fp['type'] for tag in fp_tags))


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

    return next(struct['cavity_size'] for
                struct in match['structures'] if
                struct['calc_params'] == calc_params)


def load_data(db, radius, bits, fp_tags, labeller, query):
    fingerprints, topologies, targets = [], [], []
    c = db.find(query)
    pprint(c.explain())
    for match in c:
        targets.append(get_target(match))
        topologies.append(match['topology']['class'])
        fingerprints.append(get_fp(match, radius, bits, fp_tags))

    topologies = LabelBinarizer().fit_transform(topologies)
    fingerprints = np.concatenate((fingerprints, topologies), axis=1)
    fingerprints, targets = shuffle(fingerprints, targets)
    return np.array(fingerprints), np.array(targets)


def nmae(estimator, X, y):
    y_pred = estimator.predict(X)
    return -np.mean(np.absolute(y-y_pred) / y)


def train(db, radius, bits, fp_tags, labeller, cv, fg_name):
    """

    """
    print(fg_name)

    # query = {'tags': {'$in': ['amine4aldehyde3',
    #                           'amine4aldehyde2',
    #                           'amine3aldehyde3',
    #                           'amine2aldehyde3',
    #                           'aldehyde4amine3',
    #                           'aldehyde4amine2',
    #                           'aldehyde2amine3']}}

    query = {'tags': fg_name, 'topology.class': 'FourPlusSix'}

    fingerprints, targets = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=query)
    print('dataset size:', len(targets))
    reg = RandomForestRegressor(n_estimators=100,
                                n_jobs=-1,
                                criterion='mse')

    np.random.seed(2)
    scores = cross_validate(estimator=reg,
                            X=fingerprints,
                            y=targets,
                            scoring={
                             'mse': make_scorer(mean_squared_error,
                                                greater_is_better=False),
                             'mae': make_scorer(mean_absolute_error,
                                                greater_is_better=False),
                             'nmae': nmae,
                             'r2': make_scorer(r2_score)},
                            cv=cv,
                            n_jobs=-1)
    print('mse', -scores['test_mse'].mean())
    print('mae', -scores['test_mae'].mean())
    print('nmae', -scores['test_nmae'].mean())
    print('r2', scores['test_r2'].mean())

    save_r2_data(reg, fingerprints, targets, cv)


def main():
    np.random.seed(2)
    client = pymongo.MongoClient('mongodb://localhost:27017/')

    fg_names = ['amine2aldehyde3',
                'aldehyde2amine3',
                'alkene2alkene3',
                'alkyne22alkyne23',
                # 'thiol2thiol3',
                'amine2carboxylic_acid3',
                'carboxylic_acid2amine3']
    # fg_names = ['amine2aldehyde3']
    for fg_name in fg_names:
        train(
              db=client.small.cages,
              radius=8,
              bits=512,
              fp_tags=['bb_count'],
              labeller='pywindow_plus',
              cv=5,
              fg_name=fg_name)


if __name__ == '__main__':
    main()
