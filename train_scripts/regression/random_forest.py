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
import logging
import argparse
import sqlite3


logger = logging.getLogger(__name__)


def save_r2_data(
                 cage_property,
                 reactions,
                 topologies,
                 reg,
                 fingerprints,
                 targets,
                 cv):
    np.random.seed(4)
    y_predict = cross_val_predict(reg,
                                  fingerprints,
                                  targets,
                                  cv=cv,
                                  n_jobs=-1)
    r = '_'.join(reactions)
    t = '_'.join(topologies)
    targets.dump(f'r{cage_property}_{r}_t_{t}_y_true.np')
    y_predict.dump(f'r{cage_property}_{r}_t_{t}_y_pred.np')


def load_data(db, cage_property, reactions, topologies):
    np.random.seed(2)
    query = '''
        SELECT fingerprint, topology, {}
        FROM cages
        WHERE
            reaction IN ({}) AND
            collapsed = 0 AND
            topology IN ({})
    '''.format(cage_property,
               ', '.join('?'*len(reactions)),
               ', '.join('?'*len(topologies)))

    results = ((eval(fp), top, cs) for fp, top, cs in
               db.execute(query, reactions+topologies))
    fps, tops, targets = zip(*results)
    tops = LabelBinarizer().fit_transform(tops)
    fps = np.concatenate((fps, tops), axis=1)
    fps, targets = shuffle(fps, targets)
    return np.array(fps), np.array(targets)


def train(db, cv, cage_property, reactions, topologies, save):
    logger.debug(f'Reactions: {reactions}.')
    logger.debug(f'Topologies: {topologies}.')
    fingerprints, targets = load_data(db=db,
                                      cage_property=cage_property,
                                      reactions=reactions,
                                      topologies=topologies)
    logger.debug(f'Fingerprint shape is {fingerprints.shape}.')
    logger.debug(f'Dataset size is {len(targets)}.')
    reg = RandomForestRegressor(n_estimators=100,
                                n_jobs=-1,
                                criterion='mse')

    np.random.seed(4)
    scores = cross_validate(estimator=reg,
                            X=fingerprints,
                            y=targets,
                            scoring={
                             'mse': make_scorer(mean_squared_error,
                                                greater_is_better=False),
                             'mae': make_scorer(mean_absolute_error,
                                                greater_is_better=False),
                             'r2': make_scorer(r2_score)},
                            cv=cv,
                            n_jobs=-1)
    if save:
        save_r2_data(
                     cage_property=cage_property,
                     reactions=reactions,
                     topologies=topologies,
                     reg=reg,
                     fingerprints=fingerprints,
                     targets=targets,
                     cv=cv)
    mse = -scores['test_mse'].mean()
    mae = -scores['test_mae'].mean()
    r2 = scores['test_r2'].mean()
    return mse, mae, r2


def main():
    reacts = {
        1: 'amine2aldehyde3',
        2: 'aldehyde2amine3',
        3: 'alkene2alkene3',
        4: 'alkyne2alkyne3',
        5: 'amine2carboxylic_acid3',
        6: 'carboxylic_acid2amine3',
        7: 'thiol2thiol3',
        8: 'amine4aldehyde3',
        9: 'amine4aldehyde2',
        10: 'amine3aldehyde3',
        11: 'aldehyde4amine3',
        12: 'aldehyde4amine2'
    }

    tops = {
        1: 'FourPlusSix',
        2: 'EightPlusTwelve',
        3: 'SixPlusTwelve',
        4: 'SixPlusEight',
        5: 'FourPlusFour'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('database_path')
    parser.add_argument(
        'cage_property',
        help='The cage property you want to do regression on.')
    parser.add_argument(
        '-r', '--reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reactions to train on. Given by {reacts}.')
    parser.add_argument(
        '-t', '--topologies',
        type=int, nargs='+', metavar='TOPOLOGY',
        help=f'Topologies to train on. Given by {tops}.')
    parser.add_argument(
        '--join', action='store_true',
        help=('Toggles if all reactions should be used to train '
              'one model or many.'))
    parser.add_argument(
        '-s', '--save', action='store_true',
        help='Toggles to save each models r2 results.')

    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)

    if args.join:
        mse, mae, r2 = train(
              db=db,
              cv=5,
              cage_property=args.cage_property,
              reactions=[reacts[i] for i in args.reactions],
              topologies=[tops[i] for i in args.topologies],
              save=args.save)
        print(f'mse - {mse:.2f}\nmae - {mae:.2f}\nr2 - {r2:.2f}')
    else:
        for react in args.reactions:
            mse, mae, r2 = train(
                  db=db,
                  cv=5,
                  cage_property=args.cage_property,
                  reactions=[reacts[react]],
                  topologies=[tops[i] for i in args.topologies],
                  save=args.save)
            print(f'mse - {mse:.2f}\nmae - {mae:.2f}\nr2 - {r2:.2f}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
