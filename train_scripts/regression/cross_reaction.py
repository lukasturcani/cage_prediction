from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import numpy as np
import logging
import sqlite3
import argparse


logger = logging.getLogger(__name__)


def load_data(db, cage_property, reaction, train):
    np.random.seed(2)
    sign = '=' if train else '!='
    query = f'''
        SELECT fingerprint, topology, {cage_property}
        FROM cages
        WHERE
            reaction {sign} "{reaction}" AND
            collapsed = 0 AND
            topology = "FourPlusSix"
    '''

    results = ((eval(fp), top, cs) for
               fp, top, cs in db.execute(query))
    fps, tops, targets = zip(*results)
    tops = LabelBinarizer().fit_transform(tops)
    fps = np.concatenate((fps, tops), axis=1)
    fps, targets = shuffle(fps, targets)
    return np.array(fps), np.array(targets)


def train(db, cage_property, reaction, reverse):
    logger.debug(f'{reaction} - reverse {reverse}')

    fp_train, targets_train = load_data(
                                 db=db,
                                 cage_property=cage_property,
                                 reaction=reaction,
                                 train=True if not reverse else False)
    fp_test, targets_test = load_data(
                                 db=db,
                                 cage_property=cage_property,
                                 reaction=reaction,
                                 train=False if not reverse else True)

    logger.debug(f'Fingerprint shape is {fp_train.shape}.')
    logger.debug(f'Training set size is {targets_train.shape}.')
    logger.debug(f'Test set size is {targets_test.shape}.')

    np.random.seed(4)
    reg = RandomForestRegressor(
                                n_estimators=100,
                                n_jobs=-1,
                                criterion='mse')
    reg.fit(fp_train, targets_train)
    expected = targets_test
    predicted = reg.predict(fp_test)

    mse = mean_squared_error(expected, predicted)
    mae = mean_absolute_error(expected, predicted)
    r2 = r2_score(expected, predicted)
    return mse, mae, r2


def main():
    reacts = {
        1: 'amine2aldehyde3',
        2: 'aldehyde2amine3',
        3: 'alkene2alkene3',
        4: 'alkyne2alkyne3',
        5: 'amine2carboxylic_acid3',
        6: 'carboxylic_acid2amine3',
        7: 'thiol2thiol3'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('database_path')
    parser.add_argument(
        'cage_property',
        help='The cage property you want to do regression on.')
    parser.add_argument(
        'reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reaction to train on. Given by {reacts}.')
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    for reaction in args.reactions:
        mse, mae, r2 = train(
              db=db,
              cage_property=args.cage_property,
              reaction=reacts[reaction],
              reverse=False)
        print(f'mse - {mse:.2f}\nmae - {mae:.2f}\nr2 - {r2:.2f}')
        mse, mae, r2 = train(
              db=db,
              cage_property=args.cage_property,
              reaction=reacts[reaction],
              reverse=True)
        print(f'mse - {mse:.2f}\nmae - {mae:.2f}\nr2 - {r2:.2f}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
