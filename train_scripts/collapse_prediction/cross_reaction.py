from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
import sqlite3
import logging
import argparse
import re


logger = logging.getLogger(__name__)


def load_data(db, reaction, train):
    np.random.seed(2)
    sign = '=' if train else '!='
    query = f'''
        SELECT fingerprint, topology, collapsed
        FROM cages
        WHERE
            reaction {sign} "{reaction}" AND
            collapsed IS NOT NULL AND
            topology = "FourPlusSix"
    '''

    results = ((eval(fp), top, label) for
               fp, top, label in db.execute(query))
    fps, tops, labels = zip(*results)
    tops = LabelBinarizer().fit_transform(tops)
    fps = np.concatenate((fps, tops), axis=1)
    fps, labels = shuffle(fps, labels)
    return np.array(fps), np.array(labels)


def train(db, reaction, table, reverse):
    if table:
        logger.setLevel(logging.ERROR)

    logger.debug(f'{reaction} - reverse {reverse}')

    fp_train, labels_train = load_data(
                                 db=db,
                                 reaction=reaction,
                                 train=True if not reverse else False)
    fp_test, labels_test = load_data(
                                 db=db,
                                 reaction=reaction,
                                 train=False if not reverse else True)

    logger.debug(f'Fingerprint shape is {fp_train.shape}.')
    logger.debug(f'Collected train labels:\n{Counter(labels_train)}')
    logger.debug(f'Collected test labels:\n{Counter(labels_test)}')

    np.random.seed(4)
    clf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced')
    clf.fit(fp_train, labels_train)

    expected = labels_test
    predicted = clf.predict(fp_test)
    accuracy = metrics.accuracy_score(expected, predicted)
    p0, p1 = metrics.precision_score(expected, predicted, average=None)
    r0, r1 = metrics.recall_score(expected, predicted, average=None)

    if table:
        p = re.compile(r'([a-z_]*?\d)([a-z_]*?\d)')
        lk, bb = p.search(reaction).groups()
        nums = [accuracy, p0, r0, p1, r1]
        nums = [f'{n:.2f}' for n in nums]
        row = [bb.replace('_', ' '), lk.replace('_', ' '), *nums]
        print(' & '.join(row) + r' \\')
    else:
        print()
        print(f'accuracy: {accuracy:.2f}')
        print(f'precision (shape persistent): {p0:.2f}')
        print(f'recall (shape persistent): {r0:.2f}')
        print(f'precision (collapsed): {p1:.2f}')
        print(f'precision (collapsed): {r1:.2f}')
        print('\n\n')


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
    parser.add_argument('type', choices=['train', 'test'])
    parser.add_argument(
        'reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reaction to train on. Given by {reacts}.')
    parser.add_argument(
        '--table', action='store_true',
        help='Print out results in tex table format.')
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    for reaction in args.reactions:
        train(
              db=db,
              reaction=reacts[reaction],
              table=args.table,
              reverse=args.type == 'test')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
