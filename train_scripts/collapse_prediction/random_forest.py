from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import (make_scorer,
                             accuracy_score,
                             recall_score,
                             precision_score)
import numpy as np
from collections import Counter
import pickle
import sqlite3
import logging
import argparse
import re


logger = logging.getLogger(__name__)


def load_data(db, reactions, topologies):
    np.random.seed(2)
    query = '''
        SELECT fingerprint, topology, collapsed
        FROM cages
        WHERE
            reaction IN ({}) AND
            collapsed IS NOT NULL AND
            topology IN ({})
    '''.format(', '.join('?'*len(reactions)),
               ', '.join('?'*len(topologies)))

    results = ((eval(fp), top, label) for fp, top, label in
               db.execute(query, reactions+topologies))
    fps, tops, labels = zip(*results)
    tops = LabelBinarizer().fit_transform(tops)
    fps = np.concatenate((fps, tops), axis=1)
    fps, labels = shuffle(fps, labels)
    return np.array(fps), np.array(labels)


def train(db, cv, reactions, topologies, table, save):
    if table:
        logger.setLevel(logging.ERROR)

    logger.debug(f'Reactions: {reactions}.')
    logger.debug(f'Topologies: {topologies}.')
    fingerprints, labels = load_data(db=db,
                                     reactions=reactions,
                                     topologies=topologies)

    logger.debug(f'Fingerprint shape is {fingerprints.shape}.')
    logger.debug(f'Collected labels:\n{Counter(labels)}')

    clf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced')

    np.random.seed(4)
    scores = cross_validate(estimator=clf,
                            X=fingerprints,
                            y=labels,
                            scoring={
                             'accuracy': make_scorer(accuracy_score),
                             'precision_0': make_scorer(precision_score,
                                                        pos_label=0,
                                                        labels=[0]),
                             'recall_0': make_scorer(recall_score,
                                                     pos_label=0,
                                                     labels=[0]),
                             'precision_1': make_scorer(precision_score,
                                                        pos_label=1,
                                                        labels=[1]),
                             'recall_1': make_scorer(recall_score,
                                                     pos_label=1,
                                                     labels=[1])
                            },
                            cv=cv,
                            n_jobs=-1)

    accuracy = scores['test_accuracy'].mean()
    p0 = scores['test_precision_0'].mean()
    r0 = scores['test_recall_0'].mean()
    p1 = scores['test_precision_1'].mean()
    r1 = scores['test_recall_1'].mean()
    if table:
        p = re.compile(r'([a-z_]*?\d)([a-z_]*?\d)')
        assert len(reactions) == 1
        lk, bb = p.search(reactions[0]).groups()
        nums = [accuracy, p0, r0, p1, r1]
        nums = [f'{n:.2f}' for n in nums]
        row = [bb.replace('_', ' '), lk.replace('_', ' '), *nums]
        print(' & '.join(row) + r' \\')
    else:
        print('accuracy',
              f'{accuracy:.2f}',
              sep='\n', end='\n\n')
        print('precision (shape persistent)',
              f'{p0:.2f}',
              sep='\n', end='\n\n')
        print('recall (shape persistent)',
              f'{r0:.2f}',
              sep='\n', end='\n\n')
        print('precision (collapsed)',
              f'{p1:.2f}',
              sep='\n', end='\n\n')
        print('recall (collapsed)',
              f'{r1:.2f}',
              sep='\n', end='\n\n')
        print('\n')

    if save:
        filename = '_'.join(reactions) + '.pkl'
        with open(filename, 'wb') as f:
            clf.fit(fingerprints, labels)
            pickle.dump(clf, f)


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
        '--table', action='store_true',
        help='Print out results in tex table format.')
    parser.add_argument(
        '-s', '--save', action='store_true',
        help='Toggles to save each trained model.')

    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)

    if args.join:
        train(db=db,
              cv=5,
              reactions=[reacts[i] for i in args.reactions],
              topologies=[tops[i] for i in args.topologies],
              table=args.table,
              save=args.save)
    else:
        for react in args.reactions:
            train(db=db,
                  cv=5,
                  reactions=[reacts[react]],
                  topologies=[tops[i] for i in args.topologies],
                  table=args.table,
                  save=args.save)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
