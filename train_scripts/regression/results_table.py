"""
Makes a tex table of results suitable for pasting into the paper.

"""

import logging
from random_forest import train
from rf_cr import train as cr_train
import argparse
import sqlite3
import re

logger = logging.getLogger(__name__)


def main():
    reacts = [
        'amine2aldehyde3',
        'aldehyde2amine3',
        'alkene2alkene3',
        'alkyne2alkyne3',
        'amine2carboxylic_acid3',
        'carboxylic_acid2amine3'
        # 'thiol2thiol3'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('database_path')
    parser.add_argument('cage_property')
    args = parser.parse_args()
    db = sqlite3.connect(args.database_path)

    p = re.compile(r'([a-z_]*?\d)([a-z_]*?\d)')
    for react in reacts:
        lk, bb = p.search(react).groups()
        _, mae, r2 = train(db=db,
                           cv=5,
                           cage_property=args.cage_property,
                           reactions=[react],
                           topologies=['FourPlusSix'],
                           save=False)
        _, train_mae, train_r2 = cr_train(
                                    db=db,
                                    cage_property=args.cage_property,
                                    reaction=react,
                                    reverse=True)
        _, test_mae, test_r2 = cr_train(
                                    db=db,
                                    cage_property=args.cage_property,
                                    reaction=react,
                                    reverse=False)
        nums = [mae, r2, train_mae, train_r2, test_mae, test_r2]
        nums = [f'{n:.2f}' for n in nums]
        row = [bb.replace('_', ' '), lk.replace('_', ' '), *nums]
        print(' & '.join(row) + r' \\')


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
