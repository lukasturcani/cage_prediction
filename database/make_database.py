"""
Extracts cage properties from stk population into a SQL database.

"""

import argparse
import stk
import sqlite3
import rdkit.Chem.AllChem as rdkit
import multiprocessing as mp
import numpy as np
import itertools as it
from os.path import basename, splitext
import logging

__author__ = "Lukas Turcani"
logger = logging.getLogger(__name__)


def fingerprint(cage):
    radius = 8
    bits = 512
    full_fp = []
    for mol in cage.building_blocks:
        rdkit.GetSSSR(mol.mol)
        mol.mol.UpdatePropertyCache(strict=False)
        info = {}
        fp = rdkit.GetMorganFingerprintAsBitVect(mol.mol,
                                                 radius,
                                                 bits,
                                                 bitInfo=info)
        fp = list(fp)
        for bit, activators in info.items():
            fp[bit] = len(activators)
        full_fp.extend(fp)
    return str(full_fp)


def collapsed(cage, max_diameter, window_diff, cavity_size):
    md = max_diameter
    if window_diff is None:
        return True
    elif ((4*window_diff)/(md*cage.topology.n_windows) < 0.035 and
          cavity_size > 1):
        return False
    else:
        return None


def cavity_size(cage):
    cavity = -cage._cavity_size([0, 0, 0], -1)
    return cavity if cavity > 0 else 0


def window_difference(windows):
    clusters = [list(windows)]

    # After this sum the differences in each group and then
    # sum the group totals.
    diff_sums = []
    for cluster in clusters:
        diff_sum = sum(abs(w1 - w2) for w1, w2 in
                       it.combinations(cluster, 2))

        diff_num = sum(1 for _ in it.combinations(cluster, 2))

        diff_sums.append(diff_sum / diff_num)

    return np.mean(diff_sums)


def make_entry(cage):
    try:
        cage.set_position([0, 0, 0])
        cs = cavity_size(cage)
        md, *_ = cage.max_diameter()

        windows = cage.windows()
        if windows is None:
            wd = window_std = None
        else:
            w = sorted(windows, reverse=True)[:cage.topology.n_windows]
            wd = (window_difference(w) if
                  len(w) == cage.topology.n_windows else None)
            window_std = (np.std(w) if
                          len(w) == cage.topology.n_windows else None)

        return (cage.name,
                cage.note,
                cage.topology.__class__.__name__,
                fingerprint(cage),
                collapsed(cage, md, wd, cs),
                cs,
                md,
                str(windows),
                wd,
                window_std)
    except Exception as ex:
        logger.error('Error occured.', exc_info=True)


def make_database(databases):
    db = sqlite3.connect('cage_prediction.db')
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cages (
            name TEXT,
            reaction TEXT,
            topology TEXT,
            fingerprint TEXT,
            collapsed BOOLEAN,
            cavity_size FLOAT,
            max_diameter FLOAT,
            windows TEXT,
            window_diff FLOAT,
            window_std FLOAT
        )''')

    for db_path in databases:
        logger.info(f'Starting on database {db_path}.')

        dbname, _ = splitext(basename(db_path))
        with mp.Pool() as pool:
            pop = stk.Population.load(db_path, stk.Molecule.from_dict)
            for mol in pop:
                mol.note = dbname.replace('22', '2').replace('23', '3')
            cages = pool.map(make_entry, pop)
            cursor.executemany(
             'INSERT INTO cages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
             cages)
    db.commit()
    db.close()


def main():
    parser = argparse.ArgumentParser(
             description=('Creates a SQL database of the cages '
                          'used in the paper.'))
    parser.add_argument(
        'database_path',
        help=(
         'Path to stk population files downloaded from '
         ' https://doi.org/10.14469/hpc/4618.'),
        nargs='+')

    args = parser.parse_args()
    make_database(args.database_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    logger.info('Done.')
