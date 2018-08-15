"""
Runs pywindow on a bunch of cages and saves the ones it fails on.

"""

import os
import stk
from glob import iglob
import multiprocessing as mp
import json
from functools import partial


def func(c):
    wd = c.window_difference()
    md = c.max_diameter()[0]
    c.set_position([0, 0, 0])
    cs = -c._cavity_size([0, 0, 0], -1)

    if wd is None:
        c.write(f'classifier_check_output/collapsed/{c.name}.mol')
    elif wd/md < 0.035 and cs > 1:
        c.write(f'classifier_check_output/not_collapsed/{c.name}.mol')
    else:
        c.write(f'classifier_check_output/neither/{c.name}.mol')
    return c, wd, md, cs


def func2(c, d):
    wd, cs, md = d[c.name]
    if wd is None:
        c.write(f'classifier_check_output/collapsed/{c.name}.mol')
    elif wd/md < 0.035 and cs > 1:
        c.write(f'classifier_check_output/not_collapsed/{c.name}.mol')
    else:
        c.write(f'classifier_check_output/neither/{c.name}.mol')
    return c, wd, md, cs


def load():
    with open('classifier_check_output/results.json', 'r') as f:
        return json.load(f)


def write(r):
    d = {c.name: (wd, md, cs) for c, wd, md, cs in r}
    with open('classifier_check_output/results.json', 'w') as f:
        json.dump(d, f)


def main():
    with mp.Pool() as pool:
        os.mkdir('classifier_check_output')
        os.mkdir('classifier_check_output/collapsed')
        os.mkdir('classifier_check_output/not_collapsed')
        os.mkdir('classifier_check_output/neither')

        databases = iglob(('/home/lukas/databases/liverpool_refined'
                           '/amines2aldehydes3_[1-4]_new.json'))
        p = stk.Population()
        for db in databases:
            p.add_members(stk.Population.load(db, stk.Molecule.from_dict))

        # Use this go genearte data file
        write(pool.map(func, p))
        # Use this when data file already made.
        # pool.map(partial(func, d=load()), p)


if __name__ == '__main__':
    main()
