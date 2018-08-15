"""
Goes through labels in database and writes the files in to dirs.

"""

import pymongo
import os
import stk
import rdkit.Chem.AllChem as rdkit


def Cage(match, calc_params):
    match = dict(match)
    match['topology'] = dict(match['topology'])
    mol_block = next(s['structure'] for s in match['structures']
                     if s['calc_params'] == calc_params)
    cage = stk.Cage.__new__(stk.Cage)
    cage.mol = rdkit.MolFromMolBlock(mol_block, sanitize=False)

    match['topology'].pop('react_del', None)
    Top = getattr(stk, match['topology'].pop('class'))
    cage.topology = Top(**match['topology'])

    return cage


def pywindow_plus(match, calc_params):
    struct = next(s for s in match['structures'] if
                  s['calc_params'] == calc_params)
    c = Cage(match, calc_params)

    wd = struct['window_difference']
    md = struct['max_diameter']
    cs = struct['cavity_size']

    if wd is None:
        return 1
    elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
        return 0
    else:
        return 2


def main():

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

    query = {'tags': 'aldehyde2amine3', 'topology.class': 'EightPlusTwelve'}
    db = pymongo.MongoClient('mongodb://localhost:27017').small.cages
    os.mkdir('label_check')
    os.mkdir('label_check/collapsed')
    os.mkdir('label_check/not_collapsed')
    os.mkdir('label_check/neither')
    for i, match in enumerate(db.find(query)):
        struct = next(s for s in match['structures'] if
                      s['calc_params'] == calc_params)
        collapsed = pywindow_plus(match, calc_params)
        if collapsed == 1:
            c = 'collapsed'
        elif collapsed == 2:
            c = 'neither'
        elif collapsed == 0:
            c = 'not_collapsed'
        with open(f'label_check/{c}/{i}.sdf', 'w') as f:
            f.write(struct['structure'])


if __name__ == '__main__':
    main()
