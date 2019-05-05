import stk
from glob import iglob
import argparse
from os.path import join


def main():
    db = 'structs'
    diol_db = ''
    bbs = {'amine/2': iglob(join(db, 'amine2', '*')),
           'amine2/2': iglob(join(db, 'amine2', '*')),
           'aldehyde/2': iglob(join(db, 'aldehyde2', '*')),
           'terminal_alkene/2': iglob(join(db, 'terminal_alkene2', '*')),
           'alkyne/2': iglob(join(db, 'alkyne2', '*')),
           'alkyne2/2': iglob(join(db, 'alkyne2', '*')),
           'boronic_acid/2': iglob(join(db, 'boronic_acid2', '*')),
           'carboxylic_acid/2': iglob(join(db, 'carboxylic_acid2', '*')),
           'diol/2': iglob(join(diol_db, 'diol2', '*')),
           'thiol/2': iglob(join(db, 'thiol2', '*')),
           'amine/3': iglob(join(db, 'amine3', '*')),
           'amine2/3': iglob(join(db, 'amine3', '*')),
           'aldehyde/3': iglob(join(db, 'aldehyde3', '*')),
           'terminal_alkene/3': iglob(join(db, 'terminal_alkene3', '*')),
           'alkyne/3': iglob(join(db, 'alkyne3', '*')),
           'alkyne2/3': iglob(join(db, 'alkyne3', '*')),
           'boronic_acid/3': iglob(join(db, 'boronic_acid3', '*')),
           'carboxylic_acid/3': iglob(join(db, 'carboxylic_acid3', '*')),
           'diol/3': iglob(join(diol_db, 'diol3', '*')),
           'thiol/3': iglob(join(db, 'thiol3', '*')),
           'amine/4': iglob(join(db, 'amine4', '*')),
           'amine2/4': iglob(join(db, 'amine4', '*')),
           'aldehyde/4': iglob(join(db, 'aldehyde4', '*')),
           'terminal_alkene/4': iglob(join(db, 'terminal_alkene4', '*')),
           'alkyne/4': iglob(join(db, 'alkyne', '*')),
           'alkyne2/4': iglob(join(db, 'alkyne', '*')),
           'boronic_acid/4': iglob(join(db, 'boronic_acid4', '*')),
           'carboxylic_acid/4': iglob(join(db, 'carboxylic_acid4', '*')),
           'diol/4': iglob(join(diol_db, 'diol4', '*')),
           'thiol/4': iglob(join(db, 'thiol4', '*'))}

    bb_positions = {
        0: [0, 3, 5, 6],
        1: [1, 2, 4, 7]
    }
    # labelling is ('fgname/nfgs', ...)
    reactions = {1: ('amine/2', 'aldehyde/3', [stk.FourPlusSix(), stk.EightPlusTwelve()]),
                 2: ('aldehyde/2', 'amine/3', [stk.FourPlusSix(), stk.EightPlusTwelve()]),
                 3: ('terminal_alkene/2', 'terminal_alkene/3', [stk.FourPlusSix()]),
                 4: ('alkyne/2', 'alkyne/3', [stk.FourPlusSix()]),
                 5: ('alkyne2/2', 'alkyne2/3', [stk.FourPlusSix()]),
                 6: ('carboxylic_acid/2', 'amine2/3', [stk.FourPlusSix()]),
                 7: ('amine2/2', 'carboxylic_acid/3', [stk.FourPlusSix()]),
                 8: ('thiol/2', 'thiol/3', [stk.FourPlusSix()]),
                 9: ('boronic_acid/2', 'diol/3', [stk.FourPlusSix()]),
                 10: ('diol/2', 'boronic_acid/3', [stk.FourPlusSix()]),

                 11: ('amine/3', 'aldehyde/3', [stk.FourPlusFour(bb_positions=bb_positions)]),
                 12: ('amine2/3', 'carboxylic_acid/3', [stk.FourPlusFour(bb_positions=bb_positions)]),
                 13: ('terminal_alkene/3', 'terminal_alkene/3', [stk.FourPlusFour(bb_positions=bb_positions)]),
                 14: ('alkyne/3', 'alkyne/3', [stk.FourPlusFour(bb_positions=bb_positions)]),
                 15: ('alkyne2/3', 'alkyne2/3', [stk.FourPlusFour(bb_positions=bb_positions)]),
                 16: ('boronic_acid/3', 'diol/3', [stk.FourPlusFour(bb_positions=bb_positions)]),
                 17: ('thiol/3', 'thiol/3', [stk.FourPlusFour(bb_positions=bb_positions)]),

                 18: ('amine/4', 'aldehyde/2', [stk.SixPlusTwelve()]),
                 19: ('aldehyde/4', 'amine/2', [stk.SixPlusTwelve()]),
                 20: ('terminal_alkene/4', 'terminal_alkene/2', [stk.SixPlusTwelve()]),
                 21: ('alkyne/4', 'alkyne/2', [stk.SixPlusTwelve()]),
                 22: ('alkyne2/4', 'alkyne2/2', [stk.SixPlusTwelve()]),
                 23: ('thiol/4', 'thiol/2', [stk.SixPlusTwelve()]),
                 24: ('amine2/4', 'carboxylic_acid/2', [stk.SixPlusTwelve()]),
                 25: ('carboxylic_acid/4', 'amine2/2', [stk.SixPlusTwelve()]),

                 26: ('amine/4', 'aldehyde/3', [stk.SixPlusEight()]),
                 27: ('aldehyde/4', 'amine/3', [stk.SixPlusEight()]),
                 28: ('terminal_alkene/4', 'terminal_alkene/3', [stk.SixPlusEight()]),
                 29: ('alkyne/4', 'alkyne/3', [stk.SixPlusEight()]),
                 30: ('alkyne2/4', 'alkyne2/3', [stk.SixPlusEight()]),
                 31: ('amine2/4', 'carboxylic_acid/3', [stk.SixPlusEight()]),
                 32: ('carboxylic_acid/4', 'amine2/3', [stk.SixPlusEight()]),
                 33: ('thiol/4', 'thiol/3', [stk.SixPlusEight()]),
                 }

    bb_cls = {'2': stk.StructUnit2,
              '3': stk.StructUnit3,
              '4': stk.StructUnit3}

    parser = argparse.ArgumentParser()
    parser.add_argument('reaction', type=int, nargs='+')

    args = parser.parse_args()
    bb_mols = {}

    # For each selected reaction. Check which databases it involves.
    # For each involved database, load all the StructUnit instances
    for r in args.reaction:
        *dbs, topologies = reactions[r]
        for db in dbs:
            if db not in bb_mols:
                fg, nfgs = db.split('/')
                cls = bb_cls[nfgs]
                bb_mols[db] = [cls(p, [fg]) for p in bbs[db]]

    for r in args.reaction:
        stk.Cage.cache = {}
        db1, db2, topologies = reactions[r]
        pop = stk.Population.init_all(stk.Cage,
                                      [bb_mols[db1], bb_mols[db2]],
                                      topologies)
        pop.assign_names_from(0, True)

        db1_name = db1.replace('amine2', 'amine').replace('/', '')
        db2_name = db2.replace('amine2', 'amine').replace('/', '')
        pop.dump(db1_name + db2_name + '.json')


if __name__ == '__main__':
    main()
