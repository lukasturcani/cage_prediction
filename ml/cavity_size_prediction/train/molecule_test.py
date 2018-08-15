import rdkit.Chem.AllChem as rdkit
import pickle
import pymongo
import os
import shutil
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_structures(match):
    if os.path.exists('molecule_test'):
        shutil.rmtree('molecule_test')
    os.mkdir('molecule_test')
    for i, struct in enumerate(match['structures']):
        with open(f'molecule_test/{i}.sdf', 'w') as f:
            f.write(struct['structure'])


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

    for struct in match['structures']:
        if struct['calc_params'] == calc_params:
            return struct['cavity_size']


def fingerprint(client, match, calc_params):

    topologies = []
    q = {'tags': 'amine2aldehyde3', 'topology.class': 'FourPlusSix'}
    for m in client.small.cages.find(q):
        target = get_target(m)
        if target is not None:
            topologies.append(m['topology']['class'])
    lb = LabelBinarizer().fit(topologies)
    topology = lb.transform([match['topology']['class']])

    bb1_inchi = match['building_blocks'][0]['inchi']
    bb1_block = client.small.bbs.find_one({'inchi': bb1_inchi})['structure']
    bb2_inchi = match['building_blocks'][1]['inchi']
    bb2_block = client.small.bbs.find_one({'inchi': bb2_inchi})['structure']

    bb1 = rdkit.MolFromMolBlock(bb1_block, sanitize=False)
    bb2 = rdkit.MolFromMolBlock(bb2_block, sanitize=False)
    mols = (bb1, bb2)

    for mol in mols:
        rdkit.GetSSSR(mol)
        mol.UpdatePropertyCache(strict=False)

    full_fp = []
    for m in mols:
        fp = rdkit.GetMorganFingerprintAsBitVect(m, 2, 512)
        full_fp.extend(list(fp))
    full_fp.extend(topology)
    return np.array([full_fp])


def expected(match, calc_params):
    return next(structure['cavity_size'] for structure
                in match['structures'] if
                structure['calc_params'] == calc_params)


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

    q = {'tags': 'amine2aldehyde3', 'topology.class': 'FourPlusSix'}

    reg = load_model('random_forest.pkl')
    client = pymongo.MongoClient('mongodb://localhost:27017')
    match = client.small.cages.find_one(q)

    fp = fingerprint(client, match, {'software': 'stk'}, q)
    write_structures(match)
    print(expected(match, calc_params), reg.predict(fp))


if __name__ == '__main__':
    main()
