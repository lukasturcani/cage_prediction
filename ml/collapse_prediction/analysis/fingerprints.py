import pymongo
import shutil
import os
import numpy as np
import rdkit.Chem.AllChem as rdkit
from collections import defaultdict
import heapq


def sum_fps(client):

    radius = 2
    bits = 512
    cages = client.small.cages
    bbs = client.small.bbs
    query = {'tags': 'amine2aldehyde3', 'topology.class': 'FourPlusSix'}
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

    collapsed = np.zeros(bits)
    not_collapsed = np.zeros(bits)
    # Maps an activated bit to the molecule and atoms which
    # activated it.
    activated_bits = defaultdict(list)

    for match in cages.find(query):
        struct = next((s for s in match['structures'] if
                       s['calc_params'] == calc_params), {})

        if 'pywindow_plus' not in struct.get('collapsed', {}):
            continue

        bb1_inchi = match['building_blocks'][0]['inchi']
        bb2_inchi = match['building_blocks'][1]['inchi']

        bb1 = bbs.find_one({'inchi': bb1_inchi})['structure']
        bb2 = bbs.find_one({'inchi': bb2_inchi})['structure']

        bb1 = rdkit.MolFromMolBlock(bb1, sanitize=False)
        bb2 = rdkit.MolFromMolBlock(bb2, sanitize=False)

        fp = np.zeros(bits)
        for mol in (bb1, bb2):
            rdkit.GetSSSR(mol)
            mol.UpdatePropertyCache(strict=False)
            info = {}
            fp += np.array(rdkit.GetMorganFingerprintAsBitVect(
                                                    mol,
                                                    radius,
                                                    bits,
                                                    bitInfo=info))
            for bit_index, activators in info.items():
                activated_bits[bit_index].append((mol, activators))

        if struct['collapsed']['pywindow_plus']:
            collapsed += fp
        else:
            not_collapsed += fp

    return collapsed, not_collapsed, activated_bits


def substructures(bit_index, activated_bits):
    seen = set()
    for mol, activators in activated_bits[bit_index]:
        for aid, radius in activators:
            if not radius:
                continue
            env = rdkit.FindAtomEnvironmentOfRadiusN(mol,
                                                     radius,
                                                     aid,
                                                     True)
            atoms = set()
            for bid in env:
                atoms.add(mol.GetBondWithIdx(bid).GetBeginAtomIdx())
                atoms.add(mol.GetBondWithIdx(bid).GetEndAtomIdx())
            smiles = rdkit.MolFragmentToSmiles(mol,
                                               atomsToUse=list(atoms),
                                               bondsToUse=env,
                                               canonical=True)
            if smiles not in seen:
                seen.add(smiles)
                yield smiles


def main():
    nwritten = 10
    client = pymongo.MongoClient('mongodb://localhost:27017/')

    collapsed, not_collapsed, activated_bits = sum_fps(client)

    if os.path.exists('building_blocks'):
        shutil.rmtree('building_blocks')

    os.mkdir('building_blocks')

    os.mkdir('building_blocks/collapsed')
    os.mkdir('building_blocks/not_collapsed')

    # Take the most common bits and for each building block molecule
    # look at what substructure it corresponds to.

    # Make a datastructure which maps each count to the indices at
    # which that count is found.

    collapsed_counts = defaultdict(list)
    for i, bit_count in enumerate(collapsed):
        collapsed_counts[bit_count].append(i)

    not_collapsed_counts = defaultdict(list)
    for i, bit_count in enumerate(not_collapsed):
        not_collapsed_counts[bit_count].append(i)

    # Use bit number to identify which atoms were involved in the
    # bit.

    for i, bit_count in enumerate(heapq.nlargest(nwritten,
                                                 collapsed_counts)):
        dirname = f'building_blocks/collapsed/{i}'
        os.mkdir(dirname)
        n = 0
        for bit_index in collapsed_counts[bit_count]:
            for smiles in substructures(bit_index, activated_bits):
                mol = rdkit.MolFromSmiles(smiles)
                rdkit.EmbedMolecule(mol, rdkit.ETKDG())
                rdkit.MolToMolFile(mol,
                                   os.path.join(dirname, f'{n}.sdf'))
                n += 1

    for i, bit_count in enumerate(heapq.nlargest(nwritten,
                                                 not_collapsed_counts)):
        dirname = f'building_blocks/not_collapsed/{i}'
        os.mkdir(dirname)
        n = 0
        for bit_index in not_collapsed_counts[bit_count]:
            for smiles in substructures(bit_index, activated_bits):
                mol = rdkit.MolFromSmiles(smiles)
                rdkit.EmbedMolecule(mol, rdkit.ETKDG())
                rdkit.MolToMolFile(mol,
                                   os.path.join(dirname, f'{n}.sdf'))
                n += 1


if __name__ == '__main__':
    main()
