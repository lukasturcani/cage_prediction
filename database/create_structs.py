"""
Creates 3D mol files for every molecule in the database.

"""

import os
import rdkit.Chem.AllChem as rdkit
import stk
import itertools as it
import multiprocessing as mp
from collections import defaultdict


stk.CACHE_SETTINGS['ON'] = False


def read_smiles(file):
    with open(file, 'r') as f:
        for smiles in f:
            if not smiles.isspace():
                yield smiles.strip()


def conf_energies(mol):
    ff = rdkit.UFFGetMoleculeForceField
    for conf in mol.GetConformers():
        id_ = conf.GetId()
        yield ff(mol, confId=id_).CalcEnergy(), id_


def update_stereochemistry(mol):
    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache()
    rdkit.AssignAtomChiralTagsFromStructure(mol)
    rdkit.AssignStereochemistry(mol, True, True, True)


def make_mol(ismiles, fg_name, fg):
    try:
        smiles = ismiles.replace('[AsH2]', fg)
        # Load the molecule.
        mol = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
        # Give it 3D coords.
        rdkit.EmbedMultipleConfs(mol, 100, rdkit.ETKDG())

        # Get rid of redundant conformers.
        conf = mol.GetConformer(min(conf_energies(mol))[1])
        conf = rdkit.Conformer(conf)
        conf.SetId(0)
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        update_stereochemistry(mol)

        nfgs = len(stk.StructUnit.rdkit_init(mol, fg_name)
                   .functional_group_atoms())
        dirname = f'{fg_name}{nfgs}'

        return ismiles, smiles, mol, dirname
    except Exception as ex:
        print(ex)


def main():

    fgs = [('amine', '[N]([H])[H]'),
           ('aldehyde', '[C](=[O])[H]'),
           ('carboxylic_acid', '[C](=[O])[O][H]'),
           ('boronic_acid', '[B]([O][H])[O][H]'),
           ('thiol', '[S][H]'),
           ('terminal_alkene', '[C]([H])=[C]([H])[H]'),
           ('alkyne', '[C]#[C][H]')]

    output_dir = 'structs'
    os.mkdir(output_dir)

    with mp.Pool() as pool:
        i = ((smiles, fg_name, fg) for smiles, (fg_name, fg) in
             it.product(read_smiles('database.smi'), fgs))

        results = pool.starmap(make_mol, i)

    inchis = set()
    for ismiles, smiles, mol, dirname in results:
        opath = f'{output_dir}/{dirname}'
        if not os.path.exists(opath):
            os.mkdir(opath)

        fname = len(os.listdir(opath)) + 1

        # Make sure each molecule is unique.
        inchi = rdkit.MolToInchi(mol)
        if inchi in inchis:
            print('skipping')
            continue
        inchis.add(inchi)

        rdkit.MolToMolFile(mol, f'{opath}/{fname}.mol', forceV3000=True)


if __name__ == '__main__':
    main()
