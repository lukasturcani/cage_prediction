import pymongo
import shutil
import os


def main():
    nwritten = 10
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    cages = client.small.cages
    bbs = client.small.bbs
    query = {}
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

    # cbbs, clks - collapsed bbs , collapsed lks
    # nbbs, nlks - not collapsed bbs, not collapsed lks
    cbbs, clks, nbbs, nlks = {}, {}, {}, {}

    for match in cages.find(query):
        struct = next((s for s in match['structures'] if
                       s['calc_params'] == calc_params), {})

        if 'pywindow_plus' not in struct.get('collapsed', {}):
            continue

        bb1 = match['building_blocks'][0]['inchi']
        bb2 = match['building_blocks'][1]['inchi']

        nfg1 = bbs.find_one({'inchi': bb1})['num_fgs']
        nfg2 = bbs.find_one({'inchi': bb2})['num_fgs']

        bb = bb1 if nfg1 > nfg2 else bb2
        lk = bb2 if bb is bb1 else bb1

        if struct['collapsed']['pywindow_plus']:
            cbbs[bb] = cbbs.get(bb, 0) + 1
            clks[lk] = clks.get(lk, 0) + 1
        else:
            nbbs[bb] = nbbs.get(bb, 0) + 1
            nlks[lk] = nlks.get(lk, 0) + 1

    cbbs = sorted(zip(cbbs.values(), cbbs.keys()), reverse=True)
    clks = sorted(zip(clks.values(), clks.keys()), reverse=True)
    nbbs = sorted(zip(nbbs.values(), nbbs.keys()), reverse=True)
    nlks = sorted(zip(nlks.values(), nlks.keys()), reverse=True)

    if os.path.exists('building_blocks'):
        shutil.rmtree('building_blocks')

    os.mkdir('building_blocks')

    os.mkdir('building_blocks/collapsed')
    os.mkdir('building_blocks/not_collapsed')

    folders = ['building_blocks/collapsed/bbs',
               'building_blocks/collapsed/lks',
               'building_blocks/not_collapsed/bbs',
               'building_blocks/not_collapsed/lks']
    for folder in folders:
        os.mkdir(folder)

    for i in range(nwritten):
        cbb = bbs.find_one({'inchi': cbbs[i][1]})['structure']
        cbb_fg = bbs.find_one({'inchi': cbbs[i][1]})['fg']
        clk = bbs.find_one({'inchi': clks[i][1]})['structure']
        clk_fg = bbs.find_one({'inchi': clks[i][1]})['fg']
        nbb = bbs.find_one({'inchi': nbbs[i][1]})['structure']
        nbb_fg = bbs.find_one({'inchi': nbbs[i][1]})['fg']
        nlk = bbs.find_one({'inchi': nlks[i][1]})['structure']
        nlk_fg = bbs.find_one({'inchi': nlks[i][1]})['fg']

        structs = [(cbb, cbb_fg),
                   (clk, clk_fg),
                   (nbb, nbb_fg),
                   (nlk, nlk_fg)]

        for folder, (struct, fg) in zip(folders, structs):
            with open(os.path.join(folder, f'{i+1}_{fg}.sdf'), 'w') as f:
                f.write(struct)


if __name__ == '__main__':
    main()
