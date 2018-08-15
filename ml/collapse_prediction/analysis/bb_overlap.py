import pymongo


def overlap(precursors, top):
    top_bb_freq, top_lk_freq, bot_bb_freq, bot_lk_freq = {}, {}, {}, {}
    for dset, (cbbs, clks, nbbs, nlks) in precursors.items():
        bot_bb = sorted(zip(cbbs.values(), cbbs.keys()), reverse=True)[:top]
        bot_lk = sorted(zip(clks.values(), clks.keys()), reverse=True)[:top]
        top_bb = sorted(zip(nbbs.values(), nbbs.keys()), reverse=True)[:top]
        top_lk = sorted(zip(nlks.values(), nlks.keys()), reverse=True)[:top]

        for count, inchi in top_bb:
            ...




def main():
    top = 10
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

    precursors = {}

    for match in cages.find(query):
        struct = next((s for s in match['structures'] if
                       s['calc_params'] == calc_params), {})

        if 'pywindow_plus' not in struct.get('collapsed', {}):
            continue

        key = (match['topology']['class'] +
               next(t for t in match['tags'] if
               '2' in t or '3' in t or '4' in t))

        cbbs, clks, nbbs, nlks = precursors.get(key, ({}, {}, {}, {}))

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

        precursors[key] = cbbs, clks, nbbs, nlks

    print(list(precursors.keys()))

    overlap(precursors, top)


if __name__ == '__main__':
    main()
