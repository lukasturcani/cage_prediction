from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import (make_scorer,
                             accuracy_score,
                             recall_score,
                             precision_score,
                             roc_auc_score)
import numpy as np
import pymongo
from collections import Counter
import pickle


def get_fp(match, radius, bits, fp_tags):
    struct = next(s for s in match['structures'] if
                  s['calc_params'] == {'software': 'stk'})

    return next(fp['fp'] for fp in struct['fingerprints'] if
                fp['radius'] == radius and fp['bits'] == bits and
                all(tag in fp['type'] for tag in fp_tags))


def get_label(match, labeller):
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
        if (labeller in struct.get('collapsed', {}) and
           struct['calc_params'] == calc_params):
            return struct['collapsed'][labeller]


def load_data(db, radius, bits, fp_tags, labeller, query):
    fingerprints, topologies, labels = [], [], []
    for match in db.find(query):
        label = get_label(match, labeller)
        if label in {0, 1}:
            labels.append(label)
            topologies.append(match['topology']['class'])
            fingerprints.append(get_fp(match, radius, bits, fp_tags))

    topologies = LabelBinarizer().fit_transform(topologies)
    fingerprints = np.concatenate((fingerprints, topologies), axis=1)
    fingerprints, labels = shuffle(fingerprints, labels)
    return np.array(fingerprints), np.array(labels)


def train(db, radius, bits, fp_tags, labeller, cv, fg_name):
    """

    """
    print(fg_name)
    query = {'tags': fg_name, 'topology.class': 'FourPlusSix'}
    # query = {'tags': fg_name}
    # query = {'tags': {'$in': ['amine4aldehyde3',
    #                           'amine4aldehyde2',
    #                           'amine3aldehyde3',
    #                           'amine2aldehyde3',
    #                           'aldehyde4amine3',
    #                           'aldehyde4amine2',
    #                           'aldehyde2amine3']}}

    fingerprints, labels = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=query)
    print(Counter(labels))
    clf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced')

    scores = cross_validate(estimator=clf,
                            X=fingerprints,
                            y=labels,
                            scoring={
                             'accuracy': make_scorer(accuracy_score),
                             'precision_0': make_scorer(precision_score,
                                                        pos_label=0,
                                                        labels=[0]),
                             'recall_0': make_scorer(recall_score,
                                                     pos_label=0,
                                                     labels=[0]),
                             'precision_1': make_scorer(precision_score,
                                                        pos_label=1,
                                                        labels=[1]),
                             'recall_1': make_scorer(recall_score,
                                                     pos_label=1,
                                                     labels=[1])
                            },
                            cv=cv,
                            n_jobs=-1)

    print('accuracy', scores['test_accuracy'].mean(), sep='\n', end='\n\n')
    print('precision 0', scores['test_precision_0'].mean(), sep='\n', end='\n\n')
    print('recall 0', scores['test_recall_0'].mean(), sep='\n', end='\n\n')
    print('precision 1', scores['test_precision_1'].mean(), sep='\n', end='\n\n')
    print('recall 1', scores['test_recall_1'].mean(), sep='\n', end='\n\n')
    print('\n')

    with open(f'{fg_name}.pkl', 'wb') as f:
        clf.fit(fingerprints, labels)
        pickle.dump(clf, f)


def main():
    np.random.seed(2)
    client = pymongo.MongoClient('mongodb://localhost:27017/')

    fg_names = ['amine2aldehyde3',
                'aldehyde2amine3',
                'alkene2alkene3',
                'alkyne22alkyne23',
                # 'thiol2thiol3',
                'amine2carboxylic_acid3',
                'carboxylic_acid2amine3']
    # fg_names = ['amine2aldehyde3']
    for fg_name in fg_names:
        train(
              db=client.small.cages,
              radius=8,
              bits=512,
              fp_tags=['bb_count'],
              labeller='pywindow_plus',
              cv=5,
              fg_name=fg_name)


if __name__ == '__main__':
    main()
