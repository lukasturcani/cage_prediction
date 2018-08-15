from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pymongo
from collections import Counter


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


def train(db, radius, bits, fp_tags, labeller, fg_name, reverse):
    """

    """
    print(fg_name, f'reverse {reverse}')
    q1 = {'tags': fg_name, 'topology.class': 'FourPlusSix'}
    q2 = {'tags': {'$nin': [fg_name], '$exists': True},
          'topology.class': 'FourPlusSix'}
    if reverse:
        q2, q1 = q1, q2
    fp_train, labels_train = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=q1)
    fp_test, labels_test = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=q2)

    print(Counter(labels_train), Counter(labels_test), sep='\n')
    clf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced')
    clf.fit(fp_train, labels_train)

    expected = labels_test
    predicted = clf.predict(fp_test)
    accuracy = metrics.accuracy_score(expected, predicted)
    print('accuracy', accuracy, sep='\n', end='\n\n')
    print('precision',
          metrics.precision_score(expected,
                                  predicted,
                                  average=None),
          sep='\n',
          end='\n\n')
    print('recall',
          metrics.recall_score(expected,
                               predicted,
                               average=None),
          sep='\n',
          end='\n\n')
    print('\n')


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
    for fg_name in fg_names:
        train(
              db=client.small.cages,
              radius=8,
              bits=512,
              fp_tags=['bb_count'],
              labeller='pywindow_plus',
              fg_name=fg_name,
              reverse=False)
        train(
              db=client.small.cages,
              radius=8,
              bits=512,
              fp_tags=['bb_count'],
              labeller='pywindow_plus',
              fg_name=fg_name,
              reverse=True)


if __name__ == '__main__':
    main()
