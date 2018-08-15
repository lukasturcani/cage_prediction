from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import (cross_val_score,
                                     train_test_split)
import numpy as np
import itertools as it
import pandas as pd
import pymongo
import argparse


pd.set_option('expand_frame_repr', False)


def get_fp(match, radius, bits, fp_tags):
    for struct in match['structures']:
        for fp in struct['fingerprints']:
            if (fp['radius'] == radius and
                fp['bits'] == bits and
               all(tag in fp['type'] for tag in fp_tags)):
                return fp['fp']


def get_label(match, labeller):
    for struct in match['structures']:
        if labeller in struct['collapsed']:
            return struct['collapsed'][labeller]


def load_data(db, mol_tags, radius, bits, fp_tags, labeller):
    fp_match = {'$elemMatch': {'fingerprints.radius': radius,
                               'fingerprints.bits': bits,
                               'fingerprints.type': {'$all': fp_tags}}}
    matches = db.find({'tags': {'$all': mol_tags},
                       'structures': fp_match})

    fingerprints, labels = [], []
    for match in matches:
        label = get_label(match, labeller)
        if label in {0, 1}:
            labels.append(label)
            fingerprints.append(get_fp(match, radius, bits, fp_tags))

    fingerprints, labels = shuffle(fingerprints, labels)
    return np.array(fingerprints), np.array(labels)


def train(db, mol_tags, radius, bits, fp_tags, labeller, cv):
    """

    """

    fingerprints, labels = load_data(
                                     db=db,
                                     mol_tags=mol_tags,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller)

    clf = svm.SVC(kernel='linear', class_weight='balanced')
    scores = cross_val_score(clf, fingerprints, labels, cv=cv, n_jobs=-1)

    split = train_test_split(fingerprints,
                             labels,
                             test_size=0.2,
                             stratify=labels)
    fp_train, fp_test, labels_train, labels_test = split

    clf.fit(fp_train, labels_train)

    expected = labels_test
    predicted = clf.predict(fp_test)

    clf_report = metrics.classification_report(expected, predicted)
    conf_matrix = metrics.confusion_matrix(expected, predicted)

    return scores.mean(), conf_matrix, clf_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--print', action='store_true',
                        help='Prints the result after an SVM is trained.')

    args = parser.parse_args()

    np.random.seed(42)
    client = pymongo.MongoClient()
    db = client.liverpool_refined.cages
    results = pd.DataFrame(columns=['radius',
                                    'bits',
                                    'featurization',
                                    'labeller',
                                    'tags',
                                    '5_fold_cv',
                                    'conf_matrix',
                                    'report'])

    radii = [1, 2, 4, 8, 16]
    bit_sizes = [256, 512, 1024]
    fp_tags = ['bb', 'cage', 'cage+bb']
    labellers = ['lukas', 'pywindow', 'pywindow_plus']
    mol_tag_sets = [('amines2aldehydes3', 'liverpool_refined')]

    params = it.product(radii, bit_sizes, fp_tags, labellers, mol_tag_sets)
    for radius, bits, fp_tag, labeller, mol_tags in params:
        cv_score, conf_matrix, report = train(
              db=db,
              mol_tags=mol_tags,
              radius=radius,
              bits=bits,
              fp_tags=[fp_tag],
              labeller=labeller,
              cv=5)

        results = results.append(
                       {
                        'radius': radius,
                        'bits': bits,
                        'featurization': fp_tag,
                        'labeller': labeller,
                        'tags': mol_tags,
                        '5_fold_cv': cv_score,
                        'conf_matrix': conf_matrix,
                        'report': report},
                       ignore_index=True)
        if args.print:
            print(results)

    results.to_pickle('vanilla_results.pkl')


if __name__ == '__main__':
    main()
