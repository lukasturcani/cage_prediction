from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import (cross_val_score,
                                     train_test_split)
import numpy as np
import pymongo
from collections import Counter


def get_fp(match, radius, bits, fp_tags):
    struct = next(s for s in match['structures'] if
                  s['calc_params'] == {'software': 'stk'})

    for fp in struct['fingerprints']:
        if (fp['radius'] == radius and
            fp['bits'] == bits and
           all(tag in fp['type'] for tag in fp_tags)):
            return fp['fp']


def get_label(match, labeller):
    for struct in match['structures']:
        if labeller in struct.get('collapsed', {}):
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


def train(db, radius, bits, fp_tags, labeller, cv):
    """

    """

    query = {'tags': 'aldehyde2amine3'}

    fingerprints, labels = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=query)
    print(Counter(labels))
    clf = svm.SVC(kernel='linear', class_weight='balanced')
    scores = cross_val_score(clf, fingerprints, labels, cv=cv, n_jobs=-1)

    split = train_test_split(fingerprints,
                             labels,
                             test_size=1/cv,
                             stratify=labels)
    fp_train, fp_test, labels_train, labels_test = split

    clf.fit(fp_train, labels_train)

    expected = labels_test
    predicted = clf.predict(fp_test)

    clf_report = metrics.classification_report(expected, predicted)
    conf_matrix = metrics.confusion_matrix(expected, predicted)

    return scores.mean(), conf_matrix, clf_report


def main():
    np.random.seed(42)
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    cv_score, conf_matrix, report = train(
                                          db=client.small.cages,
                                          radius=2,
                                          bits=512,
                                          fp_tags=['bb'],
                                          labeller='pywindow_plus',
                                          cv=5)
    print(cv_score, conf_matrix, report, sep='\n\n')


if __name__ == '__main__':
    main()
