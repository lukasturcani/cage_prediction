from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
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


def train(db, radius, bits, fp_tags, labeller):
    """

    """
    q1 = {'tags': 'amine2aldehyde3', 'topology.class': 'FourPlusSix'}
    q2 = {'tags': 'alkene2alkene3'}
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
    clf = svm.LinearSVC(class_weight='balanced')
    clf.fit(fp_train, labels_train)

    expected = labels_test
    predicted = clf.predict(fp_test)

    clf_report = metrics.classification_report(expected, predicted)
    conf_matrix = metrics.confusion_matrix(expected, predicted)
    accuracy = metrics.accuracy_score(expected, predicted)

    return accuracy, conf_matrix, clf_report


def main():
    np.random.seed(42)
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    accuracy, conf_matrix, report = train(
                                          db=client.small.cages,
                                          radius=2,
                                          bits=512,
                                          fp_tags=['bb'],
                                          labeller='pywindow_plus')
    print(accuracy, conf_matrix, report, sep='\n\n')


if __name__ == '__main__':
    main()
