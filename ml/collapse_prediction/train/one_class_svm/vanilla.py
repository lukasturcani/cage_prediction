from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pymongo


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
    ifps, itops, ilabels = [], [], []
    ofps, otops, olabels = [], [], []

    for match in db.find(query):
        label = get_label(match, labeller)
        if label == 1:
            ilabels.append(label)
            itops.append(match['topology']['class'])
            ifps.append(get_fp(match, radius, bits, fp_tags))
        elif label == 0:
            olabels.append(-1)
            otops.append(match['topology']['class'])
            ofps.append(get_fp(match, radius, bits, fp_tags))

    binizer = LabelBinarizer().fit(itops+otops)
    itops = binizer.transform(itops)
    otops = binizer.transform(otops)
    ifps = np.concatenate((ifps, itops), axis=1)
    ofps = np.concatenate((ofps, otops), axis=1)
    ifps, ilabels = shuffle(ifps, ilabels)
    return (np.array(ifps),
            np.array(ofps),
            np.array(ilabels),
            np.array(olabels))


def train(db, radius, bits, fp_tags, labeller, test_size):
    """

    """

    query = {'tags': 'aldehyde2amine3', 'topology.class': 'FourPlusSix'}

    ifps, ofps, ilabels, olabels = load_data(
                                     db=db,
                                     radius=radius,
                                     bits=bits,
                                     fp_tags=fp_tags,
                                     labeller=labeller,
                                     query=query)
    print(1, len(ilabels), -1, len(olabels))
    clf = svm.OneClassSVM(kernel='linear')

    train_size = int(len(ifps)*(1-test_size))
    clf.fit(ifps[:train_size])

    iexpected = ilabels[train_size:]
    ipredicted = clf.predict(ifps[train_size:])
    oexpected = olabels
    opredicted = clf.predict(ofps)

    expected = np.concatenate((iexpected, oexpected))
    predicted = np.concatenate((ipredicted, opredicted))

    clf_report = metrics.classification_report(expected, predicted)
    conf_matrix = metrics.confusion_matrix(expected, predicted)

    p = clf.predict(ifps[:train_size])
    from collections import Counter
    print(Counter(p))

    return conf_matrix, clf_report


def main():
    np.random.seed(42)
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    conf_matrix, report = train(
                                          db=client.small.cages,
                                          radius=2,
                                          bits=512,
                                          fp_tags=['bb'],
                                          labeller='pywindow_plus',
                                          test_size=0.2)
    print(conf_matrix, report, sep='\n\n')


if __name__ == '__main__':
    main()
