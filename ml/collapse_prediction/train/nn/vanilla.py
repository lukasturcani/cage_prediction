import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import argparse
import pymongo
from sklearn.model_selection import train_test_split


class InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(self.init, self.feed_dict)


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


def load_data(query, labeller, radius, bits, fp_tags):
    db = pymongo.MongoClient('mongodb://localhost:27017/').small.cages

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
    labels = np.array(labels, np.int32).reshape((-1, 1))
    labels = OneHotEncoder(sparse=False).fit_transform(labels)
    return np.array(fingerprints, np.float32), labels


def get_input_fn(batch_size, fps, labels, repeat):

    init_hook = InitHook()

    def input_fn():
        fp_ph = tf.placeholder(fps.dtype, fps.shape)
        label_ph = tf.placeholder(labels.dtype, labels.shape)

        dset = tf.data.Dataset.from_tensor_slices((fp_ph, label_ph))
        dset = dset.shuffle(1000).batch(batch_size)
        if repeat:
            dset = dset.repeat()

        iterator = dset.make_initializable_iterator()
        next_fps, next_labels = iterator.get_next()
        init_hook.init = iterator.initializer
        init_hook.feed_dict = {fp_ph: fps, label_ph: labels}
        return {'fps': next_fps}, next_labels

    return input_fn, init_hook


def nn(features, mode, params):

    training = mode == tf.estimator.ModeKeys.TRAIN

    prev_layer = features['fps']
    for i, units in enumerate(params.fc_layers, 1):
        with tf.variable_scope(f'fc_layer_{i}'):
            prev_layer = tf.layers.dense(inputs=prev_layer,
                                         units=units,
                                         name=f'fc')
            if params.batch_norm:
                prev_layer = tf.layers.batch_normalization(
                                                   inputs=prev_layer,
                                                   training=training)

            prev_layer = tf.nn.relu(prev_layer)

    return tf.layers.dense(inputs=prev_layer, units=2, name='logits')


def model_fn(features, labels, mode, params):
    logits = nn(features, mode, params)
    predictions = tf.nn.softmax(logits=logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predcitions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)

    trainer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = trainer.minimize(
                                loss=loss,
                                global_step=tf.train.get_global_step())

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
                            labels=tf.argmax(labels, 1),
                            predictions=tf.argmax(predictions, 1))
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def main():
    np.random.seed(420)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--train_steps', default=1000000, type=int)
    parser.add_argument('--save_summary_steps', default=1000, type=int)
    parser.add_argument('--save_checkpoints_steps', default=50, type=int)
    parser.add_argument('--learning_rate', default=2e-3, type=float)
    parser.add_argument('--fc_layers',
                        default=[1000, 100],
                        type=int,
                        nargs='+')
    parser.add_argument('--batch_norm', action='store_false')

    params = parser.parse_args()

    config = tf.estimator.RunConfig(
                    model_dir=params.model_dir,
                    tf_random_seed=420,
                    save_summary_steps=params.save_summary_steps,
                    save_checkpoints_steps=params.save_checkpoints_steps)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       config=config,
                                       params=params)

    query = {'tags': 'amine2aldehyde3', 'topology.class': 'FourPlusSix'}
    fps, labels = load_data(query=query,
                            labeller='pywindow_plus',
                            radius=2,
                            bits=512,
                            fp_tags=['bb'])

    split = train_test_split(fps, labels, test_size=0.2, stratify=labels)
    train_fps, eval_fps, train_labels, eval_labels = split

    train_input_fn, train_init_hook = get_input_fn(
            batch_size=params.train_batch_size,
            fps=train_fps,
            labels=train_labels,
            repeat=True)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=params.train_steps,
                                        hooks=[train_init_hook])

    eval_input_fn, eval_init_hook = get_input_fn(
            batch_size=params.eval_batch_size,
            fps=eval_fps,
            labels=eval_labels,
            repeat=False)

    eval_spec = tf.estimator.EvalSpec(
                    input_fn=eval_input_fn,
                    steps=None,
                    hooks=[eval_init_hook],
                    start_delay_secs=0,
                    throttle_secs=1)

    estimator.train(input_fn=train_input_fn,
                    hooks=[train_init_hook],
                    max_steps=1)
    estimator.evaluate(input_fn=eval_input_fn,
                       hooks=[eval_init_hook])
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
