import pandas
import os
import librosa
import csv
import sqlite3
import numpy as np
import random
import logging
from collections import Counter
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def load_labels_from_csv(csv_path, targets_mapping):
    """ Loads the dataset files with relative labels from the given CSV file """
    logging.info('Loading samples and Labels from CSV (', csv_path, ')...')
    labels = dict()  # key: label, value: set of mapped samples
    samples = dict()  # key: sample, value: label
    with open(csv_path, 'r', newline='') as csvfile:
        for record in csv.reader(csvfile):
            if record[1] not in targets_mapping.keys():
                continue
            labels.setdefault(record[1], set()).add(record[0])
            samples[record[0]] = record[1]
    return labels, samples


def load_labels_from_db(db_path, targets_mapping):
    """ Loads the dataset files with relative labels from the given SQLite DB """
    logging.info('Loading samples and Labels from DB (', db_path, ')...')
    labels = dict()  # key: label, value: set of mapped samples
    samples = dict()  # key: sample, value: label
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        for record in c.execute("SELECT name,label FROM accesses WHERE label IN ({seq})".format(
                seq=','.join(['?'] * len(targets_mapping))), list(targets_mapping.keys())).fetchall():
            labels.setdefault(record[1], set()).add(record[0])
            samples[record[0]] = record[1]

    return labels, samples


def create_balanced_dataset(targets_mapping, data_source_path, discriminant_targets=[]):
    """ Load samples and relative label randomly, such to have a balanced number of entity for each label

    :param targets_mapping:
    :param discriminant_targets: is not -1, ds will have at least all the elements mapped to this target
    :return: pandas DataFrame
    """
    logging.info('Creating balanced dataset')
    dataset = pandas.DataFrame(columns=['data', 'target', 'mels', 'mfcc'])
    dataset.target = dataset.target.astype(str)

    directory = os.path.join('DB', 'Samples')
    labels_path = os.path.join('DB', 'Samples', 'labels.csv')

    if os.path.basename(data_source_path).split('.')[-1] == 'sqlite':
        labels, samples = load_labels_from_db(os.path.join('DB', 'smartSlamDB.sqlite'), targets_mapping)
    elif os.path.basename(data_source_path).split('.')[-1] == 'csv':
        labels, samples = load_labels_from_csv(labels_path, targets_mapping)
    else:
        raise Exception('Label file not recognized')

    inv_target_mapping = Counter(set(targets_mapping.values()))
    for key, value in labels.items():
        inv_target_mapping[targets_mapping[key]] += len(value)
    max_samples = inv_target_mapping.most_common()[-1][1]
    if len(discriminant_targets) > 0:
        max_samples = max([max_samples,
                           sum(inv_target_mapping[w] for w in discriminant_targets)])

    chosen = []
    for label in labels:
        i = max_samples
        if len(labels[label]) < max_samples:
            i = len(labels[label])
        chosen.extend(random.sample(labels[label], i))
    chosen = random.sample(chosen, len(chosen))

    for filename in chosen:
        filename += '.wav'
        if os.path.exists(os.path.join(directory, filename)):
            if samples[filename[:-4]] == 'default':
                continue
            y, sr = librosa.core.load(os.path.join(directory, filename), sr=48000, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            target = samples[filename[:-4]]
            mapping = {
                'data': [y],
                'target': target,
                'mels': [mel],
                'mfcc': [mfcc]
            }
            dataset = dataset.append(pandas.DataFrame(mapping), ignore_index=True)

    return dataset


def create_random_sample(num):
    """ Loads a random sample of num elements """
    logging.info('Creating random dataset of %i element' % num)
    dataset = pandas.DataFrame(columns=['data', 'target', 'mels', 'mfcc'])
    dataset.target = dataset.target.astype(str)

    directory = os.path.join('DB', 'Samples')
    labels_path = os.path.join('DB', 'Samples', 'labels.csv')

    labels = dict()  # key: label, value: set of mapped samples
    samples = dict()  # key: sample, value: label
    with open(labels_path, 'r', newline='') as csvfile:
        for record in csv.reader(csvfile):
            if record[1] == 'default':
                continue
            labels.setdefault(record[1], set()).add(record[0])
            samples[record[0]] = record[1]

    while num:
        filename = random.sample(list(samples), 1)[0]
        if samples[filename] == 'default':
            continue
        num -= 1
        y, sr = librosa.core.load(os.path.join(directory, filename + '.wav'), sr=48000, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        target = samples[filename]
        mapping = {
            'data': [y],
            'target': target,
            'mels': [mel],
            'mfcc': [mfcc]
        }
        dataset = dataset.append(pandas.DataFrame(mapping), ignore_index=True)
        # print('>>>>> Loading file : ' + filename + ' | label: ' + target)
    return dataset


def save_dataset(dataset, output_path):
    """ Save the whole dataset as pickle file
    :param dataset: pandas DataFrame
    :param output_path: path adn file name of output
    """
    logging.info('Saving dataset to pickle file :' + output_path)
    pandas.to_pickle(dataset, output_path)


def freeze_graph(model_folder, output_graph):
    """ Freeze a saved model into a self contained file"""
    logging.info('Freezing model : ' + output_graph)
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "softmax_tensor"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        logging.info("%d ops in the final graph." % len(output_graph_def.node))


def evaluate(graph, mels, label, mapping):
    """ Check correctness of a file classification """
    logging.info('Evaluating audio classification')
    audio_feature = np.asanyarray(list(mels.flatten()), dtype=np.float32)

    true_result = mapping[label]

    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        y_out = sess.run(y, feed_dict={
            x: [audio_feature]
        })

        logging.info('true value:' + str(true_result))
        logging.info('predicted value:' + str(y_out[0].argmax()))
        logging.info('predictions:' + str(y_out))
        if y_out[0].argmax() == true_result:
            return True
        else:
            return False


def to_numeric(row, mapping):
    """ Return the numeric index of the corresponding label """
    logging.debug('Making numeric label')
    return mapping[row]


def model(features, labels, mode, params):
    """ The Model definition """
    logging.info('Model definition')
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    # mode OR= TRAIN, EVAL, INFER

    # [batch_size, image_width, image_height, channels]
    # -1 as batch size = dynamic
    # channels may be the nu ber of audio features used
    # TODO parametrize fixed values in separate config file
    # TODO find dimension programmatically
    input_layer = tf.reshape(features, [-1, 128, 2813, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,  # 32
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv1_layer'
    )
    logging.info("conv1 shape: %s" % conv1.get_shape())

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2,
        name='pool1_layer'
    )
    logging.info("pool1 shape: %s" % pool1.get_shape())

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,  # 64
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv2_layer'
    )
    logging.info("conv2 shape: %s" % conv2.get_shape())

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name='pool2_layer'
    )
    logging.info("pool2 shape: %s" % pool2.get_shape())

    # Dense Layer

    # we'll flatten our feature map (pool2) to shape [batch_size, features]
    pool2_shape_flatten = 1
    for d in pool2.get_shape()[1:]:
        pool2_shape_flatten *= int(d)
    pool2_flat = tf.reshape(pool2, [-1, pool2_shape_flatten])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=64,  # 128 # 1024
        activation=tf.nn.relu,
        name='dense_layer'
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == learn.ModeKeys.TRAIN,
        name='dropout_layer'
    )

    labels_num = params['labels_num']
    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=labels_num,
        name='logits_layer'
    )

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=labels_num, name='one_hot_labels')
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits
        )

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD",
            name='train_operation_layer'
        )

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def train(ds, model_name, mapping):
    """ Train a model with the given name and mapping with the given dataset.

    If model is already existing the training will continue from the previous status.
    """
    logging.info('Training')
    # Divide the dataset in training/testing/validation sets
    # dataset proportion
    # 0.78 ~ 75 % - Train
    # 0.14 ~ 15 % - Validation
    # 0.07 ~ 10 % - Test
    ds["target_numeric"] = ds.target.apply(to_numeric, args=[mapping])
    ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
    labels_num = len(set(mapping.values()))

    index_train = int(len(ds) * 0.80)
    index_validation = index_train + int(len(ds) * 0.20)

    # Load training and eval data
    train_data = np.asanyarray(list(ds.mels_flatten[0:index_train]), dtype=np.float32)
    train_labels = np.asanyarray(list(ds.target_numeric[0:index_train]))
    eval_data = np.asanyarray(list(ds.mels_flatten[index_train:index_validation]), dtype=np.float32)  # Returns np.array
    eval_labels = np.asanyarray(list(ds.target_numeric[index_train:index_validation]))

    # Create the Estimator
    door_classifier = learn.Estimator(
        model_fn=model,
        params={'labels_num': labels_num},
        model_dir=os.path.join('Classifier', 'model', model_name, 'convnet_model')
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=ds.shape[0] / 100 * 20)

    # Train the model
    door_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=5,
        steps=2000,  # 20000
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = door_classifier.evaluate(
        x=eval_data,
        y=eval_labels,
        metrics=metrics,
        batch_size=5
    )
    logging.info(eval_results)

    # Freeze model
    if not os.path.exists(os.path.join('Classifier', 'model', model_name, 'frozen')):
        os.mkdir(os.path.join('Classifier', 'model', model_name, 'frozen'))
    freeze_graph(
        os.path.join('Classifier', 'model', model_name, 'convnet_model'),
        os.path.join('Classifier', 'model', model_name, 'frozen', 'frozen_model.pb'),
    )
