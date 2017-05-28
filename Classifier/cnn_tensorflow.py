import pandas
import os
import librosa
import csv
import numpy as np
import random
import tensorflow as tf
import time
from collections import Counter
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def create_balanced_dataset():
    """ Load samples and relative label randomly, such to have a balanced number of entity for each label 
    
    :return: pandas DataFrame
    """
    print('Creating balanced dataset')
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

    # extra = {'label', 'nobody', 'default', 'exit'}
    extra = {'label', 'nobody', 'default'}
    max_samples = sum(len(labels[x]) for x in labels.keys() - extra)
    files = set(os.listdir(directory))

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


def dataset_to_pickle(dataset, output_path):
    """ Save the whole dataset as pickle file
    :param dataset: pandas DataFrame
    :param output_path: path adn file name of output
    """
    pandas.to_pickle(dataset, output_path)


def to1hot(row):
    """ Make the input a 1-hot vector

    :param row: 
    :return: 
    """
    # mapping = {
    #     'default': 0,
    #     'alessio': 1,
    #     'andrea': 2,
    #     'debora': 3,
    #     'mamma': 4,
    #     'papa': 5,
    #     'nobody': 6,
    #     'exit': 7,
    #     'bell': 8,
    # }
    # one_hot = np.zeros(len(mapping))
    mapping = {
        'alessio': 1,
        'andrea': 1,
        'debora': 1,
        'mamma': 1,
        'papa': 1,
        'nobody': 0,
        'exit': 1,
        'bell': 1,
    }
    return mapping[row]


def model(features, labels, mode):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    # mode OR= TRAIN, EVAL, INFER

    # [batch_size, image_width, image_height, channels]
    # -1 as batch size = dynamic
    # channels may be the nu ber of audio features used
    # TODO find dimension programmatically
    input_layer = tf.reshape(features, [-1, 128, 2813, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv1_layer'
    )
    print("conv1 shape: %s" % conv1.get_shape())

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2,
        name='pool1_layer'
    )
    print("pool1 shape: %s" % pool1.get_shape())

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv2_layer'
    )
    print("conv2 shape: %s" % conv2.get_shape())

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name='pool2_layer'
    )
    print("pool2 shape: %s" % pool2.get_shape())

    # Dense Layer

    # we'll flatten our feature map (pool2) to shape [batch_size, features]
    pool2_shape_flatten = 1
    for d in pool2.get_shape()[1:]:
        pool2_shape_flatten *= int(d)
    pool2_flat = tf.reshape(pool2, [-1, pool2_shape_flatten])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=128,  # 1024
        activation=tf.nn.relu,
        name='dense_layer'
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == learn.ModeKeys.TRAIN,
        name='dropout_layer'
    )

    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=2,
        name='logits_layer'
    )

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2, name='one_hot_labels')
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            # onehot_labels=labels,
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


def main(unused_argv):
    ####################################################################################
    # Load dataset
    print('loading dataset...')
    #   CREATE
    # ds = create_balanced_dataset()
    #   OPEN
    ds = pandas.read_pickle(os.path.join('Classifier', "balanced_dataset_48000_door_only.pickle"))
    #   WRITE FILE
    # dataset_to_pickle(ds, os.path.join('Classifier', "balanced_dataset_48000_door_only.pickle"))

    print("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds[ds["target"] == 'nobody'].index.size,
            ds[ds["target"] == 'andrea'].index.size +
            ds[ds["target"] == 'exit'].index.size +
            ds[ds["target"] == 'debora'].index.size +
            ds[ds["target"] == 'alessio'].index.size +
            ds[ds["target"] == 'mamma'].index.size +
            ds[ds["target"] == 'papa'].index.size +
            ds[ds["target"] == 'bell'].index.size,
        ) / (float)(ds.index.size)))

    ####################################################################################
    # Divide the dataset in training/testing/validation sets
    # dataset proportion
    # 0.78 ~ 75 % - Train
    # 0.14 ~ 15 % - Validation
    # 0.07 ~ 10 % - Test
    ds["one_hot_encoding"] = ds.target.apply(to1hot)
    ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
    index_train = int(len(ds) * 0.80)
    index_validation = index_train + int(len(ds) * 0.20)
    index_test = index_validation + int(len(ds) * 0.10)

    # Load training and eval data
    train_data = np.asanyarray(list(ds.mels_flatten[0:index_train]), dtype=np.float32)
    train_labels = np.asanyarray(list(ds.one_hot_encoding[0:index_train]))
    eval_data = np.asanyarray(list(ds.mels_flatten[index_train:index_validation]), dtype=np.float32)  # Returns np.array
    eval_labels = np.asanyarray(list(ds.one_hot_encoding[index_train:index_validation]))

    # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=model,
        model_dir=os.path.join('Classifier', 'model', 'cnn3', 'convnet_model')
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=ds.shape[0] / 100 * 20)

    # Train the model
    mnist_classifier.fit(
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
    eval_results = mnist_classifier.evaluate(
        x=eval_data,
        y=eval_labels,
        metrics=metrics,
        batch_size=5
    )
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
