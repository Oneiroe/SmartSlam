import tensorflow as tf
import pandas
import os
import csv
import librosa
import random
import numpy as np


def create_random_sample(num):
    """ Loads a random sample of num elements """
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
        print('>>>>> Loading file : ' + filename + ' | label: ' + target)
    return dataset


def load_graph(frozen_graph_filename):
    """ Load graph into to be used """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    print('>>> Loading graph...')
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def evaluate(graph, mels, label):
    """ Takes the input audio file/feature and classify it against the model """

    # audio_feature = np.asanyarray(list(audio.mels[0].flatten()), dtype=np.float32)
    audio_feature = np.asanyarray(list(mels.flatten()), dtype=np.float32)

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
    true_result = mapping[label]

    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        y_out = sess.run(y, feed_dict={
            x: [audio_feature]
        })

        print('true value:' + str(true_result))
        print('predictions:' + str(y_out))
        if y_out[0].argmax() == true_result:
            print('Result: CORRECT')
        else:
            print('Result: WRONG')


def predict(graph, mels):
    """ Returns the eximation of the given audio file according to the model """

    audio_feature = np.asanyarray(list(mels.flatten()), dtype=np.float32)

    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: [audio_feature]  # < 45
        })

        print('predictions:' + str(y_out))
        return y_out[0].argmax()


####################################################################################
# Load model
print('> Load model')
graph = load_graph(os.path.join('Classifier', 'model', 'door_not_door', 'freezed', 'frozen_model.pb'))

####################################################################################
# Load dataset
print('> Loading dataset')
#   CREATE
ds = create_random_sample(2)
#   OPEN
# ds = pandas.read_pickle(os.path.join('Classifier', "balanced_dataset_48000_door_only.pickle"))

####################################################################################
# Evaluate
print('> Evaluation')
for i in range(len(ds)):
    evaluate(graph, ds.mels[i], ds.target[i])
