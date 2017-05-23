import tensorflow as tf
import pandas
import os
import csv
import librosa
import random
import numpy as np


def load_full_dataset():
    """ Load all the samples and the relative labels

    :return: pandas DataFrame
    """
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

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            if filename[:-4] in samples:
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


def load_random_sample():
    """ Loads a single audio sample """
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

    while True:
        filename = random.sample(list(samples), 1)[0]
        if samples[filename] == 'default':
            continue
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


def dataset_to_pickle(dataset, output_path):
    """ Save the whole dataset as pickle file
    :param dataset: pandas DataFrame
    :param output_path: path adn file name of output
    """
    pandas.to_pickle(dataset, output_path)


def freeze_graph(model_folder):
    """ froze a saved model into a self contained file"""
    print('>>> freezing model')
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = os.path.abspath(model_folder)
    # output_graph = os.path.join(absolute_model_folder, 'frozen_model.pb')
    output_graph = os.path.join('C:\\', 'Users', 'Alessio', 'Desktop', 'LOG', 'frozen_model', 'frozen_model.pb')

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "train_prediction"

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
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


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


def restore():
    """ Load a ready-to-be-used Tensor NN """
    # 1 Retrieve saved graph
    pass
    # 2 Restore the weights inside the Session
    pass
    # 3 Remove all metadata useless for inference
    pass
    # 4 Save it to the disk
    pass


def evaluate(graph, in_audio):
    """ Takes the input audio file/feature and classify it against the model """

    audio_feature = np.vstack(in_audio.mels[0].flatten()).reshape(in_audio.shape[0], 2813, 128, 1).astype(np.float32)

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/train_prediction:0')

    mapping = {
        'default': 0,
        'alessio': 1,
        'andrea': 1,
        'debora': 1,
        'mamma': 1,
        'papa': 1,
        'nobody': 0,
        'exit': 1,
        'bell': 1,
    }
    true_result = mapping[in_audio.target[0]]

    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        y_out = sess.run(
            y,
            feed_dict={
                x: audio_feature
            }
        )
        print(y_out)  # [[ False ]] Yay, it works!
        if y_out[0].argmax()==true_result:
            print('Result: CORRECT')
        else:
            print('Result: WRONG')
####################################################################################
# Load dataset
print('> Loading full dataset')

#   CREATE
# ds = load_full_dataset()
#   OPEN
# ds = pandas.read_pickle(os.path.join('Classifier', "full_dataset_48000_no_default.pickle"))
#   WRITE FILE
# dataset_to_pickle(ds, os.path.join('Classifier', "full_dataset_48000_no_default.pickle"))

####################################################################################
# Load model
# print('> Loading model')

# model_definition = restore_model(ds.shape[0])
# sess = tf.Session()
# saver = tf.train.Saver()
# saver.restore(sess, os.path.join('Classifier', 'model', 'model_door_only.ckpt'))

####################################################################################
# Evaluate

# for data, target, mels, mfcc in ds:
#     mel_flatten = mels.flatten()
#     prediction = str()
#     sess.run(prediction, feed_dict={x: input})
#     print('data-IN: ' + str(data) + ' | eval: ' + str(prediction) + ' | real: ' + str(target))

# freeze_graph('.\Classifier\model')

ds = load_random_sample()

graph = load_graph('C:\\Users\Alessio\Desktop\LOG\\frozen_model\\frozen_model.pb')

# evaluate(graph, ds.mels[0])
evaluate(graph, ds)
