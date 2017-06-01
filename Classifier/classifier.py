import tensorflow as tf
import pandas
import librosa
import numpy as np
import os
import logging
import subprocess

OS_USER = subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]


def load_graph(frozen_graph_filename):
    """ Load graph/model to be used """
    logging.info('Loading frozen model-graph: ' + frozen_graph_filename)
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    logging.debug('Reading model file')
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    logging.debug('Importing graph')
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


def load_audio(audio_path):
    """ Take the input audio file and assemble it to be handled by the CNN """
    logging.info('Loading audio file: ' + audio_path)
    dataset = pandas.DataFrame(columns=['data', 'mels', 'mfcc'])

    y, sr = librosa.core.load(audio_path, sr=48000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    mapping = {
        'data': [y],
        'mels': [mel],
        'mfcc': [mfcc]
    }
    return dataset.append(pandas.DataFrame(mapping), ignore_index=True)


def predict(audio_path, graph, mapping):
    """ Predict the class of the given audio file according the provided model
    :param mapping: dictionary mapping the numeric output of the network to a label
    :param audio_path: path to audio file
    :param graph: already loaded tensor graph
    """
    logging.info('Prediction START')
    # Loading audio
    ds = load_audio(audio_path)

    logging.info('Audio and model loading DONE')
    ### Tensorflow
    # Prepare CNN input
    audio_feature = np.asanyarray(list(ds.mels[0].flatten()), dtype=np.float32)

    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

    # prediction
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: [audio_feature]  # < 45
        })

        logging.info('predictions:' + str(y_out))
        return mapping[y_out[0].argmax()]


@DeprecationWarning
def hierarchical_predict(audio_path):
    """ Predict the class of the given audio file according the whole hierarchical model """
    logging.info('Hierarchical prediction START')
    model_door_not_door = load_graph(os.path.join('Classifier', 'model', 'door_not_door', 'frozen', 'frozen_model.pb'))
    model_person_not_person = load_graph(
        os.path.join('Classifier', 'model', 'person_not_person', 'frozen', 'frozen_model.pb'))
    model_only_people = load_graph(os.path.join('Classifier', 'model', 'only_people', 'frozen', 'frozen_model.pb'))

    mapping_door_not_door = {
        0: 'nobody',
        1: 'door',
    }
    mapping_person_not_person = {
        2: 'person',
        0: 'exit',
        1: 'bell',
    }
    mapping_only_people = {
        0: 'alessio',
        1: 'andrea',
        2: 'debora',
        3: 'mamma',
        4: 'papa',
    }

    if predict(audio_path, model_door_not_door, mapping_door_not_door) == 'nobody':
        return 'nobody'
    else:
        intermediate_prediction = predict(audio_path, model_person_not_person, mapping_person_not_person)
        if intermediate_prediction in ['bell', 'exit']:
            return intermediate_prediction
        else:
            return predict(audio_path, model_only_people, mapping_only_people)
