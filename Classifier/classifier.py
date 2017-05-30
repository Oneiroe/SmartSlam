import tensorflow as tf
import pandas
import librosa
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)


def load_graph(frozen_graph_filename):
    """ Load graph/model to be used """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    print('> Loading graph...')
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


def load_audio(audio_path):
    """ Take the input audio file and assemble it to be handled by the CNN """
    print('> loading audio file')
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


def predict(audio_path, frozen_model_path, mapping):
    """ Predict the class of the given audio file according the provided model
    :param mapping: dictionary mapping the numeric output of the network to a label
    :param audio_path: 
    :param frozen_model_path: 
    """
    print('> predict')
    # Loading audio
    ds = load_audio(audio_path)

    # Load Graph
    graph = load_graph(frozen_model_path)

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

        print('predictions:' + str(y_out))
        return mapping[y_out[0].argmax()]


def hierarchical_predict(audio_path):
    """ Predict the class of the given audio file according the whole hierarchical model """
    model_door_not_door = os.path.join('Classifier', 'model', 'door_not_door', 'freezed', 'frozen_model.pb')
    model_person_not_person = os.path.join('Classifier', 'model', 'person_not_person', 'freezed', 'frozen_model.pb')
    model_only_people = os.path.join('Classifier', 'model', 'only_people', 'freezed', 'frozen_model.pb')

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
