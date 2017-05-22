import pandas
import os
import librosa
import csv
import numpy as np
import random
import tensorflow as tf


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def load_balanced_dataset():
    """ Load samples and relative label randomly, such to have a balanced number of entity for each label 
    
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
            labels.setdefault(record[1], set()).add(record[0])
            samples[record[0]] = record[1]

    extra = {'label', 'nobody', 'default', 'exit'}
    max_samples = max(len(labels[x]) for x in labels.keys() - extra)
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
            labels.setdefault(record[1], set()).add(record[0])
            samples[record[0]] = record[1]

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            if filename[:-4] in samples:
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
    one_hot = np.zeros(2)
    one_hot[mapping[row]] = 1.0
    return one_hot


def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    print(data.get_shape())
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 2, 2, 1],
                        # Since stride is 50, the filters moves 50 frames each time. Therefore the shape becomes ceiling(11025/50)
                        padding='SAME')
    print(conv.get_shape())

    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    print(relu.get_shape())
    # Max pooling. The kernel size spec ksize also follows the layout of
    # the data.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    print("pool_shape: %s" % pool.get_shape())

    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 2, 2, 1],
                        padding='SAME')

    print("conv: %s" % conv.get_shape())

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          # max pool aggregates 2 units into 1, therefore the shape is halved again.
                          padding='SAME')

    print("pool: %s" % pool.get_shape())

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.

    pool_shape = pool.get_shape().as_list()

    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


def error_rate(predictions, labels):
    # We use argmax to convert prediction probabilities into 1-hot encoding and compare it against the labels
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = np.zeros([NUM_LABELS, NUM_LABELS], np.float32)

    bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    return error, confusions


####################################################################################
# Load dataset

#   CREATE
# ds = load_full_dataset()
#   OPEN
# ds = pandas.read_pickle(os.path.join('Classifier', "full_dataset_48000.pickle"))
#   WRITE FILE
# dataset_to_pickle(ds, os.path.join('Classifier', "full_dataset_48000.pickle"))

#   CREATE
# ds = load_balanced_dataset()
#   OPEN
ds = pandas.read_pickle(os.path.join('Classifier', "balanced_dataset_48000.pickle"))
#   WRITE FILE
# dataset_to_pickle(ds, os.path.join('Classifier', "balanced_dataset_48000.pickle"))

print("This is the error rate if we always guess the majority: %.2f" % (
    1 - max(
        ds[ds["target"] == 'andrea'].index.size,
        ds[ds["target"] == 'nobody'].index.size,
        ds[ds["target"] == 'default'].index.size,
        ds[ds["target"] == 'exit'].index.size,
        ds[ds["target"] == 'debora'].index.size,
        ds[ds["target"] == 'alessio'].index.size,
        ds[ds["target"] == 'mamma'].index.size,
        ds[ds["target"] == 'papa'].index.size,
        ds[ds["target"] == 'bell'].index.size,
    ) / (float)(ds.index.size)))

####################################################################################
# Divide the dataset in training/testing/validation sets

# dataset proportion
# 0.78 ~ 75 % - Train
# 0.14 ~ 15 % - Validation
# 0.07 ~ 10 % - Test
ds["one_hot_encoding"] = ds.target.apply(to1hot)

index_train = int(len(ds) * 0.75)
index_validation = index_train + int(len(ds) * 0.15)
index_test = index_validation + int(len(ds) * 0.10)

ds["mels_flatten"] = ds.mels.apply(lambda mels: mels.flatten())
train_data = ds[0:index_train]
validation_data = ds[index_train:index_validation]
test_data = ds[index_validation:]

# NOTE: train_data.shape = (136, 360064) -> 2813 * 128 = 360064
# TODO find this numbers programmatically
# X: data records
train_x = np.vstack(train_data.mels_flatten).reshape(train_data.shape[0], 2813, 128, 1).astype(np.float32)
# Y: label of the records
train_y = np.vstack(train_data["one_hot_encoding"])
train_size = train_y.shape[0]
validation_x = np.vstack(validation_data.mels_flatten).reshape(validation_data.shape[0], 2813, 128, 1).astype(
    np.float32)
validation_y = np.vstack(validation_data["one_hot_encoding"])
test_x = np.vstack(test_data.mels_flatten).reshape(test_data.shape[0], 2813, 128, 1).astype(np.float32)
test_y = np.vstack(test_data["one_hot_encoding"])

####################################################################################
# set variables to train, global variables, IN and OUT
BATCH_SIZE = int(train_data.shape[0] / 20)
NUM_CHANNELS = 1
NUM_LABELS = 2
INPUT_SHAPE = (2813, 128)
SEED = 42

# This node is where we feed a batch of the training data and labels at each training step
train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

# constants for validation and tests
validation_data_node = tf.constant(validation_x)
test_data_node = tf.constant(test_x)

conv1_weights = tf.Variable(
    # The first 3 elements defines the shape of the filter, the last one is the number of feature maps it outputs
    # This 1d filter only looks at a small contiguous chunk of audio signal (550 samples, ~550ms) (
    # if the data was an image then one would probably use a 2d (greyscale) or even 3d (color) filter
    # The size of the filter can be anything, as long as it is smaller than the input
    tf.truncated_normal([2, 8, 1, 32],  # Creating 32 feature maps.
                        stddev=0.1,
                        seed=SEED))
conv1_biases = tf.Variable(tf.zeros([32]))  # Each feature needs a bias for ReLU

conv2_weights = tf.Variable(
    tf.truncated_normal([30, 8, 32, 64],  # Creating 64 feature maps.
                        stddev=0.1,
                        seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(tf.truncated_normal([90112, 512], stddev=0.1, seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
####################################################################################
# build model
logits = model(train_data_node, True)

####################################################################################
# loss function

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node))
# variable_summaries(loss)
tf.summary.scalar('loss', loss)

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

####################################################################################
# Optimizer

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0)
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
    0.01,  # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    train_size,  # Decay step.
    0.99,  # Decay rate.
    staircase=True)

# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

# Predictions for the minibatch, validati
# on set and test set.
train_prediction = tf.nn.softmax(logits)
# We'll compute them only once in a while by calling their {eval()} method.
validation_prediction = tf.nn.softmax(model(validation_data_node))
test_prediction = tf.nn.softmax(model(test_data_node))

####################################################################################
# Training

# Create a new interactive session that we'll use in
# subsequent code cells.
s = tf.InteractiveSession()

# Use our newly created session as the default for
# subsequent operations.
s.as_default()

# TensorBoard nodes initialization
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join('Classifier','LOG'), s.graph)


# Initialize all the variables we defined above.
# tf.initialize_all_variables().run()
tf.global_variables_initializer().run()

offset = 0
# This code uses the entire training set instead of mini batches
for i in range(100):
    # Train over the first 1/4th of our training set.
    #     steps = int(train_size / BATCH_SIZE)
    # for step in xrange(steps):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    # offset = 0  # (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    offset = (i * BATCH_SIZE) % (train_size - BATCH_SIZE)
    if offset > train_size:
        offset = 0
    batch_data = train_x[offset:(offset + BATCH_SIZE), :, :, :]
    batch_labels = train_y[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the graph and fetch some of the nodes.
    _, l, lr, predictions = s.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)

    # Print out the loss periodically.
    if i % 20 == 0:
        error, _ = error_rate(predictions, batch_labels)
        print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
        print('Validation error: %.5f' % error_rate(validation_prediction.eval(), validation_y)[0])
        summary = s.run(merged, feed_dict=feed_dict)
        train_writer.add_summary(summary, i)

train_writer.close()
####################################################################################
# evaluation

test_error, confusions = error_rate(test_prediction.eval(), test_y)
print('Test error: %.5f' % test_error)

res = np.argmax(test_prediction.eval(), 1) == np.argmax(test_y, 1)
right = []
wrong = []
for i, v in enumerate(res.tolist()):
    if v:
        right.append(i)
    else:
        wrong.append(i)
print('right:' + str(len(right)))
print('wrong:' + str(len(wrong)))
