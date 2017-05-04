import pandas
import os
import librosa
import csv
import numpy as np


def load_labelled_dataset():
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
            y, sr = librosa.core.load(os.path.join(directory, filename), sr=48000, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            if filename[:-4] in samples:
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
    pandas.to_pickle(dataset, output_path)


# ds = load_labelled_dataset()
ds = pandas.read_pickle(os.path.join('Classifier', "my_dataset_48000_500.pickle"))
# dataset_to_pickle(ds, os.path.join('Classifier', "my_dataset_48000.pickle"))
