import pandas
import os
import librosa
import csv
import random
from collections import Counter
from Classifier import cnn_tensorflow


def create_balanced_dataset(targets_mapping, discriminant_targets=[]):
    """ Load samples and relative label randomly, such to have a balanced number of entity for each label

    :param targets_mapping:
    :param discriminant_targets: is not -1, ds will have at least all the elements mapped to this target
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
            if record[1] not in targets_mapping.keys():
                continue
            labels.setdefault(record[1], set()).add(record[0])
            samples[record[0]] = record[1]

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


def main():
    ####################################################################################
    mapping_all = {
        'nobody': 0,
        'alessio': 1,
        'andrea': 2,
        'debora': 3,
        'mamma': 4,
        'papa': 5,
        'exit': 6,
        'bell': 7,
    }
    ####################################################################################
    # 1 - Door not door training
    mapping_door_not_door = {
        'nobody': 0,
        'alessio': 1,
        'andrea': 1,
        'debora': 1,
        'mamma': 1,
        'papa': 1,
        'exit': 1,
        'bell': 1,
    }
    name_door_not_door = 'door_not_door'
    dataset_path_door_not_door = os.path.join('Classifier', 'balanced_dataset_door_not_door.pickle')
    # Create&save/load dataset
    if os.path.exists(dataset_path_door_not_door):
        ds_door_not_door = pandas.read_pickle(dataset_path_door_not_door)
    else:
        ds_door_not_door = create_balanced_dataset(mapping_door_not_door)
        cnn_tensorflow.save_dataset(
            ds_door_not_door,
            dataset_path_door_not_door
        )
    # train/retrain model
    cnn_tensorflow.train(ds_door_not_door, name_door_not_door, mapping_door_not_door)

    ####################################################################################
    # 2 - exit, bell, person
    mapping_person_not_person = {
        'alessio': 2,
        'andrea': 2,
        'debora': 2,
        'mamma': 2,
        'papa': 2,
        'exit': 0,
        'bell': 1,
    }
    name_person_not_person = 'person_not_person'
    dataset_path_person_not_person = os.path.join('Classifier', 'balanced_dataset_person_not_person.pickle')
    # Create&save/load dataset
    if os.path.exists(dataset_path_person_not_person):
        ds_person_not_person = pandas.read_pickle(dataset_path_person_not_person)
    else:
        ds_person_not_person = create_balanced_dataset(mapping_person_not_person, [2])
        cnn_tensorflow.save_dataset(
            ds_person_not_person,
            dataset_path_person_not_person
        )
    # train/retrain model
    cnn_tensorflow.train(ds_person_not_person, name_person_not_person, mapping_person_not_person)
    ####################################################################################
    # 3 - people
    mapping_only_people = {
        'alessio': 0,
        'andrea': 1,
        'debora': 2,
        'mamma': 3,
        'papa': 4,
    }
    name_only_people = 'only_people'
    dataset_path_only_people = os.path.join('Classifier', 'balanced_dataset_only_people.pickle')
    # Create&save/load dataset
    if os.path.exists(dataset_path_only_people):
        ds_only_people = pandas.read_pickle(dataset_path_only_people)
    else:
        ds_only_people = create_balanced_dataset(mapping_only_people, [0, 1, 2, 3, 4])
        cnn_tensorflow.save_dataset(
            ds_only_people,
            dataset_path_only_people
        )
    # train/retrain model
    cnn_tensorflow.train(ds_only_people, name_only_people, mapping_only_people)


if __name__ == "__main__":
    main()
