import pandas
import os
import logging
import json
import cnn_tensorflow
import pickle

with open('configurations.json') as file:
    CONFIGURATIONS = json.load(file)


def build_dataset(name, mapping, discriminant_targets, label_path, samples_dir_path):
    dataset_path = os.path.join('model', name, 'dataset', 'balanced_dataset_' + name + '.pickle')
    if os.path.exists(dataset_path):
        ds = pandas.read_pickle(dataset_path)
    else:
        ds = cnn_tensorflow.create_balanced_dataset(mapping,
                                                    label_path,
                                                    samples_dir_path,
                                                    discriminant_targets)
        cnn_tensorflow.save_dataset(
            ds,
            dataset_path
        )
    return ds


def main():
    log_dir = os.path.join(os.getcwd(), 'LOG')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'INFO.log'),
                        level=logging.INFO,
                        format='%(asctime)-15s '
                               '%(levelname)s '
                               '--%(filename)s-- '
                               '%(message)s')

    label_source = 'csv'
    label_source_key_map = {'csv': 'csv_path', 'sqlite': 'db_sqlite_path'}

    print('''
TRAINER TOOL
Interactive process to train a model
!!!!!!!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  These operation may invalidate the saved model, use with caution
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MEMO
-   CSV has only manually labelled or verified data, while the SQLite DB contains also auto-labelled data
-   To A) rebuild a dataset B) train from zero a model,
    remove manually the folder containing the dataset/model before starting the process.
    ''')

    while True:
        i = input('''
##########################################
q) Quit

0) Choose labels source [csv | sqlite] [current: ''' + label_source + ''']
1) Train model ALL
2) Train model DOOR_NOT_DOOR
3) Train model PERSON_NOT_PERSON
4) Train model ONLY_PEOPLE
##########################################

Select...''')

        if i is 'q':
            print('Hope you didn\'t destroy anything... Bye Bye')
            break
        elif i is '0':
            i0 = ''
            while i0 not in {'csv', 'sqlite'}:
                i0 = input('Choose labels source [csv | sqlite]:')
            label_source = i0
            print('Label source: ' + label_source)

        elif i is '1':
            print('Train model ALL')
            name = 'all'
            mapping = {
                'nobody': 0,
                'alessio': 1,
                'andrea': 2,
                'debora': 3,
                'mamma': 4,
                'papa': 5,
                'exit': 6,
                'bell': 7,
            }
            discriminant_targets = [1, 2, 3, 4, 5]
            print('DATASET')
            ds = build_dataset(name,
                               mapping,
                               discriminant_targets,
                               CONFIGURATIONS[label_source_key_map[label_source]],
                               CONFIGURATIONS['samples_dir_path'])
            print("This is the error rate if we always guess the majority: %.2f" % (
                1 - max(
                    ds[ds["target"] == 'nobody'].index.size,
                    ds[ds["target"] == 'andrea'].index.size,
                    ds[ds["target"] == 'exit'].index.size,
                    ds[ds["target"] == 'debora'].index.size,
                    ds[ds["target"] == 'alessio'].index.size,
                    ds[ds["target"] == 'mamma'].index.size,
                    ds[ds["target"] == 'papa'].index.size,
                    ds[ds["target"] == 'bell'].index.size
                ) / (float)(ds.index.size)))
            print('TRAINING')
            cnn_tensorflow.train(ds, name, mapping)

        elif i is '2':
            print('Train model DOOR_NOT_DOOR')
            name = 'door_not_door'
            mapping = {
                'nobody': 0,
                'alessio': 1,
                'andrea': 1,
                'debora': 1,
                'mamma': 1,
                'papa': 1,
                'exit': 1,
                'bell': 1,
            }
            discriminant_targets = []
            print('DATASET')
            ds = build_dataset(name,
                               mapping,
                               discriminant_targets,
                               CONFIGURATIONS[label_source_key_map[label_source]],
                               CONFIGURATIONS['samples_dir_path'])
            print("This is the error rate if we always guess the majority: %.2f" % (
                1 - max(
                    ds[ds["target"] == 'nobody'].index.size,
                    ds[ds["target"] == 'andrea'].index.size +
                    ds[ds["target"] == 'exit'].index.size +
                    ds[ds["target"] == 'debora'].index.size +
                    ds[ds["target"] == 'alessio'].index.size +
                    ds[ds["target"] == 'mamma'].index.size +
                    ds[ds["target"] == 'papa'].index.size +
                    ds[ds["target"] == 'bell'].index.size
                ) / (float)(ds.index.size)))
            print('TRAINING')
            cnn_tensorflow.train(ds, name, mapping)

        elif i is '3':
            print('Train model PERSON_NOT_PERSON')
            name = 'person_not_person'
            mapping = {
                'alessio': 2,
                'andrea': 2,
                'debora': 2,
                'mamma': 2,
                'papa': 2,
                'exit': 0,
                'bell': 1,
            }
            discriminant_targets = [2]
            print('DATASET')
            ds = build_dataset(name,
                               mapping,
                               discriminant_targets,
                               CONFIGURATIONS[label_source_key_map[label_source]],
                               CONFIGURATIONS['samples_dir_path'])
            print("This is the error rate if we always guess the majority: %.2f" % (
                1 - max(
                    ds[ds["target"] == 'andrea'].index.size +
                    ds[ds["target"] == 'debora'].index.size +
                    ds[ds["target"] == 'alessio'].index.size +
                    ds[ds["target"] == 'mamma'].index.size +
                    ds[ds["target"] == 'papa'].index.size,
                    ds[ds["target"] == 'exit'].index.size,
                    ds[ds["target"] == 'bell'].index.size
                ) / (float)(ds.index.size)))
            print('TRAINING')
            cnn_tensorflow.train(ds, name, mapping)

        elif i is '4':
            print('Train model ONLY_PEOPLE')
            name = 'only_people'
            mapping = {
                'alessio': 0,
                'andrea': 1,
                'debora': 2,
                'mamma': 3,
                'papa': 4,
            }
            discriminant_targets = [2]
            print('DATASET')
            ds = build_dataset(name,
                               mapping,
                               discriminant_targets,
                               CONFIGURATIONS[label_source_key_map[label_source]],
                               CONFIGURATIONS['samples_dir_path'])
            print("This is the error rate if we always guess the majority: %.2f" % (
                1 - max(
                    ds[ds["target"] == 'andrea'].index.size,
                    ds[ds["target"] == 'debora'].index.size,
                    ds[ds["target"] == 'alessio'].index.size,
                    ds[ds["target"] == 'mamma'].index.size,
                    ds[ds["target"] == 'papa'].index.size,
                ) / (float)(ds.index.size)))
            print('TRAINING')
            cnn_tensorflow.train(ds, name, mapping)
        else:
            print('Command not recognized, try again')


if __name__ == "__main__":
    main()
