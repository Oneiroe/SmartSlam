import pandas
import os
import logging
from Classifier import cnn_tensorflow


def main():
    # LOG setup
    log_dir = os.path.join(os.getcwd(), 'LOG')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'INFO.log'),
                        level=logging.INFO,
                        format='%(asctime)-15s '
                               '%(levelname)s '
                               '--%(filename)s-- '
                               '%(message)s')

    ####################################################################################
    # 0 - all model
    name = 'all'
    logging.info('0 - model: ' + name)
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
    dataset_path = os.path.join('Classifier', 'balanced_dataset_' + name + '.pickle')
    # Create&save/load dataset
    logging.info('1 - DATASET')
    if os.path.exists(dataset_path):
        ds = pandas.read_pickle(dataset_path)
    else:
        ds = cnn_tensorflow.create_balanced_dataset(mapping, [1, 2, 3, 4, 5])
        cnn_tensorflow.save_dataset(
            ds,
            dataset_path
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
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

    # train/retrain model
    logging.info('1 - TRAINING')
    cnn_tensorflow.train(ds, name, mapping)

    ####################################################################################
    # 1 - Door not door
    name = 'door_not_door'
    logging.info('1 - model: ' + name)
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
    dataset_path = os.path.join('Classifier', 'balanced_dataset_' + name + '.pickle')
    # Create&save/load dataset
    logging.info('1 - DATASET')
    if os.path.exists(dataset_path):
        ds = pandas.read_pickle(dataset_path)
    else:
        ds = cnn_tensorflow.create_balanced_dataset(mapping)
        cnn_tensorflow.save_dataset(
            ds,
            dataset_path
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
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

    # train/retrain model
    logging.info('1 - TRAINING')
    cnn_tensorflow.train(ds, name, mapping)

    ####################################################################################
    # 2 - exit, bell, person
    name = 'person_not_person'
    logging.info('2 - Model: ' + name)
    mapping = {
        'alessio': 2,
        'andrea': 2,
        'debora': 2,
        'mamma': 2,
        'papa': 2,
        'exit': 0,
        'bell': 1,
    }
    dataset_path = os.path.join('Classifier', 'balanced_dataset_' + name + '.pickle')
    # Create&save/load dataset
    logging.info('2 - DATASET')
    if os.path.exists(dataset_path):
        ds = pandas.read_pickle(dataset_path)
    else:
        ds = cnn_tensorflow.create_balanced_dataset(mapping, [2])
        cnn_tensorflow.save_dataset(
            ds,
            dataset_path
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds[ds["target"] == 'andrea'].index.size +
            ds[ds["target"] == 'debora'].index.size +
            ds[ds["target"] == 'alessio'].index.size +
            ds[ds["target"] == 'mamma'].index.size +
            ds[ds["target"] == 'papa'].index.size,
            ds[ds["target"] == 'exit'].index.size,
            ds[ds["target"] == 'bell'].index.size
        ) / (float)(ds.index.size)))

    # train/retrain model
    logging.info('2 - TRAINING')
    cnn_tensorflow.train(ds, name, mapping)

    ####################################################################################
    # 3 - people
    name = 'only_people'
    logging.info('3 - model: ' + name)
    mapping = {
        'alessio': 0,
        'andrea': 1,
        'debora': 2,
        'mamma': 3,
        'papa': 4,
    }
    dataset_path = os.path.join('Classifier', 'balanced_dataset_only_people.pickle')
    # Create&save/load dataset
    logging.info('3 - DATASET')
    if os.path.exists(dataset_path):
        ds = pandas.read_pickle(dataset_path)
    else:
        ds = cnn_tensorflow.create_balanced_dataset(mapping, [0, 1, 2, 3, 4])
        cnn_tensorflow.save_dataset(
            ds,
            dataset_path
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds[ds["target"] == 'andrea'].index.size,
            ds[ds["target"] == 'debora'].index.size,
            ds[ds["target"] == 'alessio'].index.size,
            ds[ds["target"] == 'mamma'].index.size,
            ds[ds["target"] == 'papa'].index.size,
        ) / (float)(ds.index.size)))

    # train/retrain model
    logging.info('3 - TRAINING')
    cnn_tensorflow.train(ds, name, mapping)


if __name__ == "__main__":
    main()
