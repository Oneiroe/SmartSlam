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
    logging.info('1 - Door not door model')
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
    name_all = 'all'
    dataset_path_all = os.path.join('Classifier', 'balanced_dataset_all.pickle')
    # Create&save/load dataset
    logging.info('1 - DATASET')
    if os.path.exists(dataset_path_all):
        ds_all = pandas.read_pickle(dataset_path_all)
    else:
        ds_all = cnn_tensorflow.create_balanced_dataset(mapping_all, [1, 2, 3, 4, 5])
        cnn_tensorflow.save_dataset(
            ds_all,
            dataset_path_all
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds_all[ds_all["target"] == 'nobody'].index.size,
            ds_all[ds_all["target"] == 'andrea'].index.size,
            ds_all[ds_all["target"] == 'exit'].index.size,
            ds_all[ds_all["target"] == 'debora'].index.size,
            ds_all[ds_all["target"] == 'alessio'].index.size,
            ds_all[ds_all["target"] == 'mamma'].index.size,
            ds_all[ds_all["target"] == 'papa'].index.size,
            ds_all[ds_all["target"] == 'bell'].index.size
        ) / (float)(ds_all.index.size)))

    # train/retrain model
    logging.info('1 - TRAINING')
    cnn_tensorflow.train(ds_all, name_all, mapping_all)

    ####################################################################################
    # 1 - Door not door
    logging.info('1 - Door not door model')
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
    logging.info('1 - DATASET')
    if os.path.exists(dataset_path_door_not_door):
        ds_door_not_door = pandas.read_pickle(dataset_path_door_not_door)
    else:
        ds_door_not_door = cnn_tensorflow.create_balanced_dataset(mapping_door_not_door)
        cnn_tensorflow.save_dataset(
            ds_door_not_door,
            dataset_path_door_not_door
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds_door_not_door[ds_door_not_door["target"] == 'nobody'].index.size,
            ds_door_not_door[ds_door_not_door["target"] == 'andrea'].index.size +
            ds_door_not_door[ds_door_not_door["target"] == 'exit'].index.size +
            ds_door_not_door[ds_door_not_door["target"] == 'debora'].index.size +
            ds_door_not_door[ds_door_not_door["target"] == 'alessio'].index.size +
            ds_door_not_door[ds_door_not_door["target"] == 'mamma'].index.size +
            ds_door_not_door[ds_door_not_door["target"] == 'papa'].index.size +
            ds_door_not_door[ds_door_not_door["target"] == 'bell'].index.size
        ) / (float)(ds_door_not_door.index.size)))

    # train/retrain model
    logging.info('1 - TRAINING')
    cnn_tensorflow.train(ds_door_not_door, name_door_not_door, mapping_door_not_door)

    ####################################################################################
    # 2 - exit, bell, person
    logging.info('2 - Person not person Model')

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
    logging.info('2 - DATASET')
    if os.path.exists(dataset_path_person_not_person):
        ds_person_not_person = pandas.read_pickle(dataset_path_person_not_person)
    else:
        ds_person_not_person = cnn_tensorflow.create_balanced_dataset(mapping_person_not_person, [2])
        cnn_tensorflow.save_dataset(
            ds_person_not_person,
            dataset_path_person_not_person
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds_person_not_person[ds_person_not_person["target"] == 'andrea'].index.size +
            ds_person_not_person[ds_person_not_person["target"] == 'debora'].index.size +
            ds_person_not_person[ds_person_not_person["target"] == 'alessio'].index.size +
            ds_person_not_person[ds_person_not_person["target"] == 'mamma'].index.size +
            ds_person_not_person[ds_person_not_person["target"] == 'papa'].index.size,
            ds_person_not_person[ds_person_not_person["target"] == 'exit'].index.size,
            ds_person_not_person[ds_person_not_person["target"] == 'bell'].index.size
        ) / (float)(ds_person_not_person.index.size)))

    # train/retrain model
    logging.info('2 - TRAINING')
    cnn_tensorflow.train(ds_person_not_person, name_person_not_person, mapping_person_not_person)
    ####################################################################################
    # 3 - people
    logging.info('3 - people model')

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
    logging.info('3 - DATASET')
    if os.path.exists(dataset_path_only_people):
        ds_only_people = pandas.read_pickle(dataset_path_only_people)
    else:
        ds_only_people = cnn_tensorflow.create_balanced_dataset(mapping_only_people, [0, 1, 2, 3, 4])
        cnn_tensorflow.save_dataset(
            ds_only_people,
            dataset_path_only_people
        )

    logging.info("This is the error rate if we always guess the majority: %.2f" % (
        1 - max(
            ds_person_not_person[ds_person_not_person["target"] == 'andrea'].index.size,
            ds_person_not_person[ds_person_not_person["target"] == 'debora'].index.size,
            ds_person_not_person[ds_person_not_person["target"] == 'alessio'].index.size,
            ds_person_not_person[ds_person_not_person["target"] == 'mamma'].index.size,
            ds_person_not_person[ds_person_not_person["target"] == 'papa'].index.size,
        ) / (float)(ds_person_not_person.index.size)))

    # train/retrain model
    logging.info('3 - TRAINING')
    cnn_tensorflow.train(ds_only_people, name_only_people, mapping_only_people)


if __name__ == "__main__":
    main()
