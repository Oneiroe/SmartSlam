import pandas
import os
from Classifier import cnn_tensorflow


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
        ds_door_not_door = cnn_tensorflow.create_balanced_dataset(mapping_door_not_door)
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
        ds_person_not_person = cnn_tensorflow.create_balanced_dataset(mapping_person_not_person, [2])
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
        ds_only_people = cnn_tensorflow.create_balanced_dataset(mapping_only_people, [0, 1, 2, 3, 4])
        cnn_tensorflow.save_dataset(
            ds_only_people,
            dataset_path_only_people
        )
    # train/retrain model
    cnn_tensorflow.train(ds_only_people, name_only_people, mapping_only_people)


if __name__ == "__main__":
    main()
