#!/usr/bin/env python3

import time
import logging
import os
import sqlite3
import csv
import subprocess
import json
from collections import Counter

from RaspberryScripts import sound_record
from ManagementTelegramBOT import management_telegram_bot as telegram_bot
from Classifier import classifier

OS_USER = subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]
CONFIG = os.path.join(os.path.abspath(os.sep), 'home', OS_USER, 'SmartSlam', 'config.json')


def db_save_access(db_path, audio_path, label='default'):
    """ Create a new entry into DB of the given audio or update it if already existing. """
    logging.info('Insert record into DB...')
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        name = os.path.basename(audio_path)[:-4]
        timestamp = name.strip('sample-')
        c.execute('INSERT OR REPLACE INTO `accesses`(`timestamp`,`name`,`label`,`path`) VALUES(?,?,?,?)',
                  (timestamp, name, label, audio_path))
        conn.commit()


def csv_save_access(csv_path, audio_path, label='default'):
    """ Create a new entry into CSV label file of given audio"""
    logging.info('Insert record into CSV...')
    with open(csv_path, 'a', newline='') as csv_file:
        name = os.path.basename(audio_path)[:-4]
        csv.writer(csv_file).writerow([name, label])


def main():
    os.chdir(os.path.join(os.path.abspath(os.sep), 'home', OS_USER, 'SmartSlam'))
    log_dir = os.path.join(os.getcwd(), 'LOG')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'INFO.log'),
                        level=logging.INFO,
                        format='%(asctime)-15s '
                               '%(levelname)s '
                               '--%(filename)s-- '
                               '%(message)s')

    logging.info('START #############################################################')

    ### LOAD SETTINGS
    logging.info('Setting Up...')
    with open(CONFIG) as file:
        config = json.load(file)
    # DB
    db_sqlite_path = config['db_sqlite_path']
    csv_path = config['csv_path']

    # CNN
    nn_model_path = config['nn_model_path']
    nn_model_mapping = {int(key): config['nn_model_mapping'][key] for key in config['nn_model_mapping']}
    nn_model_graph = classifier.load_graph(nn_model_path)

    logging.info('Setting Up...COMPLETE')

    # Working cycle
    logging.info('Starting application loop')
    while True:
        logging.info('#########################################################')
        logging.info('Waiting for PIR detection...')
        # Wait PIR
        sound_record.wait_pir()
        # Turn LED ON
        sound_record.led_on()
        # Sound Record
        audio_path = sound_record.record()
        # save DB
        db_save_access(db_sqlite_path, audio_path)
        # save CSV
        csv_save_access(csv_path, audio_path)

        ### CHECKPOINT 1
        # bot notification: only name
        # telegram_bot.notify_sample_audio(audio_path)

        logging.info('Classifying new record...')
        # Classification
        prediction, probabilities = classifier.predict(audio_path, nn_model_graph, nn_model_mapping)
        # save DB
        db_save_access(db_sqlite_path, audio_path, prediction)
        # save CSV
        # TODO

        logging.info('BOT Sending notification of new record classification...')
        # bot notification: name and prediction
        telegram_bot.notify_sample_audio(audio_path, (prediction, probabilities))

        # bot notification: name, prediction and audio file
        # telegram_bot.notify_sample_audio(audio_path, (prediction, probabilities), 30)

        # Turn LED OFF
        sound_record.led_off()
        logging.info('Process accomplished, starting new cycle')
        logging.info('#########################################################')

        time.sleep(0.1)


if __name__ == "__main__":
    main()
