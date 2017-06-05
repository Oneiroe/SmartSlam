#!/usr/bin/env python3

import sys
import time
import telepot
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from telepot.loop import MessageLoop
import requests
import json
import signal
import logging
import os
import subprocess
import threading
import sqlite3
from collections import Counter

#####################
# SETUP

OS_USER = subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]
DATA = os.path.join(os.path.abspath(os.sep), 'home', OS_USER, 'SmartSlam', 'ManagementTelegramBOT', 'credentials.json')

with open(DATA) as file:
    data = json.load(file)
    data['known_clients'] = set(data['known_clients'])
    bot = telepot.Bot(data['telegram_bot'])


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


#########################
# SYSTEM SIGNALS HANDLING

def signal_handler(signal, frame):
    """ Handles SIGINT, KILL signal [CTRL+C] notifying each client about the BOT closure. """
    logging.info('Closing BOT...')
    for client in data['known_clients']:
        bot.sendMessage(client, 'Mi sto spegnendo...Vendicami!')
    sys.exit(0)


#########################
def wake_up():
    """ Launched every time the BOT is started

    :return: message to be shown containing the global IP
    """
    logging.info('Waking Up...')
    ip = str(get_ip(), 'utf-8')
    msg = 'Goooood morning Vietnam! Find me at: ' + ip
    return msg


def notify_sample_audio(sample_path, prediction='', duration=0):
    """ Notify the users of a new audio record, optionally send also class prediction and the audio file
    :type sample_path: audio sample path
    :type prediction: tuple (prediction, probabilities) where the first is a string and the second a Counter dictionary
    """
    logging.info('Notifying user of new audio sample (', sample_path, ')...')
    sample_name = os.path.basename(sample_path)[:-4]

    msg = 'New audio record!\nNAME: ' + sample_name
    if prediction:
        logging.info('Writing predictions into message...')
        msg += '\nCLASS: ' + prediction[0]
        msg += '\n------------------------------'
        msg += '\nDETAILS: '
        for i in prediction[1].most_common():
            msg += '\n' + "{:.2%}".format(i[1]) + ' ' + i[0]
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text='Rectify classification?',
                                                   callback_data='/rectify')]])

    for user in data['users']:
        logging.info('Sending message notification')
        if duration:
            with open(sample_path, 'rb') as bit_file:
                logging.info('Sending audio file in another thread')
                threading.Thread(target=bot.sendVoice, args=(user, bit_file, msg, duration)).start()
        else:
            # TODO like this if only msg is send without prediction errors arise
            bot.sendMessage(user, msg, reply_markup=keyboard)


#########################
# BOT COMMANDS
def get_ip():
    """ Returns the global IP of the network the raspberry is connected to """
    link = 'http://showip.net'
    max_attempts = 10  # Maximal number of attempts for an API call before giving up
    connection_timeout = 3  # Timeout before raise a timeout exception
    retry_timeout = 5  # Timeout before retry a request after receiving a 50x HTTP error

    logging.info('Retrieving Global IP...')
    for attempt in range(max_attempts):
        try:
            request = requests.get(link, timeout=connection_timeout)
            request.raise_for_status()  # Rise exception if response code different from 200
            ip = request.content
            return ip

        # In the event of a network problem (e.g. DNS failure, refused connection, etc)
        except requests.exceptions.ConnectionError as err:
            logging.warning(str(err) + ' -- line: ' + str(sys.exc_info()[-1].tb_lineno))
            time.sleep(retry_timeout)  # in sec
            continue

        # Triggered Timeout
        except requests.exceptions.Timeout as err:
            logging.warning(str(err) + ' -- line: ' + str(sys.exc_info()[-1].tb_lineno))
            time.sleep(retry_timeout)  # in sec
            continue

        # Response code different from 200
        except requests.exceptions.HTTPError as err:
            logging.warning(str(err) + ' -- line: ' + str(sys.exc_info()[-1].tb_lineno))
            if request.status_code > 500:
                # Probable connection error, wait and retry
                time.sleep(retry_timeout)  # in sec
                continue
            else:
                return 'Houston, We\'ve Got a Problem...Try again later'

        # Unknown ambiguous request error
        except requests.exceptions.RequestException as err:
            logging.warning(str(err) + ' -- line: ' + str(sys.exc_info()[-1].tb_lineno))
            return 'Houston, We\'ve Got a Problem...Try again later'

    logging.warning('Request MAX_ATTEMPTS reached')
    return 'Houston, We\'ve Got a Problem...Try again later'


#########################
# BOT LOGIC
def handle(msg):
    """ Reaction to an explicit command by the user """
    chat_id = msg['chat']['id']
    command = msg['text']

    logging.info('Got command: ' + str(command) + ' From ChatID: ' + str(chat_id))

    if chat_id not in data['known_clients']:
        bot.sendMessage(chat_id, 'vai via!')
        return

    if command == '/get_ip' or command == '/get_ip@RaspSemBot':
        ip = str(get_ip(), 'utf-8')
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text='Go to WebUI', url=ip)]]
        )
        bot.sendMessage(chat_id, ip, reply_markup=keyboard)
    else:
        bot.sendMessage(chat_id, 'Comando non riconoscuto')


def on_callback(msg):
    """  Reaction to an interaction with an InLineKeyboard """
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    logging.info('Got Callback Query:' + str(query_id) + ' Command:' + query_data + ' From ChatID:' + str(from_id))

    # SWITCH/CASE to identify the call and respond consequently
    if query_data == '/rectify':
        options = [i.split('% ')[1] for i in
                   msg['message']['text'].split('\n')[msg['message']['text'].split('\n').index('DETAILS: ') + 1:]]
        options_buttons = [[InlineKeyboardButton(text=i, callback_data='/rectify_' + i)] for i in options]
        options_buttons.append([InlineKeyboardButton(text='<< Go back', callback_data='/back')])
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=options_buttons)
        bot.editMessageReplyMarkup((from_id, msg['message']['message_id']), reply_markup=keyboard)
    elif '/rectify_' in query_data:
        rectified_classification = query_data[len('/rectify_'):]
        new = msg['message']['text'].split('\n')
        new[2] = 'UPDATED: ' + rectified_classification
        new_message = ''.join('%s\n' % i for i in new)

        sample_name = new[1][6:] + '.wav'
        db_save_access(os.path.join(data['db_path'], 'smartSlamDB.sqlite'),
                       os.path.join(data['db_path'], 'Samples', sample_name),
                       rectified_classification)

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text='Rectify classification?', callback_data='/rectify')]]
        )
        bot.editMessageText((from_id, msg['message']['message_id']), new_message, reply_markup=keyboard)

    elif query_data == '/back':
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text='Rectify classification?', callback_data='/rectify')]]
        )
        bot.editMessageReplyMarkup((from_id, msg['message']['message_id']), reply_markup=keyboard)


def main():
    # LOG setup
    os.chdir(os.path.join(os.path.abspath(os.sep), 'home', OS_USER, 'SmartSlam', 'ManagementTelegramBOT'))
    log_dir = os.path.join(os.getcwd(), 'LOG')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'INFO.log'),
                        level=logging.INFO,
                        format='%(asctime)-15s '
                               '%(levelname)s '
                               '--%(filename)s-- '
                               '%(message)s')

    # BOT
    logging.info('Setting up Bot...')

    wake_up_msg = wake_up()
    for client in data['known_clients']:
        bot.sendMessage(client, wake_up_msg)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
        signal.signal(sig, signal_handler)

    # STARTING WAITING CYCLE
    logging.info('STARTING LISTENING loop')

    MessageLoop(bot, {'chat': handle,
                      'callback_query': on_callback}).run_as_thread()
    while 1:
        time.sleep(10)


if __name__ == "__main__":
    main()
