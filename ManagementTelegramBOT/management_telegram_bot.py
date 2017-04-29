#!/usr/bin/env python3

import sys
import time
import telepot
import requests
import json
import signal
import logging
import os
import subprocess
import threading


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
    ip = get_ip()
    msg = 'Goooood morning Vietnam! Find me at: ' + str(ip)
    return msg


def get_user():
    """ Get the user who is running the script """
    # logging.info('Getting user...') #  Not possible because the command is run before log configuration
    return subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]


def notify_audio_sample(sample_name, sample_path, duration):
    """ Notify the users of a new audio record """
    logging.info('New audio sample')

    msg = 'New audio sample: ' + sample_name
    file = open(sample_path, 'rb')
    for user in data['users']:
        logging.info('Sending message notification')
        bot.sendMessage(user, msg)

        logging.info('Sending audio file in another thread')
        threading.Thread(target=bot.sendAudio, args=(user, file, None, duration)).start()

        # TODO select label to apply


#########################
# BOT COMMANDS
def get_ip():
    """ Returns the global IP of the network the raspberry is connected to
    :return: global IP
    """
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
    chat_id = msg['chat']['id']
    command = msg['text']

    logging.info('Got command: ' + str(command) + ' From ChatID: ' + str(chat_id))

    if chat_id not in data['known_clients']:
        bot.sendMessage(chat_id, 'vai via!')
        return

    if command == '/get_ip' or command == '/get_ip@RaspSemBot':
        bot.sendMessage(chat_id, get_ip())
    else:
        bot.sendMessage(chat_id, 'Comando non riconoscuto')


#####################
# SETUP

os.chdir(os.path.join(os.path.abspath(os.sep), 'home', get_user(), 'SmartSlam', 'ManagementTelegramBOT'))

data = json.load(open('credentials.json'))
data['known_clients'] = set(data['known_clients'])
bot = telepot.Bot(data['telegram_bot'])


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

    # BOT
    logging.info('Setting up Bot...')

    wake_up_msg = wake_up()
    for client in data['known_clients']:
        bot.sendMessage(client, wake_up_msg)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
        signal.signal(sig, signal_handler)

    # STARTING WAITING CYCLE
    logging.info('STARTING LISTENING loop')

    bot.message_loop(handle)
    while 1:
        time.sleep(10)


if __name__ == "__main__":
    main()
