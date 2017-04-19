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
    ip = requests.get('http://showip.net').content
    msg = 'Goooood morning Vietnam! Find me at: ' + str(ip)
    return msg


def get_user():
    """ Get the user who is running the script """
    # logging.info('Getting user...') #  Not possible because the command is run before log configuration
    return subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]


#########################
# BOT COMMANDS
def get_ip():
    """ Returns the global IP of the network the raspberry is connected to
    :return: global IP
    """
    return requests.get('http://showip.net').content


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

# LOG setup
os.chdir(os.path.join('home', get_user(), 'SmartSlam', 'RaspberryScripts'))
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

data = json.load(open('credentials.json'))

bot = telepot.Bot(data['telegram_bot'])
data['known_clients'] = set(data['known_clients'])

for client in data['known_clients']:
    bot.sendMessage(client, wake_up())

for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
    signal.signal(sig, signal_handler)

# STARTING WAITING CYCLE
logging.info('STARTING LISTENING loop')

bot.message_loop(handle)
while 1:
    time.sleep(10)
