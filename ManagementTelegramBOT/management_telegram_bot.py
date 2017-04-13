import sys
import time
import random
import datetime
import telepot
import requests
import json
import signal


# TODO change prints with true logging

#########################
# SYSTEM SIGNALS HANDLING

def signal_handler(signal, frame):
    """ Handles SIGINT, KILL signal [CTRL+C] notifying each client about the BOT closure. """
    print('Closing BOT...')
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

    print('Got command: %s' % command)
    print('From ChatID: %s' % chat_id)

    if chat_id not in data['known_clients']:
        bot.sendMessage(chat_id, 'vai via!')
        return

    if command == '/get_ip' or command == '/get_ip@RaspSemBot':
        bot.sendMessage(chat_id, get_ip())
    else:
        bot.sendMessage(chat_id, 'Comando non riconoscuto')


# START UP

data = json.load(open('credentials.json'))

bot = telepot.Bot(data['telegram_bot'])
data['known_clients'] = set(data['known_clients'])

for client in data['known_clients']:
    bot.sendMessage(client, wake_up())

for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
    signal.signal(sig, signal_handler)

# STARTING WAITING CYCLE

bot.message_loop(handle)
print('I am listening...')

while 1:
    time.sleep(10)
