#!/usr/bin/env python3

import time
import RPi.GPIO as GPIO
import json
import logging
import os
import subprocess

# USER who is running the script
OS_USER = subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]
SETTINGS = os.path.join(os.path.abspath(os.sep), 'home', OS_USER, 'SmartSlam', 'RaspberryScripts', 'settings.json')


#####################

def led_on():
    """ Turns on LED """
    logging.info('Turning LED ON...')
    with open(SETTINGS) as file:
        settings = json.load(file)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(settings['led_pin_board'], GPIO.OUT)

        GPIO.output(settings['led_pin_board'], GPIO.HIGH)


def led_off():
    """ Turns off LED """
    logging.info('Turning LED OFF...')
    with open(SETTINGS) as file:
        settings = json.load(file)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(settings['led_pin_board'], GPIO.OUT)

        GPIO.output(settings['led_pin_board'], GPIO.LOW)


def led_switch():
    """ Switch LED status """
    logging.info('Switching LED status...')
    with open(SETTINGS) as file:
        settings = json.load(file)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(settings['led_pin_board'], GPIO.OUT)

        if GPIO.input(settings['led_pin_board']) == 0:
            GPIO.output(settings['led_pin_board'], GPIO.HIGH)
        else:
            GPIO.output(settings['led_pin_board'], GPIO.LOW)


def record():
    """ Records sound, return absolute path to recorded file """
    logging.info('Setting up...')
    with open(SETTINGS) as file:
        settings = json.load(file)
        # Loading algorithm settings
        hw = str(settings['audio_device_hw_number'])
        duration = str(settings['sample_duration'])
        samples_dir = settings['samples_dir']

    #####################
    # Recording
    logging.info('Start RECORDING')
    base_name = 'sample'
    timestamp = time.strftime('%Y-%m-%d-%H%M%S')
    name = base_name + '-' + timestamp

    full_output_path = os.path.join(samples_dir, name + '.wav')

    subprocess.call([
        "arecord",
        "-f",  # Quality
        "cd",
        "-r",  # Sample Rate
        "48000",
        "-D",  # Device
        "hw:" + hw,
        "-d",  # Duration
        duration,
        full_output_path  # Output file
    ])

    # TODO logging recording response in case of error
    # TODO timeout in case of error
    logging.info('File: ' + full_output_path)
    logging.info('End RECORDING')

    return full_output_path


def wait_pir(DEBUG=False):
    """ Loops till PIR doesn't detect something"""
    # setting GPIO PINs
    with open(SETTINGS) as file:
        settings = json.load(file)

        logging.info('Setting GPIO PINS')
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(settings['pir_pin_board'], GPIO.IN)

    #####################
    # PIR cycle

    logging.info('Starting PIR waiting cycle...')

    if DEBUG:
        while input('insert 1 to start...') != '1':
            time.sleep(0.1)
    else:
        while GPIO.input(settings['pir_pin_board']) is not 1:
            time.sleep(0.1)

    logging.info('PIR detection')
    return
