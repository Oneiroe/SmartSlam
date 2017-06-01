# SmartSlam
Door "slam" sound recognizer using neural networks and RaspberryPi

Idea : "when someone familiar accesses your home it is usually possible to recognize him/her by the sounds produced. 
Can it be done by a machine too?"

## Overview

Devices:
- Raspberry Pi 3 Model B (OS: Raspbian GNU/Linux 8.0 jessie)
- High sensitive environmental MONO microphone
- PIR (Passive InfraRed) motion sensor

The PIR sensor placed outside the door is the recording trigger.

Sound recording, fixed to 30sec, is performed near door entrance at a sample rate of 48000 Hz.

The majority of captured file has been manually labelled to train a **Convolutional Neural Network** using
[Google TensorFlow framework](https://www.tensorflow.org/); 
Training is the only operation performed outside the Raspberry because of computational power limits. 
Once trained, a file containing the resulting model has been put into the Raspberry to make on-device prediction.
The model generated recognize this 8 sound category: 
Nobody (just people passing outside the door), 
Exit (someone exit house), 
Bell (someone ring the bell),
Person 1-5 (test house people).
Overall accuracy reach ~60%, rising to ~80% if we consider all the people as a single category.

A [Telegram chat bot](https://core.telegram.org/bots) is used to instantly notify users of new accesses 
(*timestamp*,*classification* and *audio file*), 
while through a webUI ([flask](http://flask.pocoo.org/)) permits to consult​ the accesses log and refine the classifications done.

## Disclaimer

This project has been developed for "Seminars in Software and services for the information society" class 
of Master of Science in Engineering in Computer Science from Sapienza University of Rome.

Dataset used will not be distributed as being composed of privacy-sensitive data.

