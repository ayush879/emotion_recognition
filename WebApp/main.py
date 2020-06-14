#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt

# Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response

### Audio imports ###
from library.speech_emotion_recognition import *


# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

# Home page
# @app.route('/', methods=['GET'])
# def index():
#   return render_template('index.html')

# Audio Index
@app.route('/', methods=['GET'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 15sec to record the audio.")

    return render_template('audio.html', display_button=False)

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 16  # in sec
    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    #flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True)


# Audio Emotion Analysis
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'Audio_2DCNN.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')

    # Predict emotion in voice at each time step
    step = 1  # in sec
    sample_rate = 16000  # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(
        rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join(
        "static/js/db", "audio_emotions.txt"), mode='w')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions))
                    for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(),
                      columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db', 'audio_emotions_dist.txt'), sep=',')

    # Sleep
    time.sleep(0.5)

    return render_template('audio_dash.html', emo=major_emotion, prob=emotion_dist)


if __name__ == '__main__':
    app.run(debug=True)
