from flask import Flask, render_template, request
import joblib
# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

#Basic Pages - 

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/about', methods = ['GET', 'POST'])
def about():
    return render_template("about.html")

@app.route('/contact', methods = ['GET', 'POST'])
def contact():
    return render_template("contact.html")

#Language Choose -

@app.route('/lang-choose/happy', methods = ['GET', 'POST'])
def lang_choose_happy():
    return render_template("/lang-choose/happy.html")

@app.route('/lang-choose/sad', methods = ['GET', 'POST'])
def lang_choose_sad():
    return render_template("/lang-choose/sad.html")

@app.route('/lang-choose/fear', methods = ['GET', 'POST'])
def lang_choose_fear():
    return render_template("/lang-choose/fear.html")

@app.route('/lang-choose/broken', methods = ['GET', 'POST'])
def lang_choose_broken():
    return render_template("/lang-choose/broken.html")

@app.route('/lang-choose/angry', methods = ['GET', 'POST'])
def lang_choose_angry():
    return render_template("/lang-choose/angry.html")

@app.route('/lang-choose/gratitude', methods = ['GET', 'POST'])
def lang_choose_gratitude():
    return render_template("/lang-choose/gratitude.html")

@app.route('/lang-choose/romantic', methods = ['GET', 'POST'])
def lang_choose_romantic():
    return render_template("/lang-choose/romantic.html")

@app.route('/lang-choose/boredom', methods = ['GET', 'POST'])
def lang_choose_boredom():
    return render_template("/lang-choose/boredom.html")

@app.route('/lang-choose/calm', methods = ['GET', 'POST'])
def lang_choose_calm():
    return render_template("/lang-choose/calm.html")

#Redirect to selected Language Song - 

@app.route('/lang/hindi/happy', methods = ['GET', 'POST'])
def song_hindi_happy():
    return render_template("/lang/hindi/happy.html")

@app.route('/lang/eng/happy', methods = ['GET', 'POST'])
def song_eng_happy():
    return render_template("/lang/eng/happy.html")

@app.route('/lang/hindi/sad', methods = ['GET', 'POST'])
def song_hindi_sad():
    return render_template("/lang/hindi/sad.html")

@app.route('/lang/eng/sad', methods = ['GET', 'POST'])
def song_eng_sad():
    return render_template("/lang/eng/sad.html")

@app.route('/lang/hindi/fear', methods = ['GET', 'POST'])
def song_hindi_fear():
    return render_template("/lang/hindi/fear.html")

@app.route('/lang/eng/fear', methods = ['GET', 'POST'])
def song_eng_fear():
    return render_template("/lang/eng/fear.html")

@app.route('/lang/hindi/broken', methods = ['GET', 'POST'])
def song_hindi_broken():
    return render_template("/lang/hindi/broken.html")

@app.route('/lang/eng/broken', methods = ['GET', 'POST'])
def song_eng_broken():
    return render_template("/lang/eng/broken.html")

@app.route('/lang/hindi/angry', methods = ['GET', 'POST'])
def song_hindi_angry():
    return render_template("/lang/hindi/angry.html")

@app.route('/lang/eng/anrgy', methods = ['GET', 'POST'])
def song_eng_anrgy():
    return render_template("/lang/eng/anrgy.html")

@app.route('/lang/hindi/gratitude', methods = ['GET', 'POST'])
def song_hindi_gratitude():
    return render_template("/lang/hindi/gratitude.html")

@app.route('/lang/eng/gratitude', methods = ['GET', 'POST'])
def song_eng_gratitude():
    return render_template("/lang/eng/gratitude.html")

@app.route('/lang/hindi/romantic', methods = ['GET', 'POST'])
def song_hindi_romantic():
    return render_template("/lang/hindi/romantic.html")

@app.route('/lang/eng/romantic', methods = ['GET', 'POST'])
def song_eng_romantic():
    return render_template("/lang/eng/romantic.html")

@app.route('/lang/hindi/boredom', methods = ['GET', 'POST'])
def song_hindi_boredom():
    return render_template("/lang/hindi/boredom.html")

@app.route('/lang/eng/boredom', methods = ['GET', 'POST'])
def song_eng_boredom():
    return render_template("/lang/eng/boredom.html")

@app.route('/lang/hindi/calm', methods = ['GET', 'POST'])
def song_hindi_calm():
    return render_template("/lang/hindi/calm.html")

@app.route('/lang/eng/calm', methods = ['GET', 'POST'])
def song_eng_calm():
    return render_template("/lang/eng/calm.html")

#Emotion Detection Model -

@app.route('/emotionDetection', methods = ['GET', 'POST'])
def emotion_detection():
    # Utils
    pipe_lr = joblib.load(open("models/emotion_detection_in_text_pipe_lr.pkl", "rb"))

    # Fxn
    def predict_emotions(txt):
        results = pipe_lr.predict([txt])
        return results[0]


    raw_text = request.form.get("emotionText")
    if raw_text:
        prediction = predict_emotions(raw_text)
        prediction = prediction.capitalize()

        return render_template('emotion_detection.html', emoPred=prediction)
    return render_template('emotion_detection.html')


if __name__ == "__main__":
    app.run(debug=True)
