import os.path
import time
from os import mkdir
from pathlib import Path

from flask import Flask, request, make_response

import numpy
import speech_recognition
import numpy as np
import python_speech_features as mfcc
import pickle
from scipy.io import wavfile
from scipy.io.wavfile import read

from sklearn import preprocessing

from sklearn.mixture import GaussianMixture

r = speech_recognition.Recognizer()

app = Flask(__name__)


def getAudio():
    with speech_recognition.Microphone() as source:
        print('Say something...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        return audio


def getText(sound):
    try:
        text = r.recognize_google(sound, language="ru").lower()
        print('You said: ' + text + '\n')
    # loop back to continue to listen for commands if unrecognizable speech is received
    except speech_recognition.UnknownValueError:
        print('....')
        text = getText()
    return text


@app.route('/voiceAPI/api/v1.0/addVoice', methods=['POST'])
def addWordToTrain():
    if request.files:
        source = request.files['file']
        Name = request.headers['Login']

        paths = sorted(Path('\\training_set\\').glob(Name + '*_sample?.wav'))
        iter = len(paths) / 6 + 1

        if not os.path.exists(os.getcwd() + "\\training_set"):
            mkdir(os.getcwd() + "\\training_set")
        files = os.listdir(path=os.getcwd() + "\\training_set")

        source.save(os.getcwd() + "\\training_set\\" + Name + iter + "_sample" + str(len(files) + 1) + ".wav")
        OUTPUT_FILENAME = Name + str(iter) + "_sample" + str(len(files) + 1) + ".wav"

        file = open(os.getcwd() + "\\testing_set_addition.txt", 'a')
        file.write(str(OUTPUT_FILENAME) + '\n')


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines
    delta to make it 40 dim feature vector"""

    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True, nfft=16800)

    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined


@app.route('/voiceAPI/api/v1.0/train_model', methods=['GET'])
def train_model():
    source = os.getcwd() + "\\training_set\\"
    if not os.path.exists(os.getcwd() + "\\trained_models"):
        mkdir(os.getcwd() + "\\trained_models")
    dest = os.getcwd() + "\\trained_models\\"
    train_file = os.getcwd() + "\\testing_set_addition.txt"
    file_path = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    for path in file_path:
        path = path.strip()
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        if count == 6:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)
            # dumping the trained gaussian model
            picklefile = path.split("_")[0] + ".gmm"

            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1
    return "Train complete"


@app.route('/voiceAPI/api/v1.0/get-result-command', methods=['POST'])
def test_model():
    if request.files:
        source = request.files['file']
        print(request.headers['Login'])

        model_path = os.getcwd() + "\\trained_models\\"
        test_file = os.getcwd() + "\\testing_set_addition.txt"
        file_path = open(test_file, 'r')

        gmm_files = [os.path.join(model_path, fname) for fname in os.listdir(model_path) if fname.endswith('.gmm')]
        models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
        speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

        source.save(source.filename)
        src = wavfile.read(str(source.filename))

        sr = src[0]

        print(sr)
        audio = numpy.array(src[1], dtype=float)
        vector = extract_features(audio, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        print(log_likelihood)
        print(winner)
        print("\tdected as - ", speakers[winner])
        harvard = speech_recognition.AudioFile(str(source.filename))
        with harvard as s:
            audio = r.record(s)
        os.remove(str(source.filename))
        time.sleep(1.0)
        if request.headers['Login'].equals(speakers[winner]):
            return getText(audio)
        else:
            return ""
    return "lol you crazy"


@app.route('/voiceAPI/api/v1.0/login', methods=['POST'])
def login():
    login = request.headers['Login']
    password = request.headers['Password']
    return make_response("Success", 200)


@app.route('/voiceAPI/api/v1.0/registration', methods=['POST'])
def registration():
    login = request.headers['Login']
    password = request.headers['Password']
    email = request.headers['email']
    return make_response("Success", 200)


if __name__ == '__main__':
    app.run(host="192.168.0.127", port=8000, debug=True)
