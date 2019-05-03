from flask import Flask, jsonify, render_template, request, send_file
from numpy import *
import os
from RBF import a, b

from RBF import RBFNetwork

R = RBFNetwork

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = ''

STATIC_URL = '/static/'

STATICFILES_DIRS = (
    os.path.join('static'),
)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/home')
def homee():
    return render_template("home.html")


@app.route('/downloads')
def downloads():
    return render_template("downloads.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/test')
def train():
    return render_template("test.html")
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT,'file/')
    print(target)

    if not  os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    return render_template("test.html", variable = a, variable1 = b)



@app.route('/downloads/PresentationFinal.pptx', methods = ['GET'])
def sendfile():
    return send_file('downloads/PresentationFinal.pptx')
    return jsonify({  "res": "success" })

@app.route('/downloads/Presentation1.pptx', methods = ['GET'])
def sendfile1():
    return send_file('./downloads/Presentation1.pptx')
    return jsonify({  "res": "success" })

@app.route('/downloads/DokumentaciaHVI.pdf', methods = ['GET'])
def sendfile2():
    return send_file('./downloads/DokumentaciaHVI.pdf')
    return jsonify({  "res": "success" })

if __name__ == '__main__':
    app.run(debug = True)
