import operator
import os

from flask import Flask, redirect, render_template, flash, request
from werkzeug.utils import secure_filename

from model import Model
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='dev')
app.config['UPLOAD_FOLDER'] = 'static/user'

current_model = None


def draw_graph(train_acc, val_acc, file: str):
    plt.figure()
    plt.plot(train_acc, label='train accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.xlabel('num epochs')
    plt.legend()
    plt.savefig(file)


@app.route('/train', methods=("GET", "POST"))
def train():
    pic = ""
    messages = []
    if request.method == "POST":
        normalize = 'norm' in request.form
        epochs = request.form['epochs']
        if not epochs:
            flash("Please, enter the number of epochs")
        else:
            epochs = int(epochs)
            global current_model
            current_model = Model('static/data', normalize, epochs)
            train_acc, test_acc = current_model.build_and_train()
            pic = 'static/accuracy.jpg'
            draw_graph(train_acc, test_acc, 'static/accuracy.jpg')
            messages = ['Your model is trained']
    return render_template("train.html", tab='train', pic=pic, messages=messages)


@app.route('/test', methods=("GET", "POST"))
def test():
    pred = ''
    probabilities = {}
    classes = ['crying', 'heart_eyes', 'scream', 'smiling', 'thinking']
    if current_model is None:
        flash('You should train your model first!')
    elif request.method == "POST":
        imagefile = request.files['image']
        if imagefile:
            filename = secure_filename(imagefile.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            imagefile.save(path)
            dct = current_model.predict(path)
            pred = max(dct.items(), key=operator.itemgetter(1))[0]
            probabilities = {key: int(value*1000)/10 for key, value in dct.items()}
        else:
            flash('Please, upload image')
    return render_template("test.html", tab='test', pred=pred, probabilities=probabilities, classes=classes)


@app.route('/')
def index():
    return redirect("/train")


if __name__ == '__main__':
    app.run()
