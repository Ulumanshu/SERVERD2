# USAGE
# Start the server:
# python3 Flask_Keras_Multi.py
# Open web link and save letters to pc:D
from keras.models import load_model
import numpy as np
import flask
from flask import Flask, render_template, url_for, request, flash, redirect, jsonify, Response
from werkzeug.datastructures import ImmutableMultiDict
import tensorflow as tf
from werkzeug.serving import run_simple
import base64
import re
from scipy.misc import imread, imresize, toimage
import json
import os
import threading
import string
from model.train_former import Train_Former as T
from model.multi_trainer import Multi_Trainer as L
import time
import gevent
from gevent.queue import Queue
from gevent.pywsgi import WSGIServer
import datetime
import sys
import imageio

def prynt(print_me):
    sys.stdout.write(print_me)
    sys.stdout.flush()


class Zemodel:

    @staticmethod
    def loadmodel(path):
        model = load_model(path)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        print("Model from {} file loaded".format(path))
        return model

    def __init__(self, path, labels_path, name):
        self.model = self.loadmodel(path)
        self.graph = tf.get_default_graph()
        self.label_path = labels_path
        self.name = name
    
    @property
    def labels(self):
        with open(self.label_path) as f:
            labels_dict = json.load(f)
        return labels_dict
    
    def zpredict(self, x):
        with self.graph.as_default():
#            preds = self.model.predict(x)
            probs = self.model.predict_proba(x/255.0)
            prob_round = [round(float(e) * 100, 2)for e in probs[0] if len(probs) > 0]
            probs_dict = {}
            for k, v in self.labels.items():
                probs_dict[k] = prob_round[v]
#            print("PReds", preds, np.argmax(preds, axis=1))
            print("PRobs", probs_dict)
#            results = np.argmax(preds, axis=1)[0]
#            prediction = list(filter(lambda x: x[1] == results, self.labels.items()))[0]
#            print(prediction[0], probs_dict)
#            return prediction[0], probs_dict
            return probs_dict

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.debug = True
ROOT = str(app.root_path)
count = T(
    save_dir="./static/Own_classes/save",
    train_dir="./static/Own_classes/train",
    json_dir="./model/"
)
train = L(
    save_dir="./static/Own_classes/save",
    train_dir="./static/Own_classes/train",
    json_dir="./model/"
)

def moira(*args):
    venv_path = '/home/wooden/Desktop/V.R.Enjoyment/Keras/bin/python'
    script = 'trainer_app.py'
    comm = venv_path + ' ' + script
    if args:
        for arg in args:
            comm += ' ' + arg
    os.system(comm)


class ServerSentEvent(object):

    def __init__(self, data):
        self.data = data
        self.event = None
        self.id = None
        self.desc_map = {
            self.data : "data",
            self.event : "event",
            self.id : "id"
        }

    def encode(self):
        if not self.data:
            return ""
        lines = ["%s: %s" % (v, k) 
                 for k, v in self.desc_map.items() if k]
        
        return "%s\n\n" % "\n".join(lines)

subscriptions = []

def prepare_image(image, target):
    # resize the input image and preprocess it
    image = re.search(r'base64,(.*)', str(image)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(image))
    image = imageio.imread(base64.b64decode(image), pilmode="L")
    image = count.resize_file(image)
#    toimage(image).show()
    image = imresize(image, target)
    image = image.reshape(1, 42, 42, 1)
    # return the processed image
    return image

@app.route("/")
@app.route("/Home")
def home():
    # render homepage html from template
    return render_template('Home.html')

@app.route("/About")
def About():
    # render html from template
    with open('./model/TrFo_Self.json') as f:
        dataset = json.load(f)
    return render_template('About.html', title="About", value=dataset)

@app.route("/reload/<string:command>")
def reload(command):
    res = {}
    if command == "reload":
        if len(threading.enumerate()) > 1:
                res.update(value="Server Busy!")
                
                return jsonify(
                    success=True,
                    data=res
                )
        model_C = Zemodel(
            "./model/models_multi/model_Classifajar.h5",
            "./model/models_multi/labels_Classifajar.json",
            "classifajar"
        )
        model_u = Zemodel(
            "./model/models_multi/model_uppercase.h5",
            "./model/models_multi/labels_uppercase.json",
            "uppercase"
        )
        model_l = Zemodel(
            "./model/models_multi/model_lowercase.h5",
            "./model/models_multi/labels_lowercase.json",
            "lowercase"
        )
        model_n = Zemodel(
            "./model/models_multi/model_numbers.h5",
            "./model/models_multi/labels_numbers.json",
            "numbers"
        )
        res.update(value="Models Reloaded!")
        
    return jsonify(sucess=True, data=res)
    
@app.route("/Train")
def Train():
    return render_template('Train.html', title="Train")

@app.route("/startT/<string:button_id>/<string:checkboxes>", methods=["GET"])
def startT(button_id, checkboxes):
    res = {}
    if button_id == "start_train":
        t = threading.Thread(target=moira, args=(checkboxes and checkboxes or '', ))
        if len(threading.enumerate()) > 1:
            res.update({'value': 'STILL TRAINING!!!'})
            return jsonify(
                success=True,
                data=res
            )
        else:
            res.update({'value': 'LETS SEE!!!'})
            lock = threading.Lock()
            with lock:
                t.start()
        return jsonify(
            success=True,
            data=res
        )
    else:
        res.update({'value': 'ERROr Random!!!'})
        return jsonify(
            success=True,
            data=res
        )
    
@app.route("/trainprogress", methods=["POST"])
def trainprogress():
    req_data = request.get_json()
    header = request.headers.get('wooden')
    req_data['model'] = header
    try:
        data = json.dumps(req_data)
    except:
        return 'ERROR'
    def notify():
        for sub in subscriptions[:]:
            sub.put(req_data)
    
    gevent.spawn(notify)
    return "OK"

@app.route("/drawchart")
def drawchart():
    def gen():
        q = Queue()
        subscriptions.append(q)
        try:
            while True:
                result = q.get()
                event = ServerSentEvent(str(result))
                yield event.encode()
        except GeneratorExit:
            subscriptions.remove(q)
    return Response(gen(), mimetype="text/event-stream")
    
@app.route("/postman", methods=["GET","POST"])
def postman():
    # refresh dataset nfo
    if flask.request.method == "GET":
        response = request.args.to_dict(flat=False)
        # on server bug with response['key'][0], on localhost without - vice versa :(
        response_key = response['key']
        def string_response(response_data):
            if type(response_data) == list and len(response_data) == 1:
                resp = str(response_data[0])
                return resp
            elif type(response_data) == list and len(response_data) > 1:
                resp = response_data
                return resp
            elif type(response_data) == list and len(response_data) == 0:
                resp = ""
                return resp
            elif type(response_data) == str:
                resp = response_data
                return resp
            return 'fail'
        if string_response(response_key) == "refresh_data":
            count.accountant()
            with open('./model/TrFo_Self.json') as f:
                dataset = json.load(f)
            return jsonify(success=True,
                           data=render_template('refresh_dataset.html', value=dataset))
        elif string_response(response_key) == "train_fill":
            count.File_Copy()
            count.accountant()
            with open('./model/TrFo_Self.json') as f:
                dataset = json.load(f)
            return jsonify(
                success=True, data=render_template('refresh_dataset.html', value=dataset)
            )
        elif string_response(response_key) == "train_purge":
            count.Purge_Train()
            count.accountant()
            with open('./model/TrFo_Self.json') as f:
                dataset = json.load(f)
            return jsonify(
                success=True, data=render_template('refresh_dataset.html', value=dataset)
            )
        elif string_response(response_key) == "dataset_view":
            selection = string_response(response['selection'])
            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            if selection != "none":
                sel_id = string_response(response['id'])
                path_to_choice = count.save_dir + "/" + sel_id + "/" + selection
                f_count, f_list = count.count_file(path_to_choice)
                f_list = map(
                    lambda e : e + '?{}'.format(date),
                    f_list
                )
                return jsonify(
                    success=True,
                    data=render_template(
                        'gallery.html',
                        dire=path_to_choice,
                        files=sorted(f_list, key=count.sort_key)
                    )
                )
            elif selection == "none":
                return jsonify(success=True, data=render_template('gallery.html', files=[]))
        elif string_response(response_key) == "delete_file": 
            paths = response.get('del_list[]')
            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            if len(paths) > 0:
                for path in paths:
                    path = re.search(r"(.+\?)", str(path)).group(1)[:-1]
                    count.delete_file(string_response(path))
                path_to_choice = re.search(r"(.+\/)", str(paths[0])).group(1)[:-1]
                count.rename_dir_files(path_to_choice)
                f_count, f_list = count.count_file(path_to_choice)
                f_list = map(
                    lambda e : e + '?{}'.format(date),
                    f_list
                )
                return jsonify(
                    success=True,
                    data=render_template(
                        'gallery.html',
                        dire=path_to_choice,
                        files=sorted(f_list, key=count.sort_key)
                    )
                )
            else:
                return jsonify(success=True, data=render_template('gallery.html', files=[]))
            
    return "just_in_case"

#@app.route("/upload/<directory><filename>")
#def send_image(directory, filename):
#    print(directory, filename)
#    return send_from_directory(directory, filename)


@app.route("/View", methods=["GET","POST"])
def dataset_view():
    res = {}
    numbr, categories = count.count_dir(count.save_dir)
    for cat in categories:
        n, i = count.count_dir(count.save_dir + '/' + cat)
        res[cat] = sorted(i)
    return render_template(
        'Dataset_view.html', title="Dataset View", directory=count.save_dir, value=res
    )

@app.route("/save", methods=["GET","POST"])
def save():
    results = []
    filebase_dir = "./static/Own_classes/save"
    dir_lowercase = '/lowercase'
    dir_uppercase = '/uppercase'
    dir_numbers = '/numbers'
    if flask.request.method == "POST":
        response = request.get_data()
        alt_image = request.form["image"]
        image = re.search(r'base64,(.+)', str(alt_image)).group(1)
        c_class = re.search(r'&correct_class=(.)', str(response)).group(1)
        if len(re.search(r'&correct_class=(.+)', str(response)).group(1)) > 5:
            c_class = re.search(r"&correct_class=\w\w\w\w\w\w_(.)", str(response)).group(1)
        save_dir = []
        upercase = string.ascii_uppercase
        lowercase = string.ascii_lowercase
        digits = string.digits
        all_Valid = upercase + lowercase + digits
        if c_class in upercase:
            dir_end = "/letter_" + c_class
            e = filebase_dir + dir_uppercase + dir_end
            save_dir = e
        elif c_class in lowercase:
            dir_end = "/letter_" + c_class
            e = filebase_dir + dir_lowercase + dir_end
            save_dir = e
        elif c_class in digits:
            dir_end = "/number_" + c_class
            e = filebase_dir + dir_numbers + dir_end
            save_dir = e
        elif c_class not in all_Valid:
            error_msg = ["there is no such dir", c_class, "there is no such dir", c_class]
            json_res = jsonify(error_msg)
            return json_res
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        cnt = len(next(os.walk(save_dir))[2]) + 1
        fn_root = dir_end[1:] + "_" + str(cnt)
        fname = '%s.png' % fn_root
        # TODO resize on save
        image = imageio.imread(base64.b64decode(image), pilmode="L")
        image = count.resize_file(image)
        toimage(image).save(os.path.join(save_dir, fname))
        results.append(save_dir)
        results.append(fname)
    # return jsonified list for js parse, html display
    json_res = jsonify(results)
    return json_res

@app.route("/predict", methods=["POST"])
def predict():
    models = {
        "classifajar": model_C,
        "lowercase": model_l,
        "uppercase": model_u,
        "numbers": model_n
    }
    def dict_max(labels_dict):
        res = tuple()
        maximum = 0.0
        for key, value in labels_dict.items():
            if value > maximum:
                maximum = value
        res = list(filter(lambda x: x[1] == maximum, labels_dict.items()))[0]
        return res
    # if somebody accidently pushes a button "predict" href="/predict"
    results = None
    percents = []
    req_data = request.get_json()
    # preprocess the image and prepare it for classification
    image = prepare_image(req_data['image'], target=(42, 42))
    checkboxes = req_data['checkboxes'] and req_data['checkboxes'].split(' ')
    # classify the input image and then initialize the list
    # of predictions to return to the client
    if 'classifajar' in checkboxes:
        results_c = model_C.zpredict(image)
        percents.append(filter(lambda l: l[1] > 5.0, results_c.items()))
        res_c = dict_max(results_c)
        res = models[res_c[0]].zpredict(image)
        percents.append(filter(lambda l: l[1] > 5.0, res.items()))
        results = dict_max(res)
        return jsonify(
            result=results[0],
            html_prc=render_template('predict_info.html', percent=percents),
            success=True
        )
    else:
        results = {}
        for check in checkboxes:
            results_perc = models[check].zpredict(image)
            percents.append(filter(lambda l: l[1] > 5.0, results_perc.items()))
            res = dict_max(results_perc)
            results[res[0]] = res[1]
        results = dict_max(results)
        return jsonify(
            result=results[0],
            html_prc=render_template('predict_info.html', percent=percents),
            success=True
        )

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    model_C = Zemodel(
        "./model/models_multi/model_Classifajar.h5",
        "./model/models_multi/labels_Classifajar.json",
        "classifajar"
    )
    model_u = Zemodel(
        "./model/models_multi/model_uppercase.h5",
        "./model/models_multi/labels_uppercase.json",
        "uppercase"
    )
    model_l = Zemodel(
        "./model/models_multi/model_lowercase.h5",
        "./model/models_multi/labels_lowercase.json",
        "lowercase"
    )
    model_n = Zemodel(
        "./model/models_multi/model_numbers.h5",
        "./model/models_multi/labels_numbers.json",
        "numbers"
    )
#    run_simple("localhost", 5000, app, use_reloader=True, use_debugger=True, use_evalex=True)
#    app.run(host='0.0.0.0', port=5000)
    server = WSGIServer(("", 5000), app)
    server.serve_forever()
