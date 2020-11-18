#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import random
from flask import Flask, render_template, Response, request, redirect, url_for
from flask_socketio import SocketIO
import pty
import os
import subprocess
import select
import termios
import struct
import fcntl
import shlex
#from fastai.tabular import * 
#from fastai.tabular import * 
from scipy.io import arff
import pandas as pd
import numpy as np
from flask_table import Table, Col

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

deleterand = "rng"


__version__ = "0.4.0.1"

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")
app.config["SECRET_KEY"] = "secret!"
app.config["fd"] = None
app.config["child_pid"] = None
socketio = SocketIO(app)

def pout(output):
    output = output + "\r\n"
    socketio.emit("pty-output", {"output": output}, namespace="/pty")
    return "nothing"

def set_winsize(fd, row, col, xpix=0, ypix=0):
    winsize = struct.pack("HHHH", row, col, xpix, ypix)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def read_and_forward_pty_output():
    max_read_bytes = 1024 * 20
    while True:
        socketio.sleep(0.01)
        if app.config["fd"]:
            timeout_sec = 0
            (data_ready, _, _) = select.select([app.config["fd"]], [], [], timeout_sec)
            if data_ready:
                output = os.read(app.config["fd"], max_read_bytes).decode()
                socketio.emit("pty-output", {"output": output}, namespace="/pty")
        
@app.route('/background_process_test')
def background_process_test():
    output = "Hello \n"            
    socketio.emit("pty-output", {"output": output}, namespace="/pty")
    "socketio.start_background_task(target=read_and_forward_pty_output)"
    return

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("pty-input", namespace="/pty")
def pty_input(data):
    """write to the child pty. The pty sees this as if you are typing in a real
    terminal.
    """
#if app.config["fd"]:
# pout("writing to ptd: %s" % data["input"])
#os.write(app.config["fd"], data["input"].encode())

@socketio.on("resize", namespace="/pty")
def resize(data):
    if app.config["fd"]:
        set_winsize(app.config["fd"], data["rows"], data["cols"])


@socketio.on("connect", namespace="/pty")
def connect():
    """new client connected"""

    if app.config["child_pid"]:
        # already started child process, don't start another
        return

    # create child process attached to a pty we can read from and write to
    (child_pid, fd) = pty.fork()
    if child_pid == 0:
        # this is the child process fork.
        # anything pouted here will show up in the pty, including the output
        # of this subprocess
        subprocess.run(app.config["cmd"])
    else:
        # this is the parent process fork.
        # store child fd and pid
        app.config["fd"] = fd
        app.config["child_pid"] = child_pid
        set_winsize(fd, 50, 50)
        #pout(app.config["cmd"])
        #cmd = " ".join(shlex.quote(c) for c in app.config["cmd"])
        print("child pid is", child_pid)
#         pout(
#             f"starting background task with command `{cmd}` to continously read "
#             "and forward pty output to client"
#         )
        socketio.start_background_task(target=read_and_forward_pty_output)
        pout("task started")
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
from scipy.io import arff
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Add missing dummy columns for target datasets
def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0

#Add missing dummy columns for target datasets
def fix_columns( d, columns ):  

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print("extra columns: {}", extra_cols)

    d = d.reindex(columns=columns)
    return d

# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name].astype(str))
    return le.classes_

# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    
# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    
    dummies = pd.get_dummies(df[target])
    return df[result].to_numpy().astype(np.float32), dummies.to_numpy().astype(np.float32)

def segregate_data(df, label):
    # Break into X (predictors) & y (prediction)
    x, y = to_xy(df, label)

    #pout y.shape
    # Create a test/train split.  25% test
    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42)
    
    return x_train, x_test, y_train, y_test


import pandas as pd
import io
import requests
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger
from keras import callbacks
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def train_model(x_train, y_train, savepath, load_model=False, save_model=False, epochs=20, verbose=0):

    # Create neural net
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(32, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(16, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dense(8, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
        
    model.add(Dense(y_train.shape[1], activation='softmax'))

    if(load_model):
        model.load_weights(savepath)
        model.pop()
        model.add(Dense(y_train.shape[1], activation='softmax'))

    #pout(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    #checkpointer = callbacks.ModelCheckpoint(filepath="Checkpoints/checkpoint-{epoch:02d}.hdf5", verbose=0,  save_best_only=True, monitor='loss')
    #csv_logger = CSVLogger('Checkpoints/training_set_dnnanalysis.csv',separator=',', append=False)
    model.fit(x_train, y_train, verbose=verbose, epochs=epochs)
    
    
    if(save_model == True):
        model.save_weights(savepath)
    
    return model

def generate_metrics(model, x_test, y_test, savepltfile, multiclass=False):
    # Measure accuracy
    pred = model.predict(x_test)
    pred = np.argmax(pred,axis=1)
    y_eval = np.argmax(y_test,axis=1)

    accuracy = str(metrics.accuracy_score(y_eval, pred))
    precision = str(metrics.precision_score(y_eval, pred, average="macro"))
    recall = str(metrics.recall_score(y_eval, pred, average="macro"))
    confusion_matrix = str(metrics.confusion_matrix(y_eval, pred))
    fscore = str(metrics.f1_score(y_eval, pred, average="macro"))
    
    # pout the metrics
    pout("Accuracy: " + accuracy)
    pout("Precision: " + precision)
    pout("Recall: " + recall)
    pout("Confusion Matrix: \n" + confusion_matrix)
    pout("F1-score: " + fscore)

    if(multiclass==False):
        pout("ROC curve(AUC): " + str(metrics.roc_auc_score(y_eval, pred)))

        # Plot the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(y_eval, pred)
        roc_auc = metrics.auc(fpr, tpr)
        # Plot ROC curve
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()
        plt.savefig(savepltfile)
        plt.close()
        
    #pout(model.summary())
    return [accuracy, precision, recall, fscore]

#matplotlib inline

import numpy as np
from keras.layers import Input, Dense, Activation, BatchNormalization, PReLU, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def build_models(n_features):
    """Creates three different models, one used for source only training, two used for domain adaptation"""
    inputs = Input(shape=(n_features,)) 
    x4 = Dense(64, activation='linear')(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)  
    
    x4 = Dense(32, activation='linear')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4) 
    
    x4 = Dense(16, activation='linear')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4) 

    source_classifier = Dense(8, activation='linear')(x4)
    source_classifier = BatchNormalization()(source_classifier)
    source_classifier = Activation("relu")(source_classifier) 
    
    source_classifier = Dense(2, activation='sigmoid', name="mo")(source_classifier)  
    
    
    domain_classifier = Dense(32, activation='linear', name="do4")(x4)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("relu", name="do6")(domain_classifier)
    
    domain_classifier = Dense(32, activation='linear', name="do7")(domain_classifier)
    domain_classifier = BatchNormalization(name="do8")(domain_classifier)
    domain_classifier = Activation("relu", name="do9")(domain_classifier)
    
    domain_classifier = Dense(16, activation='linear', name="do10")(domain_classifier)
    domain_classifier = BatchNormalization(name="do11")(domain_classifier)
    domain_classifier = Activation("relu", name="do12")(domain_classifier)
    
    #domain_classifier = Dropout(0.5, name='do_drop')(domain_classifier)
    domain_classifier = Dense(2, activation='sigmoid', name="do")(domain_classifier)

    
    source_classification_model = Model(inputs=inputs, outputs=[source_classifier])
    source_classification_model.compile(optimizer="Adam",
              loss={'mo': 'categorical_crossentropy'}, metrics=['accuracy'], )


    domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_classification_model.compile(optimizer="Adam",
                  loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])
    
    
    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    comb_model.compile(optimizer="Adam",
              loss={'mo': 'categorical_crossentropy', 'do': 'categorical_crossentropy'},
              loss_weights={'mo': 1, 'do': 1}, metrics=['accuracy'], )

    #pout(comb_model.summary())
    #pout(domain_classification_model.summary())

    embeddings_model = Model(inputs=inputs, outputs=[x4])
    embeddings_model.compile(optimizer="Adam",loss = 'categorical_crossentropy', metrics=['accuracy'])
                        
                        
    return comb_model, source_classification_model, domain_classification_model, embeddings_model

def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr

def train(Xs, ys, Xt, yt,  enable_dann = True, n_iterations = 15000):
    
    batch_size = 32
    
    comb_model, source_classification_model, domain_classification_model, embeddings_model = build_models(Xs.shape[1])
    
    y_class_dummy = np.ones((len(Xt), 2))
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))
        
#     sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_class = np.array(([1] * batch_size + [1] * batch_size))
    sample_weights_adversarial = np.array(([0.1] * batch_size + [0.1] * batch_size))#np.ones((batch_size * 2,))

#     S_batches = batch_generator([Xs, ys], batch_size)
#     T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),2))], batch_size)

    S_batches = batch_generator([Xs, ys], batch_size)
    T_batches = batch_generator([Xt, yt], batch_size)
        
    for i in range(n_iterations):
        # # pout(y_class_dummy.shape, ys.shape)
        y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))

        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)

        X_adv = np.concatenate([X0, X1])
#         y_class = np.concatenate([y0, np.zeros_like(y0)])
        y_class = np.concatenate([y0, y1])

        adv_weights = []
        for layer in comb_model.layers:
            if (layer.name.startswith("do")):
                adv_weights.append(layer.get_weights())
        
#         pout(X_adv)
#         pout(y_class)
#         pout(y_adversarial_1)
#         pout(sample_weights_class)
#         pout(sample_weights_adversarial)
#         break
        
        if(enable_dann):
            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism 
            stats = comb_model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                                     sample_weight=[sample_weights_class, sample_weights_adversarial])
            
            k = 0
            for layer in comb_model.layers:
                if (layer.name.startswith("do")):
                    layer.set_weights(adv_weights[k])
                    k += 1

            class_weights = []
            
        
            for layer in comb_model.layers:
                if (not layer.name.startswith("do")):
                    class_weights.append(layer.get_weights())
            
            stats2 = domain_classification_model.train_on_batch(X_adv, y_adversarial_2)

            k = 0
            for layer in comb_model.layers:
                if (not layer.name.startswith("do")):
                    layer.set_weights(class_weights[k])
                    k += 1

        else:
            source_classification_model.train_on_batch(X0,y0)
            
       
        if ((i + 1) % 1000 == 0):
            #pout(i, comb_model.metrics_names)
            #pout(i, domain_classification_model.metrics_names)
            #pout(source_classification_model.predict(Xt))
            y_test_hat_t = to_categorical(source_classification_model.predict(Xt).argmax(1))
            y_test_hat_s = to_categorical(source_classification_model.predict(Xs).argmax(1))
            
            pout("Iteration %d, src acc =  %.3f, tgt acc = %.3f,  mo_loss = %.3f, do_loss = %.3f, domain_acc = %.3f"%(i, accuracy_score(ys, y_test_hat_s), accuracy_score(yt, y_test_hat_t), stats[1], stats[2], stats2[1]))
    return source_classification_model

def load_datasets(attack_type, path_source, path_target):
    cat_names = ['proto', 'state', 'service']

    other_names = ['label', 'is_ftp_login', 'is_sm_ips_ports', 'is_guest_login']

    ###Import training dataset###
    df_source = pd.read_csv(path_source)
    df_source.drop(columns=['id'],  inplace=True)
    df_source.drop(columns=['attack_cat'],  inplace=True)
    df_source.drop(df_source.columns[0], axis=1, inplace=True)
    
    for col in df_source.columns:
        if(col in cat_names):
            encode_text_dummy(df_source, col)
        elif(col in other_names):
            continue
        else:
            encode_numeric_zscore(df_source, col)
            
    
    x_train_source, x_test_source, y_train_source, y_test_source = segregate_data(df_source,'label')
    
    pout(str(df_source.shape))
    
    cat_names = ['proto', 'state', 'service']
    
    other_names = ['label', 'is_ftp_login', 'is_sm_ips_ports', 'is_guest_login']
    
    ###Import training dataset###
    df_target = pd.read_csv(path_target)
    df_target.drop(columns=['id'],  inplace=True)
    df_target.drop(columns=['attack_cat'],  inplace=True)
    df_target.drop(df_target.columns[0], axis=1, inplace=True)
    
    for col in df_target.columns:
        if(col in cat_names):
            encode_text_dummy(df_target, col)
        elif(col in other_names):
            continue
        else:
            encode_numeric_zscore(df_target, col)
    
    df_target = fix_columns(df_target, df_source.columns)
    pout(str(df_target.shape))
    
    x_train_target, x_test_target, y_train_target, y_test_target = segregate_data(df_target,'label')
    
    return x_train_source, x_test_source, y_train_source, y_test_source, x_train_target, x_test_target, y_train_target, y_test_target

def training_code(x_train_source, x_test_source, y_train_source, y_test_source, x_train_target, x_test_target, y_train_target, y_test_target):
    pout(str(x_train_source.shape))
    pout(str(x_train_target.shape))
    
    num_samples = 25
    
    base = {}
    TL_target = {}
    TL_source = {}
    GAN_target = {}
    GAN_source = {}
    
    
    for i in range(1, 10):
        num_samples = num_samples * 2
        
        indexes = np.random.randint(x_train_target.shape[0], size=num_samples)
        x_train_target_sample = x_train_target[indexes, :]
        y_train_target_sample = y_train_target[indexes, :]
    
        
        pout('-------------Train the source model--------------------')
        source_model = train_model(x_train_source, y_train_source, 'source_model', save_model=True, verbose=0)
        output_source = generate_metrics(source_model, x_test_source, y_test_source, 
                             savepltfile='Base.png', multiclass = False)
    
    
        pout('-------------Base case - test on target dataset--------------------')
        target_model = train_model(x_train_target_sample, y_train_target_sample, 'target_model')
        output_base_target = generate_metrics(target_model, x_test_target, y_test_target, 
                             savepltfile='Base.png', multiclass = False)
    
        pout('-------------TL - test on target dataset---------------')
        TL_model = train_model(x_train_target_sample, y_train_target_sample, 'source_model', load_model=True)
        output_TL_target = generate_metrics(TL_model, x_test_target, y_test_target, 
                             savepltfile='Base.png', multiclass = False)
    
        pout('-------------TL - test on source dataset----------------')
        output_TL_source = generate_metrics(TL_model, x_test_source, y_test_source, 
                             savepltfile='Base.png', multiclass = False)
    
    
        pout('-------------Training GAN----------------')
        embs = train(x_train_source, y_train_source, 
                     x_train_target_sample, y_train_target_sample, 
                     enable_dann = True, n_iterations = 10000)
    
        pout('-------------Testing the target dataset--------------------')
        output_GAN_target = generate_metrics(embs, x_test_target, y_test_target, 
                             savepltfile='Base.png', multiclass = False)
    
        pout('-------------Testing the source dataset--------------------')
        output_GAN_source = generate_metrics(embs, x_test_source, y_test_source, 
                             savepltfile='Base.png', multiclass = False)
    
        base[num_samples] = output_base_target[0]
        TL_target[num_samples] = output_TL_target[0]
        TL_source[num_samples] = output_TL_source[0]
        GAN_target[num_samples] = output_GAN_target[0]
        GAN_source[num_samples] = output_GAN_source[0]
        
    return base, TL_target, TL_source, GAN_target, GAN_source

def perform_experiments(attack_type, path_source, path_target):
    x_train_source, x_test_source, y_train_source, y_test_source, x_train_target, x_test_target, y_train_target, y_test_target = load_datasets(attack_type, path_source, path_target)
    base, TL_target, TL_source, GAN_target, GAN_source = training_code(x_train_source, x_test_source, y_train_source, y_test_source, x_train_target, x_test_target, y_train_target, y_test_target)
    
    pout(str(base))
    pout(str(TL_source))
    pout(str(TL_target))
    pout(str(GAN_source))
    pout(str(GAN_target))
    tablelist = []
    
    class ItemTable(Table):
        name = Col('    Number of samples    ')
        base = Col('    Base Case    ')
        TL_target = Col('    TL target    ')
        TL_source = Col('    TL source    ')
        GAN_target = Col('    GAN target    ')
        GAN_source = Col('    GAN source    ')
        classes = ['tableheader']

    
    class Item(object):
        def __init__(self, name, base, TL_target, TL_source, GAN_target, GAN_source):
            self.name = name
            self.base = base
            self.TL_target = TL_target
            self.TL_source = TL_source
            self.GAN_target = GAN_target
            self.GAN_source = GAN_source 
    
    items = []
    for key in base.keys():
        items.append(Item(key, str("{:0.2f}".format(float(base[key]) * 100)) + "%", 
                          str("{:0.2f}".format(float(TL_target[key]) * 100)) + "%", 
                          str("{:0.2f}".format(float(TL_source[key]) * 100)) + "%", 
                          str("{:0.2f}".format(float(GAN_target[key]) * 100)) + "%", 
                          str("{:0.2f}".format(float(GAN_source[key]) * 100)) + "%"))      
        
    table = ItemTable(items)
 
    pout(table.__html__())
    returnhtml = (table.__html__())
    return returnhtml

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

def main():
    parser = argparse.ArgumentParser(
        description=(
            "A fully functional terminal in your browser. "
            "https://github.com/cs01/pyxterm.js"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", default=5000, help="port to run server on")
    parser.add_argument("-host", "--host", default='127.0.0.1', help="host to run server on (use 0.0.0.0 to allow access from other hosts)")
    parser.add_argument("--debug", action="store_true", help="debug the server")
    parser.add_argument("--version", action="store_true", help="pout version and exit")
    parser.add_argument(
        "--command", default="bash", help="Command to run in the terminal"
    )
    parser.add_argument(
        "--cmd-args",
        default="",
        help="arguments to pass to command (i.e. --cmd-args='arg1 arg2 --flag')",
    )
    args = parser.parse_args()
    if args.version:
        pout(__version__)
        exit(0)
    pout(f"serving on http://127.0.0.1:{args.port}")
    app.config["cmd"] = [args.command] + shlex.split(args.cmd_args)
    socketio.run(app, debug=args.debug, port=args.port, host=args.host)


@app.route('/Exploits')
def exploits():
    pout("Running experiments for attack_type exploits")

    path_source = 'pyxtermjs/Dataset/UNSW_NB15_training-set_No_Exploits.csv'
    path_target = 'pyxtermjs/Dataset/UNSW_NB15_training-set_Exploits.csv'
    
    return perform_experiments("Exploits", path_source, path_target)
    
@app.route('/Reconnaissance')
def reconnaisance():
    pout("Running experiments for attack_type reconnaissance")

    path_source = 'pyxtermjs/Dataset/UNSW_NB15_training-set_No_Reconnaissance.csv'
    path_target = 'pyxtermjs/Dataset/UNSW_NB15_training-set_Reconnaissance.csv'
    
    return perform_experiments("Reconnaissance", path_source, path_target)

@app.route('/Shellcode')
def shellcode():
    pout("Running experiments for attack_type shellcode")

    path_source = 'pyxtermjs/Dataset/UNSW_NB15_training-set_No_Shellcode.csv'
    path_target = 'pyxtermjs/Dataset/UNSW_NB15_training-set_Shellcode.csv'
    
    return perform_experiments("Shellcode", path_source, path_target)

if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
