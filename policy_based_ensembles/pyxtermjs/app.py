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
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import io
import requests
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger
from keras import callbacks
from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
from flask_table import Table, Col
from flask import jsonify

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
    df[name] = le.fit_transform(df[name])
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
#     return df[result].to_numpy().astype(np.float32), dummies.to_numpy().astype(np.float32)
    return df[result], dummies 
def segregate_data(df, label):
    # Break into X (predictors) & y (prediction)
    x, y = to_xy(df, label)

    #print y.shape
    # Create a test/train split.  25% test
    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25)#, random_state=42)
    
    return x_train, x_test, y_train, y_test

ENCODING = 'utf-8'

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%:{}".format(v,round(100*(s[v]/t),2),s[v]))
    return "[{}]".format(",".join(result))
        
def analyze(df):
    #print()
    #print("Analyzing: {}".format(filename))
    #df = pd.read_csv(filename,encoding=ENCODING)
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

pout("Column functions read.")

def train_model(x_train, y_train, savepath, load_model=False, save_model=False, epochs=60, verbose=0):

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

    #print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
#     checkpointer = callbacks.ModelCheckpoint(filepath="Checkpoints/checkpoint-{epoch:02d}.hdf5", verbose=0,  save_best_only=True, monitor='loss')
#     csv_logger = CSVLogger('Checkpoints/training_set_dnnanalysis.csv',separator=',', append=False)
    model.fit(x_train, y_train, verbose=verbose, epochs=epochs)
    
    
    if(save_model == True):
        model.save(savepath)
    
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
    
    # Print the metrics
    print("Accuracy: " + accuracy)
    print("Precision: " + precision)
    print("Recall: " + recall)
    print("Confusion Matrix: \n" + confusion_matrix)
    print("F1-score: " + fscore)

    # if(multiclass==False):
    #     print("ROC curve(AUC): " + str(metrics.roc_auc_score(y_eval, pred)))

    #     # Plot the ROC curve
    #     fpr, tpr, thresholds = metrics.roc_curve(y_eval, pred)
    #     roc_auc = metrics.auc(fpr, tpr)
    #     # Plot ROC curve
    #     plt.title('Receiver Operating Characteristic')
    #     plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
    #     plt.legend(loc='lower right')
    #     plt.plot([0,1],[0,1],'r--')
    #     plt.xlim([-0.1,1.2])
    #     plt.ylim([-0.1,1.2])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     #plt.show()
    #     plt.savefig(savepltfile)
    #     plt.close()
        
    # #print(model.summary())
    return [accuracy, precision, recall, fscore]

pout("Model training methods read.")

col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'difficulty']

cat_names = ['protocol_type', 'flag', 'service']

oth_names = ['class', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

###Import training dataset###
path = 'pyxtermjs/KDDTrain+.txt'
df = pd.read_csv(path, names=col_names)
df.drop(columns=['difficulty'],  inplace=True)
df.drop(columns=['num_outbound_cmds'],  inplace=True)
# df['class'] = (df['class'] =='normal').astype(int)

for col in df.columns:
    if(col in cat_names):
        encode_text_index(df, col)
    elif(col in oth_names):
        continue
    else:
        encode_numeric_zscore(df, col)

dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
probe = ['ipsweep', 'nmap', 'portsweep', 'satan']
u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']

for index, row in df.iterrows():
    if row['class'] in dos:
        df.at[index, 'class'] = 1
    elif row['class'] in probe:
        df.at[index, 'class'] = 2
    elif row['class'] in u2r:
        df.at[index, 'class'] = 3
    elif row['class'] in r2l:
        df.at[index, 'class'] = 4
    else:
        df.at[index, 'class'] = 0
    
#Drop classes 3 and 4
df = df[df['class'] != 3]
df = df[df['class'] != 4]

print(df.head())    
headrows = df.head(4)
pout("The dataset contains the following features:")
printhead = "\r\n   ".join(headrows)
pout(printhead)        
# ###Import test dataset###
# path_test = 'Dataset/KDDTest+.txt'
# df_test = pd.read_csv(path_test, names=col_names)
# df_test.drop(columns=['difficulty'],  inplace=True)
# df_test.drop(columns=['num_outbound_cmds'],  inplace=True)
# df_test['class'] = (df_test['class'] =='normal').astype(int)

# for col in df_test.columns:
#     if(col in cat_names):
#         encode_text_index(df_test, col)
#     else:
#         encode_numeric_zscore(df_test, col)

# print("Training dataset size: {}".format(df_train.shape))
# print("Testing dataset size: {}".format(df_test.shape))
printinfo = df['class'].value_counts()
pout("Dimensions of this dataset: ")
userow = [str(df.shape)]
pout(",".join(userow))
# analyze(df)

x_train, x_test, y_train, y_test = segregate_data(df,'class')

ds_train = np.concatenate((x_train, y_train), axis=1)
# x_train, y_train = np.hsplit(temp, np.array([40]))

ds_train_icmp = ds_train[ds_train[:, 1] == 0] #####0=ICMP, 1=TCP, 2=UDP
ds_train_tcp = ds_train[ds_train[:, 1] == 1]  #####0=ICMP, 1=TCP, 2=UDP
ds_train_udp = ds_train[ds_train[:, 1] == 2]  #####0=ICMP, 1=TCP, 2=UDP

icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=500)
tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=5000)
udp_idx = np.random.randint(ds_train_udp.shape[0], size=0)

ds_train_1 = np.concatenate((ds_train_icmp[icmp_idx, :], 
                            ds_train_tcp[tcp_idx, :], 
                            ds_train_udp[udp_idx, :]), axis=0)

############################################################################################
############################################################################################

icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=0)
tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=5000)
udp_idx = np.random.randint(ds_train_udp.shape[0], size=0)

ds_train_2 = np.concatenate((ds_train_icmp[icmp_idx, :], 
                            ds_train_tcp[tcp_idx, :], 
                            ds_train_udp[udp_idx, :]), axis=0)


############################################################################################
############################################################################################

icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=500)
tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=0)
udp_idx = np.random.randint(ds_train_udp.shape[0], size=5000)

ds_train_3 = np.concatenate((ds_train_icmp[icmp_idx, :], 
                            ds_train_tcp[tcp_idx, :], 
                            ds_train_udp[udp_idx, :]), axis=0)


############################################################################################
############################################################################################


icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=0)
tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=0)
udp_idx = np.random.randint(ds_train_udp.shape[0], size=5000)

ds_train_4 = np.concatenate((ds_train_icmp[icmp_idx, :], 
                            ds_train_tcp[tcp_idx, :], 
                            ds_train_udp[udp_idx, :]), axis=0)

############################################################################################
############################################################################################

icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=500)
tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=5000)
udp_idx = np.random.randint(ds_train_udp.shape[0], size=3000)

ds_train_5 = np.concatenate((ds_train_icmp[icmp_idx, :], 
                            ds_train_tcp[tcp_idx, :], 
                            ds_train_udp[udp_idx, :]), axis=0)

userow = [str(ds_train_1.shape)]
pout(",".join(userow))
userow = [str(ds_train_2.shape)]
pout(",".join(userow))
userow = [str(ds_train_3.shape)]
pout(",".join(userow))
userow = [str(ds_train_4.shape)]
pout(",".join(userow))
userow = [str(ds_train_5.shape)]
pout(",".join(userow))

columns_idx_1 = np.random.randint(40, size=10)
columns_idx_2 = np.random.randint(40, size=10)
columns_idx_3 = np.random.randint(40, size=30)
columns_idx_4 = np.random.randint(40, size=20)
columns_idx_5 = np.random.randint(40, size=10)

x_train_1, y_train_1 = np.hsplit(ds_train_1, np.array([40]))
x_train_2, y_train_2 = np.hsplit(ds_train_2, np.array([40]))
x_train_3, y_train_3 = np.hsplit(ds_train_3, np.array([40]))
x_train_4, y_train_4 = np.hsplit(ds_train_4, np.array([40]))
x_train_5, y_train_5 = np.hsplit(ds_train_5, np.array([40]))

x_train_1 = x_train_1[:, columns_idx_1]
x_train_2 = x_train_2[:, columns_idx_2]
x_train_3 = x_train_3[:, columns_idx_3]
x_train_4 = x_train_4[:, columns_idx_4]
x_train_5 = x_train_5[:, columns_idx_5]

userow = [str(x_train_1.shape)]
pout(",".join(userow))
userow = [str(x_train_2.shape)]
pout(",".join(userow))
userow = [str(x_train_3.shape)]
pout(",".join(userow))
userow = [str(x_train_4.shape)]
pout(",".join(userow))
userow = [str(x_train_5.shape)]
pout(",".join(userow))

model_1 = train_model(x_train_1, y_train_1, 'model_1', epochs=60)
# generate_metrics(model_1, x_test, y_test, savepltfile, multiclass=False)
print("Model 1 trained")
model_2 = train_model(x_train_2, y_train_2, 'model_2', epochs=60)
print("Model 2 trained")
model_3 = train_model(x_train_3, y_train_3, 'model_3', epochs=60)
print("Model 3 trained")
model_4 = train_model(x_train_4, y_train_4, 'model_4', epochs=60)
print("Model 4 trained")
model_5 = train_model(x_train_5, y_train_5, 'model_5', epochs=60)
print("Model 5 trained")
# x_train, x_test, y_train, y_test = segregate_data(df,'class')

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                            random_state=0)
forest.fit(x_train, y_train)

print(forest.feature_importances_)
importances = forest.feature_importances_


# @app.route('/retrain')
# def retrain():
#     return render_template("index.html")
# x_train, x_test, y_train, y_test = segregate_data(df,'class')

# ds_train = np.concatenate((x_train, y_train), axis=1)
# # x_train, y_train = np.hsplit(temp, np.array([40]))

# ds_train_icmp = ds_train[ds_train[:, 1] == 0] #####0=ICMP, 1=TCP, 2=UDP
# ds_train_tcp = ds_train[ds_train[:, 1] == 1]  #####0=ICMP, 1=TCP, 2=UDP
# ds_train_udp = ds_train[ds_train[:, 1] == 2]  #####0=ICMP, 1=TCP, 2=UDP

# icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=500)
# tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=5000)
# udp_idx = np.random.randint(ds_train_udp.shape[0], size=0)

# ds_train_1 = np.concatenate((ds_train_icmp[icmp_idx, :], 
#                             ds_train_tcp[tcp_idx, :], 
#                             ds_train_udp[udp_idx, :]), axis=0)

# ############################################################################################
# ############################################################################################

# icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=0)
# tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=5000)
# udp_idx = np.random.randint(ds_train_udp.shape[0], size=0)

# ds_train_2 = np.concatenate((ds_train_icmp[icmp_idx, :], 
#                             ds_train_tcp[tcp_idx, :], 
#                             ds_train_udp[udp_idx, :]), axis=0)


# ############################################################################################
# ############################################################################################

# icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=500)
# tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=0)
# udp_idx = np.random.randint(ds_train_udp.shape[0], size=5000)

# ds_train_3 = np.concatenate((ds_train_icmp[icmp_idx, :], 
#                             ds_train_tcp[tcp_idx, :], 
#                             ds_train_udp[udp_idx, :]), axis=0)


# ############################################################################################
# ############################################################################################


# icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=0)
# tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=0)
# udp_idx = np.random.randint(ds_train_udp.shape[0], size=5000)

# ds_train_4 = np.concatenate((ds_train_icmp[icmp_idx, :], 
#                             ds_train_tcp[tcp_idx, :], 
#                             ds_train_udp[udp_idx, :]), axis=0)

# ############################################################################################
# ############################################################################################

# icmp_idx = np.random.randint(ds_train_icmp.shape[0], size=500)
# tcp_idx = np.random.randint(ds_train_tcp.shape[0], size=5000)
# udp_idx = np.random.randint(ds_train_udp.shape[0], size=3000)

# ds_train_5 = np.concatenate((ds_train_icmp[icmp_idx, :], 
#                             ds_train_tcp[tcp_idx, :], 
#                             ds_train_udp[udp_idx, :]), axis=0)

# userow = [str(ds_train_1.shape)]
# pout(",".join(userow))
# userow = [str(ds_train_2.shape)]
# pout(",".join(userow))
# userow = [str(ds_train_3.shape)]
# pout(",".join(userow))
# userow = [str(ds_train_4.shape)]
# pout(",".join(userow))
# userow = [str(ds_train_5.shape)]
# pout(",".join(userow))

# columns_idx_1 = np.random.randint(40, size=10)
# columns_idx_2 = np.random.randint(40, size=10)
# columns_idx_3 = np.random.randint(40, size=30)
# columns_idx_4 = np.random.randint(40, size=20)
# columns_idx_5 = np.random.randint(40, size=10)

# x_train_1, y_train_1 = np.hsplit(ds_train_1, np.array([40]))
# x_train_2, y_train_2 = np.hsplit(ds_train_2, np.array([40]))
# x_train_3, y_train_3 = np.hsplit(ds_train_3, np.array([40]))
# x_train_4, y_train_4 = np.hsplit(ds_train_4, np.array([40]))
# x_train_5, y_train_5 = np.hsplit(ds_train_5, np.array([40]))

# x_train_1 = x_train_1[:, columns_idx_1]
# x_train_2 = x_train_2[:, columns_idx_2]
# x_train_3 = x_train_3[:, columns_idx_3]
# x_train_4 = x_train_4[:, columns_idx_4]
# x_train_5 = x_train_5[:, columns_idx_5]

# userow = [str(x_train_1.shape)]
# pout(",".join(userow))
# userow = [str(x_train_2.shape)]
# pout(",".join(userow))
# userow = [str(x_train_3.shape)]
# pout(",".join(userow))
# userow = [str(x_train_4.shape)]
# pout(",".join(userow))
# userow = [str(x_train_5.shape)]
# pout(",".join(userow))

# model_1 = train_model(x_train_1, y_train_1, 'model_1', epochs=60)
# # generate_metrics(model_1, x_test, y_test, savepltfile, multiclass=False)
# print("Model 1 trained")
# model_2 = train_model(x_train_2, y_train_2, 'model_2', epochs=60)
# print("Model 2 trained")
# model_3 = train_model(x_train_3, y_train_3, 'model_3', epochs=60)
# print("Model 3 trained")
# model_4 = train_model(x_train_4, y_train_4, 'model_4', epochs=60)
# print("Model 4 trained")
# model_5 = train_model(x_train_5, y_train_5, 'model_5', epochs=60)
# print("Model 5 trained")
# # x_train, x_test, y_train, y_test = segregate_data(df,'class')

# # Build a forest and compute the feature importances
# forest = ExtraTreesClassifier(n_estimators=250,
#                             random_state=0)
# forest.fit(x_train, y_train)

# print(forest.feature_importances_)
# importances = forest.feature_importances_


@socketio.on("pty-input", namespace="/pty")
def pty_input(data):
    """write to the child pty. The pty sees this as if you are typing in a real
    terminal.
    """
#if app.config["fd"]:
# print("writing to ptd: %s" % data["input"])
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
        # anything printed here will show up in the pty, including the output
        # of this subprocess
        subprocess.run(app.config["cmd"])
    else:
        # this is the parent process fork.
        # store child fd and pid
        app.config["fd"] = fd
        app.config["child_pid"] = child_pid
        set_winsize(fd, 50, 50)
        #print(app.config["cmd"])
        #cmd = " ".join(shlex.quote(c) for c in app.config["cmd"])
        print("child pid is", child_pid)
#         print(
#             f"starting background task with command `{cmd}` to continously read "
#             "and forward pty output to client"
#         )
        socketio.start_background_task(target=read_and_forward_pty_output)
        print("task started")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A fully functional terminal in your browser. "
            "https://github.com/cs01/pyxterm.js"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", default=5000, help="port to run server on")
    parser.add_argument("-host", "--host", default='0.0.0.0', help="host to run server on (use 0.0.0.0 to allow access from other hosts)")
    parser.add_argument("--debug", action="store_true", help="debug the server")
    parser.add_argument("--version", action="store_true", help="print version and exit")
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
        print(__version__)
        exit(0)
    print(f"serving on http://127.0.0.1:{args.port}")
    app.config["cmd"] = [args.command] + shlex.split(args.cmd_args)
    socketio.run(app, debug=args.debug, port=args.port, host=args.host)



@app.route('/combined')
def combined_ICMP():
    pout("\r\nYou chose to combine policies")
    pout("Starting output stream.")
    # imp_dict = dict(zip(df.columns, forest.feature_importances_))
    # sorted_imp_dict = sorted((value, key) for (key,value) in imp_dict.items())

    # print(sorted_imp_dict)
    tablelist = []
    totalList = []
    #####Change here 0=ICMP, 1=TCP, 2=UDP
    for cpt in range(0,3):
        current_protocol_type = cpt #######CHANGE VALUE VIA FUNCTION

        frac_a_1 = ds_train_1[ds_train_1[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_1.shape[0]
        frac_a_2 = ds_train_2[ds_train_2[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_2.shape[0]
        frac_a_3 = ds_train_3[ds_train_3[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_3.shape[0]
        frac_a_4 = ds_train_4[ds_train_4[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_4.shape[0]
        frac_a_5 = ds_train_5[ds_train_5[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_5.shape[0]

        pout("==============Frac_a==============")
        pout(str(frac_a_1))
        pout(str(frac_a_2))
        pout(str(frac_a_3))
        pout(str(frac_a_4))
        pout(str(frac_a_5))

        frac_b_1 = 0
        frac_b_2 = 0
        frac_b_3 = 0
        frac_b_4 = 0
        frac_b_5 = 0

        for column_idx in columns_idx_1:
            frac_b_1 += importances[column_idx]

        for column_idx in columns_idx_2:
            frac_b_2 += importances[column_idx]
            
        for column_idx in columns_idx_3:
            frac_b_3 += importances[column_idx]

        for column_idx in columns_idx_4:
            frac_b_4 += importances[column_idx]

        for column_idx in columns_idx_5:
            frac_b_5 += importances[column_idx]

        pout("==============Frac_b==============")
        pout(str(frac_b_1))
        pout(str(frac_b_2))
        pout(str(frac_b_3))
        pout(str(frac_b_4))
        pout(str(frac_b_5))

        frac_1 = frac_a_1 * frac_b_1
        frac_2 = frac_a_2 * frac_b_2
        frac_3 = frac_a_3 * frac_b_3
        frac_4 = frac_a_4 * frac_b_4
        frac_5 = frac_a_5 * frac_b_5

        pout("==============Frac_combined==============")
        pout(str(frac_1))
        pout(str(frac_2))
        pout(str(frac_3))
        pout(str(frac_4))
        pout(str(frac_5))

        ds_test = np.concatenate((x_test, y_test), axis=1)
        ds_test_temp = ds_test[ds_test[:, 1] == current_protocol_type]  
        ds_test_temp = ds_test_temp[:2000,:]
        x_test_temp, y_test_temp = np.hsplit(ds_test_temp, np.array([40]))

        usestring = "Num of samples: {0}".format(y_test_temp.shape[0])
        pout(usestring)
        x_test_1 = x_test_temp[:, columns_idx_1]
        x_test_2 = x_test_temp[:, columns_idx_2]
        x_test_3 = x_test_temp[:, columns_idx_3]
        x_test_4 = x_test_temp[:, columns_idx_4]
        x_test_5 = x_test_temp[:, columns_idx_5]

        pred_1 = model_1.predict(x_test_1)
        pred_2 = model_2.predict(x_test_2)
        pred_3 = model_3.predict(x_test_3)
        pred_4 = model_4.predict(x_test_4)
        pred_5 = model_5.predict(x_test_5)

        #####################################################################

        pred_policy_a = (frac_a_1 * np.array(pred_1) + 
                    frac_a_2 * np.array(pred_2) + 
                    frac_a_3 * np.array(pred_3) + 
                    frac_a_4 * np.array(pred_4) + 
                    frac_a_5 * np.array(pred_5))/ (frac_a_1 + frac_a_2 + frac_a_3 +frac_a_4 + frac_a_5)
        pred_policy_a = np.argmax(pred_policy_a,axis=1)

        #####################################################################

        pred_policy_b = (frac_b_1 * np.array(pred_1) + 
                    frac_b_2 * np.array(pred_2) + 
                    frac_b_3 * np.array(pred_3) + 
                    frac_b_4 * np.array(pred_4) + 
                    frac_b_5 * np.array(pred_5))/ (frac_b_1 + frac_b_2 + frac_b_3 +frac_b_4 + frac_b_5)
        pred_policy_b = np.argmax(pred_policy_b,axis=1)

        #####################################################################

        pred_policy_comb = (frac_1 * np.array(pred_1) + 
                    frac_2 * np.array(pred_2) + 
                    frac_3 * np.array(pred_3) + 
                    frac_4 * np.array(pred_4) + 
                    frac_5 * np.array(pred_5))/ (frac_1 + frac_2 + frac_3 +frac_4 + frac_5)
        pred_policy_comb = np.argmax(pred_policy_comb,axis=1)


        #####################################################################

        pred_naive = (np.array(pred_1) + np.array(pred_2) + np.array(pred_3) + np.array(pred_4) + np.array(pred_5)) / 5.0
        pred_naive = np.argmax(pred_naive,axis=1)

        #####################################################################

        y_eval = np.argmax(y_test_temp, axis=1)

        #####################################################################

        #Average accuracy of combined ensembles: Only Policy A
        pout('Accuracy-Policy_a')
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_policy_a)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%")
        pout(accuracy)

        #Average accuracy of combined ensembles: Only Policy B
        pout('Accuracy-Policy_b')
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_policy_b)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%")        
        pout(accuracy)

        #Average accuracy of combined ensembles: Combined Policies
        pout('Accuracy-Combined_Policies')
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_policy_comb)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%") 
        pout(accuracy)

        pout('Accuracy-naive')
        #Average accuracy of combined ensembles
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_naive)) * 100)) 
        totalList.append(accuracy)
        tablelist.append(accuracy + "%") 
        pout(accuracy)

        # pout('Individual model accuracy')
        # pred_1 = np.argmax(pred_1,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_1))
        # pout(accuracy)

        # pred_2 = np.argmax(pred_2,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_2))
        # pout(accuracy)

        # pred_3 = np.argmax(pred_3,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_3))
        # pout(accuracy)

        # pred_4 = np.argmax(pred_4,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_4))
        # pout(accuracy)

        # pred_5 = np.argmax(pred_5,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_5))
        # pout(accuracy)
    
    tablelist.append(str("{:.2f}".format((float(totalList[0]) * 0.33) + (float(totalList[4]) * 0.33) + (float(totalList[8]) * 0.33))) + "%")
    tablelist.append(str("{:.2f}".format((float(totalList[1]) * 0.33) + (float(totalList[5]) * 0.33) + (float(totalList[9]) * 0.33))) + "%")
    tablelist.append(str("{:.2f}".format((float(totalList[2]) * 0.33) + (float(totalList[6]) * 0.33) + (float(totalList[10]) * 0.33)))+ "%")
    tablelist.append(str("{:.2f}".format((float(totalList[3]) * 0.33) + (float(totalList[7]) * 0.33) + (float(totalList[11]) * 0.33)))+ "%")


    class ItemTable(Table):
        name = Col('      ')
        pol_a = Col('Feature Distribution')
        pol_b = Col('Feature Importance')
        pol_comb = Col('Combined Policies')
        naive = Col('Naive Ensemble')

    n_groups = 4
    labels =["ICMP Samples", "TCP Samples", "UDP Samples", "Whole Dataset"]
    disttotal = ("{:.2f}".format((float(totalList[0]) * 0.33) + (float(totalList[4]) * 0.33) + (float(totalList[8]) * 0.33)))
    dist_policy = (float(totalList[0]),float(totalList[4]),float(totalList[8]), float(disttotal))
    print(totalList)
    imptotal = ("{:.2f}".format((float(totalList[1]) * 0.33) + (float(totalList[5]) * 0.33) + (float(totalList[9]) * 0.33)))
    imp_policy = (float(totalList[1]),float(totalList[5]),float(totalList[9]), float(imptotal))
    combtotal = ("{:.2f}".format((float(totalList[2]) * 0.33) + (float(totalList[6]) * 0.33) + (float(totalList[10]) * 0.33)))
    comb_policy = (float(totalList[2]),float(totalList[6]),float(totalList[10]), float(combtotal))
    naivetotal = ("{:.2f}".format((float(totalList[3]) * 0.33) + (float(totalList[7]) * 0.33) + (float(totalList[11]) * 0.33)))
    naive = (float(totalList[3]),float(totalList[7]),float(totalList[11]), float(naivetotal))


    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.95

    rects1 = plt.bar(index,dist_policy, bar_width,
    alpha=opacity,
    color='y',
    label='Feature Distribution Policy')

    rects2 = plt.bar(index + bar_width, imp_policy, bar_width,
    alpha=opacity,
    color='b',
    label='Feature Importance Policy')

    rects3 = plt.bar(index + (bar_width * 2), comb_policy, bar_width,
    alpha=opacity,
    color='g',
    label='Combined Policy Ensemble')

    rects4 = plt.bar(index + (bar_width * 3), naive, bar_width,
    alpha=opacity,
    color='r',
    label='Naive Ensemble')

    plt.ylabel('Accuracy (%)')
    plt.title('Combined Policies Ensemble vs Naive Ensemble')
    plt.xticks(index + bar_width, ("ICMP Samples", "TCP Samples", "UDP Samples", "Whole Dataset"))
    plt.legend()

    global deleterand

    import os
    if os.path.exists("pyxtermjs/results" + deleterand + ".png"):
        os.remove("pyxtermjs/results" + deleterand + ".png")
    else:
        print("The file does not exist")

    plt.tight_layout()
    randinit = str(random.randrange(1000))
    plt.savefig('pyxtermjs/results'+ randinit + '.png', transparent = True)
    deleterand = randinit

    class Item(object):
        def __init__(self, name, pol_a, pol_b, pol_comb, naive):
            self.name = name
            self.pol_a = pol_a
            self.pol_b = pol_b
            self.pol_comb = pol_comb
            self.naive = naive 
    items = [Item('ICMP Samples',tablelist.pop(0),tablelist.pop(0),tablelist.pop(0),tablelist.pop(0)),
            Item('TCP Samples',tablelist.pop(0),tablelist.pop(0),tablelist.pop(0),tablelist.pop(0)),
            Item('UDP Samples',tablelist.pop(0),tablelist.pop(0),tablelist.pop(0),tablelist.pop(0)),
            Item('Whole Dataset',tablelist.pop(0),tablelist.pop(0),tablelist.pop(0),tablelist.pop(0))]
    table = ItemTable(items)

    print(table.__html__())
    returnhtml = (table.__html__())
    returnhtml = returnhtml + "<br>\n<br>\n<img src='results" + randinit + ".png' alt='Accuracy Chart' class='center'>"
    print(returnhtml)
    return returnhtml
    


@app.route('/policy-a')
def combined_TCP():
    pout("\r\nYou chose Policy A")
    pout("Starting output stream.")
    
    # imp_dict = dict(zip(df.columns, forest.feature_importances_))
    # sorted_imp_dict = sorted((value, key) for (key,value) in imp_dict.items())

    # print(sorted_imp_dict)
    tablelist = []
    totalList = []
    #####Change here 0=ICMP, 1=TCP, 2=UDP
    for cpt in range(0, 3):
        current_protocol_type = cpt #######CHANGE VALUE VIA FUNCTION

        frac_a_1 = ds_train_1[ds_train_1[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_1.shape[0]
        frac_a_2 = ds_train_2[ds_train_2[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_2.shape[0]
        frac_a_3 = ds_train_3[ds_train_3[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_3.shape[0]
        frac_a_4 = ds_train_4[ds_train_4[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_4.shape[0]
        frac_a_5 = ds_train_5[ds_train_5[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_5.shape[0]

        pout("==============Frac_a==============")
        pout(str(frac_a_1))
        pout(str(frac_a_2))
        pout(str(frac_a_3))
        pout(str(frac_a_4))
        pout(str(frac_a_5))

        frac_b_1 = 0
        frac_b_2 = 0
        frac_b_3 = 0
        frac_b_4 = 0
        frac_b_5 = 0

        for column_idx in columns_idx_1:
            frac_b_1 += importances[column_idx]

        for column_idx in columns_idx_2:
            frac_b_2 += importances[column_idx]
            
        for column_idx in columns_idx_3:
            frac_b_3 += importances[column_idx]

        for column_idx in columns_idx_4:
            frac_b_4 += importances[column_idx]

        for column_idx in columns_idx_5:
            frac_b_5 += importances[column_idx]

        pout("==============Frac_b==============")
        pout(str(frac_b_1))
        pout(str(frac_b_2))
        pout(str(frac_b_3))
        pout(str(frac_b_4))
        pout(str(frac_b_5))

        frac_1 = frac_a_1 * frac_b_1
        frac_2 = frac_a_2 * frac_b_2
        frac_3 = frac_a_3 * frac_b_3
        frac_4 = frac_a_4 * frac_b_4
        frac_5 = frac_a_5 * frac_b_5

        pout("==============Frac_combined==============")
        pout(str(frac_1))
        pout(str(frac_2))
        pout(str(frac_3))
        pout(str(frac_4))
        pout(str(frac_5))

        ds_test = np.concatenate((x_test, y_test), axis=1)
        ds_test_temp = ds_test[ds_test[:, 1] == current_protocol_type]
        ds_test_temp = ds_test_temp[:2000,:]
        x_test_temp, y_test_temp = np.hsplit(ds_test_temp, np.array([40]))

        usestring = "Num of samples: {0}".format(y_test_temp.shape[0])
        pout(usestring)
        x_test_1 = x_test_temp[:, columns_idx_1]
        x_test_2 = x_test_temp[:, columns_idx_2]
        x_test_3 = x_test_temp[:, columns_idx_3]
        x_test_4 = x_test_temp[:, columns_idx_4]
        x_test_5 = x_test_temp[:, columns_idx_5]

        pred_1 = model_1.predict(x_test_1)
        pred_2 = model_2.predict(x_test_2)
        pred_3 = model_3.predict(x_test_3)
        pred_4 = model_4.predict(x_test_4)
        pred_5 = model_5.predict(x_test_5)

        #####################################################################

        pred_policy_a = (frac_a_1 * np.array(pred_1) + 
                    frac_a_2 * np.array(pred_2) + 
                    frac_a_3 * np.array(pred_3) + 
                    frac_a_4 * np.array(pred_4) + 
                    frac_a_5 * np.array(pred_5))/ (frac_a_1 + frac_a_2 + frac_a_3 +frac_a_4 + frac_a_5)
        pred_policy_a = np.argmax(pred_policy_a,axis=1)

        #####################################################################

        pred_policy_b = (frac_b_1 * np.array(pred_1) + 
                    frac_b_2 * np.array(pred_2) + 
                    frac_b_3 * np.array(pred_3) + 
                    frac_b_4 * np.array(pred_4) + 
                    frac_b_5 * np.array(pred_5))/ (frac_b_1 + frac_b_2 + frac_b_3 +frac_b_4 + frac_b_5)
        pred_policy_b = np.argmax(pred_policy_b,axis=1)

        #####################################################################

        pred_policy_comb = (frac_1 * np.array(pred_1) + 
                    frac_2 * np.array(pred_2) + 
                    frac_3 * np.array(pred_3) + 
                    frac_4 * np.array(pred_4) + 
                    frac_5 * np.array(pred_5))/ (frac_1 + frac_2 + frac_3 +frac_4 + frac_5)
        pred_policy_comb = np.argmax(pred_policy_comb,axis=1)


        #####################################################################

        pred_naive = (np.array(pred_1) + np.array(pred_2) + np.array(pred_3) + np.array(pred_4) + np.array(pred_5)) / 5.0
        pred_naive = np.argmax(pred_naive,axis=1)

        #####################################################################

        y_eval = np.argmax(y_test_temp, axis=1)

        #####################################################################

        #Average accuracy of combined ensembles: Only Policy A
        pout('Accuracy-Policy_a')
        accuracy = str( "{:.2f}".format((metrics.accuracy_score(y_eval, pred_policy_a)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%")
        pout(accuracy)

        #Average accuracy of combined ensembles: Only Policy B
        # pout('Accuracy-Policy_b')
        # accuracy = str(metrics.accuracy_score(y_eval, pred_policy_b))
        # pout(accuracy)

        #Average accuracy of combined ensembles: Combined Policies
        # pout('Accuracy-Combined_Policies')
        # accuracy = str(metrics.accuracy_score(y_eval, pred_policy_comb))
        # pout(accuracy)

        pout('Accuracy-naive')
        #Average accuracy of combined ensembles
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_naive)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%")
        pout(accuracy)

        # pout('Individual model accuracy')
        # pred_1 = np.argmax(pred_1,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_1))
        # pout(accuracy)

        # pred_2 = np.argmax(pred_2,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_2))
        # pout(accuracy)

        # pred_3 = np.argmax(pred_3,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_3))
        # pout(accuracy)

        # pred_4 = np.argmax(pred_4,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_4))
        # pout(accuracy)

        # pred_5 = np.argmax(pred_5,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_5))
        # pout(accuracy)

    tablelist.append(str("{:.2f}".format((float(totalList[0]) * 0.33) + (float(totalList[2]) * 0.33) + (float(totalList[4]) * 0.33)))+ "%")
    tablelist.append(str("{:.2f}".format((float(totalList[1]) * 0.33) + (float(totalList[3]) * 0.33) + (float(totalList[5]) * 0.33)))+ "%")
    class ItemTable(Table):
        name = Col('      ')
        pol_a = Col('Policy Based')
        naive = Col('Naive Ensemble')

    n_groups = 4
    labels =["ICMP Samples", "TCP Samples", "UDP Samples", "Whole Dataset"]
    imptotal = ("{:.2f}".format((float(totalList[0]) * 0.33) + (float(totalList[2]) * 0.33) + (float(totalList[4]) * 0.33)))
    dist_policy = (float(totalList[0]),float(totalList[2]),float(totalList[4]), float(imptotal))
    print(totalList)
    naivetotal = ("{:.2f}".format((float(totalList[1]) * 0.33) + (float(totalList[3]) * 0.33) + (float(totalList[5]) * 0.33)))
    naive = (float(totalList[1]),float(totalList[3]),float(totalList[5]), float(naivetotal))


    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.95

    rects1 = plt.bar(index,dist_policy, bar_width,
    alpha=opacity,
    color='y',
    label='Feature Distribution Policy')

    rects2 = plt.bar(index + bar_width, naive, bar_width,
    alpha=opacity,
    color='r',
    label='Naive Ensemble')

    plt.ylabel('Accuracy (%)')
    plt.title('Feature Distribution Policy vs Naive Ensemble')
    plt.xticks(index + bar_width, ("ICMP Samples", "TCP Samples", "UDP Samples", "Whole Dataset"))
    plt.legend()

    global deleterand

    import os
    if os.path.exists("pyxtermjs/results" + deleterand + ".png"):
        os.remove("pyxtermjs/results" + deleterand + ".png")
    else:
        print("The file does not exist")

    plt.tight_layout()
    randinit = str(random.randrange(1000))
    plt.savefig('pyxtermjs/results'+ randinit + '.png',transparent = True)
    deleterand = randinit

    class Item(object):
        def __init__(self, name, pol_a, naive):
            self.name = name
            self.pol_a = pol_a
            self.naive = naive 
    items = [Item('ICMP Samples',tablelist.pop(0),tablelist.pop(0)),
            Item('TCP Samples',tablelist.pop(0),tablelist.pop(0)),
            Item('UDP Samples',tablelist.pop(0),tablelist.pop(0)),
            Item('Whole Dataset',tablelist.pop(0),tablelist.pop(0))]
    table = ItemTable(items)

    print(table.__html__())
    returnhtml = (table.__html__()) + "\n<br>\n<br>\n"
    returnhtml = returnhtml + "<img src='results" + randinit + ".png' alt='Accuracy Chart' class='center'>"
    print(returnhtml)
    return returnhtml


@app.route('/policy-b')
def combined_UDP():
    pout("\r\nYou chose Policy B")
    pout("Starting output stream.")
    # imp_dict = dict(zip(df.columns, forest.feature_importances_))
    # sorted_imp_dict = sorted((value, key) for (key,value) in imp_dict.items())

    # print(sorted_imp_dict)
    tablelist = []
    totalList = []
    #####Change here 0=ICMP, 1=TCP, 2=UDP
    for cpt in range(0,3):
        current_protocol_type = cpt #######CHANGE VALUE VIA FUNCTION

        frac_a_1 = ds_train_1[ds_train_1[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_1.shape[0]
        frac_a_2 = ds_train_2[ds_train_2[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_2.shape[0]
        frac_a_3 = ds_train_3[ds_train_3[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_3.shape[0]
        frac_a_4 = ds_train_4[ds_train_4[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_4.shape[0]
        frac_a_5 = ds_train_5[ds_train_5[:, 1] == current_protocol_type].shape[0] * 1.0 / ds_train_5.shape[0]

        pout("==============Frac_a==============")
        pout(str(frac_a_1))
        pout(str(frac_a_2))
        pout(str(frac_a_3))
        pout(str(frac_a_4))
        pout(str(frac_a_5))

        frac_b_1 = 0
        frac_b_2 = 0
        frac_b_3 = 0
        frac_b_4 = 0
        frac_b_5 = 0

        for column_idx in columns_idx_1:
            frac_b_1 += importances[column_idx]

        for column_idx in columns_idx_2:
            frac_b_2 += importances[column_idx]
            
        for column_idx in columns_idx_3:
            frac_b_3 += importances[column_idx]

        for column_idx in columns_idx_4:
            frac_b_4 += importances[column_idx]

        for column_idx in columns_idx_5:
            frac_b_5 += importances[column_idx]

        pout("==============Frac_b==============")
        pout(str(frac_b_1))
        pout(str(frac_b_2))
        pout(str(frac_b_3))
        pout(str(frac_b_4))
        pout(str(frac_b_5))

        frac_1 = frac_a_1 * frac_b_1
        frac_2 = frac_a_2 * frac_b_2
        frac_3 = frac_a_3 * frac_b_3
        frac_4 = frac_a_4 * frac_b_4
        frac_5 = frac_a_5 * frac_b_5

        pout("==============Frac_combined==============")
        pout(str(frac_1))
        pout(str(frac_2))
        pout(str(frac_3))
        pout(str(frac_4))
        pout(str(frac_5))

        ds_test = np.concatenate((x_test, y_test), axis=1)
        ds_test_temp = ds_test[ds_test[:, 1] == current_protocol_type]  
        ds_test_temp = ds_test_temp[:2000,:]
        x_test_temp, y_test_temp = np.hsplit(ds_test_temp, np.array([40]))

        usestring = "Num of samples: {0}".format(y_test_temp.shape[0])
        pout(usestring)
        x_test_1 = x_test_temp[:, columns_idx_1]
        x_test_2 = x_test_temp[:, columns_idx_2]
        x_test_3 = x_test_temp[:, columns_idx_3]
        x_test_4 = x_test_temp[:, columns_idx_4]
        x_test_5 = x_test_temp[:, columns_idx_5]

        pred_1 = model_1.predict(x_test_1)
        pred_2 = model_2.predict(x_test_2)
        pred_3 = model_3.predict(x_test_3)
        pred_4 = model_4.predict(x_test_4)
        pred_5 = model_5.predict(x_test_5)

        #####################################################################

        pred_policy_a = (frac_a_1 * np.array(pred_1) + 
                    frac_a_2 * np.array(pred_2) + 
                    frac_a_3 * np.array(pred_3) + 
                    frac_a_4 * np.array(pred_4) + 
                    frac_a_5 * np.array(pred_5))/ (frac_a_1 + frac_a_2 + frac_a_3 +frac_a_4 + frac_a_5)
        pred_policy_a = np.argmax(pred_policy_a,axis=1)

        #####################################################################

        pred_policy_b = (frac_b_1 * np.array(pred_1) + 
                    frac_b_2 * np.array(pred_2) + 
                    frac_b_3 * np.array(pred_3) + 
                    frac_b_4 * np.array(pred_4) + 
                    frac_b_5 * np.array(pred_5))/ (frac_b_1 + frac_b_2 + frac_b_3 +frac_b_4 + frac_b_5)
        pred_policy_b = np.argmax(pred_policy_b,axis=1)

        #####################################################################

        pred_policy_comb = (frac_1 * np.array(pred_1) + 
                    frac_2 * np.array(pred_2) + 
                    frac_3 * np.array(pred_3) + 
                    frac_4 * np.array(pred_4) + 
                    frac_5 * np.array(pred_5))/ (frac_1 + frac_2 + frac_3 +frac_4 + frac_5)
        pred_policy_comb = np.argmax(pred_policy_comb,axis=1)


        #####################################################################

        pred_naive = (np.array(pred_1) + np.array(pred_2) + np.array(pred_3) + np.array(pred_4) + np.array(pred_5)) / 5.0
        pred_naive = np.argmax(pred_naive,axis=1)

        #####################################################################

        y_eval = np.argmax(y_test_temp, axis=1)

        #####################################################################

        # #Average accuracy of combined ensembles: Only Policy A
        # pout('Accuracy-Policy_a')
        # accuracy = str(metrics.accuracy_score(y_eval, pred_policy_a))
        # pout(accuracy)

        #Average accuracy of combined ensembles: Only Policy B
        pout('Accuracy-Policy_b')
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_policy_b)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%")
        pout(accuracy)

        # #Average accuracy of combined ensembles: Combined Policies
        # pout('Accuracy-Combined_Policies')
        # accuracy = str(metrics.accuracy_score(y_eval, pred_policy_comb))
        # pout(accuracy)

        pout('Accuracy-naive')
        #Average accuracy of combined ensembles
        accuracy = str("{:.2f}".format((metrics.accuracy_score(y_eval, pred_naive)) * 100))
        totalList.append(accuracy)
        tablelist.append(accuracy + "%")
        pout(accuracy)

        # pout('Individual model accuracy')
        # pred_1 = np.argmax(pred_1,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_1))
        # pout(accuracy)

        # pred_2 = np.argmax(pred_2,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_2))
        # pout(accuracy)

        # pred_3 = np.argmax(pred_3,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_3))
        # pout(accuracy)

        # pred_4 = np.argmax(pred_4,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_4))
        # pout(accuracy)

        # pred_5 = np.argmax(pred_5,axis=1)
        # accuracy = str(metrics.accuracy_score(y_eval, pred_5))
        # pout(accuracy)

    tablelist.append(str("{:.2f}".format((float(totalList[0]) * 0.33) + (float(totalList[2]) * 0.33) + (float(totalList[4]) * 0.33)))+ "%")
    tablelist.append(str("{:.2f}".format((float(totalList[1]) * 0.33) + (float(totalList[3]) * 0.33) + (float(totalList[5]) * 0.33)))+ "%")
    class ItemTable(Table):
        name = Col('      ')
        pol_b = Col('Policy Based')
        naive = Col('Naive Ensemble')

    n_groups = 4
    labels =["ICMP Samples", "TCP Samples", "UDP Samples", "Whole Dataset"]
    imptotal = ("{:.2f}".format((float(totalList[0]) * 0.33) + (float(totalList[2]) * 0.33) + (float(totalList[4]) * 0.33)))
    imp_policy = (float(totalList[0]),float(totalList[2]),float(totalList[4]), float(imptotal))
    print(totalList)
    naivetotal = ("{:.2f}".format((float(totalList[1]) * 0.33) + (float(totalList[3]) * 0.33) + (float(totalList[5]) * 0.33)))
    naive = (float(totalList[1]),float(totalList[3]),float(totalList[5]), float(naivetotal))


    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.95

    rects1 = plt.bar(index,imp_policy, bar_width,
    alpha=opacity,
    color='b',
    label='Feature Importance Policy')

    rects2 = plt.bar(index + bar_width, naive, bar_width,
    alpha=opacity,
    color='r',
    label='Naive Ensemble')

    plt.ylabel('Accuracy (%)')
    plt.title('Feature Importance Policy vs Naive Ensemble')
    plt.xticks(index + bar_width, ("ICMP Samples", "TCP Samples", "UDP Samples", "Whole Dataset"))
    plt.legend()

    global deleterand

    import os
    if os.path.exists("pyxtermjs/results" + deleterand + ".png"):
        os.remove("pyxtermjs/results" + deleterand + ".png")
    else:
        print("The file does not exist")

    plt.tight_layout()
    randinit = str(random.randrange(1000))
    plt.savefig('pyxtermjs/results'+ randinit + '.png', transparent = True)
    deleterand = randinit



    class Item(object):
        def __init__(self, name, pol_b, naive):
            self.name = name
            self.pol_b = pol_b
            self.naive = naive 
    items = [Item('ICMP Samples',tablelist.pop(0),tablelist.pop(0)),
            Item('TCP Samples',tablelist.pop(0),tablelist.pop(0)),
            Item('UDP Samples',tablelist.pop(0),tablelist.pop(0)),
            Item('Whole Dataset',tablelist.pop(0),tablelist.pop(0))]
    table = ItemTable(items)

    print(table.__html__())
    returnhtml = (table.__html__())
    returnhtml = returnhtml + "<br>\n<br>\n<img src='results" + randinit + ".png' alt='Accuracy Chart' class='center'>"
    print(returnhtml)
    return returnhtml

if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
