from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import cPickle as pickle
import numpy as np
import pandas as pd
import librosa
import librosa.feature as lfr
import glob
import os
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution1D, UpSampling2D, MaxPooling1D, Embedding, LSTM
from keras.preprocessing import sequence
from keras.layers.recurrent import GRU

%matplotlib inline

def get_file_num(fn):
    return fn.split('/')[-1].split('.')[0]

def get_labels():
    parent_dir="/Users/zhouqiang/Downloads/urbansound/UrbanSound/data"
    sub_dirs=['air_conditioner','children_playing','drilling','gun_shot','siren', 'car_horn','dog_bark','engine_idling','jackhammer','street_music']
    labels = {'air_conditioner':1,'children_playing':2,'drilling':3,'gun_shot':4,'siren':5,
             'car_horn':6,'dog_bark':7,'engine_idling':8,'jackhammer':9,'street_music':10
            }

    wav_file_ext, mp3_file_ext = "*.wav", "*.mp3"
    labels_dict = dict()

    result = []
    for sub_dir in sub_dirs:
        label = labels[sub_dir]
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, wav_file_ext)):
            fn = get_file_num(fn)
            labels_dict[fn]=label

    for sub_dir in sub_dirs:
        label = labels[sub_dir]
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, mp3_file_ext)):
            fn = get_file_num(fn)
            labels_dict[fn]=label
    return labels_dict
 
labels_dict = get_labels() 
   
def get_sequence_from_file():
    pf=open('/Users/zhouqiang/Downloads/urbansound/UrbanSound/src/mfcc.pkl', 'r')
    mfccs=[]
    file_names = []
    mfcc_count = 0
    X = []
    y = []
    labels = []
    start_index = []
    try:
        start_row_index = 0
        for i in range(1106):
        #for i in range(6):
            mfcc_count += 1
            tup=pickle.load(pf)

            # tup[0] is file name, it only has the numbers as a string
            # tup[1] is mfcc vectors, its shape is like (20, ...)
            file_name = tup[0]
            label = labels_dict[file_name]
            file_names.append( file_name )
            mfcc = np.mean(tup[1], axis=0).flatten()
#             print tup[1].shape
            for m in mfcc:
                X.append( m )
                y.append( file_name )
                labels.append(label)
            start_index.append( start_row_index )
            start_row_index += mfcc.shape[0]
        X = np.array(X).reshape(-1, 1)
#         print X.shape
        cluster_labels = get_cluster_for_points(X)
        y = np.array( y )
        df = pd.DataFrame( { "x": pd.Series( cluster_labels ) , "y": pd.Series(y) } )
        return df, file_names, labels
    except EOFError:
        print mfcc_count
    

def get_cluster_for_points(X):
    components = []
    kmeans = KMeans(n_clusters=100, random_state=0).fit(X)
    return kmeans.labels_

