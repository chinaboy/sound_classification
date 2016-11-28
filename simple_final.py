import soundfile as sf
from sklearn.feature_selection import SelectKBest
from operator import itemgetter
import copy
import logging
import glob
import os
import cPickle as pickle
from librosa import feature as ftr
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
%matplotlib inline

def generate_features_from_all_files():
    parent_dir="/Users/zhouqiang/Downloads/urbansound/UrbanSound/data"
    sub_dirs=['air_conditioner','children_playing','drilling','gun_shot','siren', 'car_horn','dog_bark','engine_idling','jackhammer','street_music']
    labels = {'air_conditioner':1,'children_playing':2,'drilling':3,'gun_shot':4,'siren':5,
             'car_horn':6,'dog_bark':7,'engine_idling':8,'jackhammer':9,'street_music':10
            } 
    wav_file_ext, mp3_file_ext = "*.wav", "*.mp3"
    wav_lst, mp3_lst = [], []
    for sub_dir in sub_dirs:
        label = labels[sub_dir]
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, wav_file_ext)):
            wav_lst.append( (fn, label) )
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, mp3_file_ext)):
            mp3_lst.append( (fn , label))
    #TODO pickle two lists
    process_lst(wav_lst, mp3_lst)
    return wav_lst, mp3_lst

# read in wav and mp3
# for 24 bit, we use soundfile
# for mp3, we use librosa
#
def process_lst(wav_lst, mp3_lst):
    mp3_error = []
    logging.basicConfig(filename='24bit_error.log',level=logging.DEBUG)
    features_per_file = []
    for wav, label in wav_lst:
        try:
            data, rate = librosa.load(wav)
            features_per_file.append( extract_feature_per_file(data, rate, wav, label) )
        except Exception as e:
            logging.warning(wav)
            s_data, s_rate = sf.read(wav)
            features_per_file.append( extract_feature_per_file(s_data, s_rate, wav, label) )
    for mp, label in mp3_lst:
        try:
            mp_data, mp_rate  = librosa.load(mp)
            features_per_file.append( extract_feature_per_file(mp_data, mp_rate, mp, label) )
        except Exception as e:
            mp3_error.append(mp)
    # put list into dataframe and then pickle the df
    df = pd.DataFrame( features_per_file, columns = ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 'num', 'label'])
    df.to_pickle('simple_features_final.pkl')        

def extract_feature_per_file(data, sample_rate, path_to_file, label):
    sh = data.shape
    if len(sh) == 2:
        data = data[:, 1]
    if len(sh)>2:
        print "more than 2 channels"
        return
    stft = np.abs(librosa.stft(data))
    mfcc = np.mean( ftr.mfcc(y=data, sr=sample_rate).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T,axis=0)
    num = get_file_num(path_to_file)
    return [ mfcc, chroma, mel, contrast, tonnetz, num, label ]

def get_file_num(fn):
    return fn.split('/')[-1].split('.')[0]
    
def load_features():
    f = open('simple_features_final.pkl', 'rb')
    df=pickle.load(f)
    return df
 
# X_train has n columns. Train a model on every n-1 columns of data. Remember, take the same column out of test data.
def train_backward_selection(X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]
    max_f = min(8, n_features)
    n_trees = min(128, n_features*n_features)
    cls=RandomForestClassifier(n_estimators=n_trees, max_features=max_f)
    cls.fit(X_train, y_train)
    predict_y = cls.predict(X_test)
    score = accuracy_score( y_test,  predict_y )
    return np.argsort(cls.feature_importances_).tolist()[0], score

def select_features(X, y):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

    feature_list_snapshot = []
    features = [ col for col in X_train.columns ]
    print features
    while len(features)>1:
        batch_X_train = X_train[features]
        batch_X_test = X_test[features]

        throw_away, score = train_backward_selection(batch_X_train, batch_X_test, np.asarray(y_train, dtype="int"),  np.asarray(y_test, dtype="int"))
        feature_list_snapshot.append( (copy.deepcopy(features), score) )

        col_name_throw = batch_X_train.columns[throw_away]
        print col_name_throw
        features.remove( col_name_throw )
        
    best_features_set = sorted(feature_list_snapshot, key=itemgetter(1))[-1]
    print best_features_set[0]
    return feature_list_snapshot
       
def main():
    # Kick off the show
    generate_features_from_all_files()

    df = load_features()
    add_columns(df, 20, 'mfcc')
    add_columns(df, 12, 'chroma')
    add_columns(df, 128, 'mel')
    add_columns(df, 7, 'contrast')
    add_columns(df, 6, 'tonnetz')
    df=df.drop(['chroma', 'mel', 'contrast', 'tonnetz', 'mfcc'], axis=1)
    y=df['label']
    X=df.drop(['num', 'label'], axis=1)

    # feature selection
    snapshot=select_features(X, y)
    for s in snapshot:
        print s