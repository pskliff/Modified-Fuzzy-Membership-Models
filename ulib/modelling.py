import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import categorical_accuracy, binary_accuracy,mae, mape
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import r2_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb


def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s


def make_lgbm_model(config):
  if config['is_classification']:
    gbm = lgb.LGBMClassifier(boosting_type=config['boosting_type'],
                             objective='binary',
                             metric=config['metric'],
                             n_estimators=config['n_estimators'],
                             learning_rate=config['learning_rate'],
                             max_depth=config['max_depth'],
                             verbosity=config['verbose'], 
                             num_threads=config['num_threads'],
                             monotone_constraints=config['monotone_constraints'])
  else:
    
    gbm = lgb.LGBMRegressor(boosting_type=config['boosting_type'],
                             objective='regression',
                             metric=config['metric'],
                             n_estimators=config['n_estimators'],
                             learning_rate=config['learning_rate'],
                             max_depth=config['max_depth'],
                             verbosity=config['verbose'], 
                             num_threads=config['num_threads'],
                             monotone_constraints=config['monotone_constraints'])
    
  return gbm


def make_rf_model(config):
  if config['is_classification']:
    rf = RandomForestClassifier(n_estimators=config['n_estimators'])
  else:
    rf = RandomForestRegressor(n_estimators=config['n_estimators'])
    
  return rf


def make_binary_model(config):
    """
    Fast model to test hypothesis
    """
    model = Sequential()
    batch_size = config['batch_size']
    first_layer_out = config['dense_layers'][0]
    first_layer_in = config["input_dim"]
    
    if first_layer_out > 0:
      model.add(Dense(first_layer_out, input_dim=(first_layer_in),
                      kernel_initializer='normal',
                      activation=config['activation'][0]))
    
      if len(config["dense_layers"]) > 1:
        for i, ls in enumerate(config["dense_layers"]):
          model.add(Dense(ls, kernel_initializer=config['kernel_init'][i],
                          activation=config['activation'][i]))

      model.add(Dense(config['num_classes'], activation="sigmoid"))
    else:
      model.add(Dense(config['num_classes'], input_dim=(first_layer_in),
                      kernel_initializer='normal', activation=config['last_activation']))
    
    return model

