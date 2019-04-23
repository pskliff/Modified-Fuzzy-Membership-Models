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


class ml_classifier:
  def __init__(self, config):
    self.config = config
    self.get_model_method = config['get_model_method']
#     self.threshold = threshold
    self.model = None
    self.ss = None
    self.best_score = 0
    self.best_thres = 0.5
    self.thres_chosen = False
    
    
  def fit(self, X, y):
    ss = self.config['scaler']
    X = ss.fit_transform(X)


    # get model
    model = self.get_model_method(self.config)

    if self.config["verbose"]:
      print(model)


    # prepare model for fitting (loss, optimizer, etc)

    model.fit(X, y)
    
    self.model = model
    self.ss = ss
    
    
    # TODO: train test split
    if not self.thres_chosen:
      self._choose_threshold(X, y)
    
    return model, ss
  
  
  def predict_proba(self, X):
    X_t = self.ss.transform(X)
#     print('keras predict')
#     print(self.model.predict(X_t))
    y_proba = self.model.predict_proba(X_t)[:, 1]
#     print('Proba')
#     print(y_proba)
    
    return y_proba
  
  
  def predict(self, X):
    y_proba = self.predict_proba(X)
    return (y_proba > self.best_thres).astype(int)
  
   
  def _choose_threshold(self, X, y):
    y_proba = self.predict_proba(X).reshape(X.shape[0])
    best_score = 0
    best_thres = 0
    y_true = (y > 0.5).astype(int) # `y` is in [0, 1]??

    for th in np.arange(0.1, 1.01, 0.05):
      y_hat = (y_proba > th).astype(int)

      score = f1_score(y_true, y_hat)

      if score > best_score:
        best_score = score
        best_thres = th
    
    self.best_score, self.best_thres = best_score, best_thres
    return best_score, best_thres
  
  
  def get_thres_score(self):
    return self.best_thres, self.best_score

  
  
  
  
class ml_regressor:
  def __init__(self, config):
    self.config = config
    self.get_model_method = config['get_model_method']
#     self.threshold = threshold
    self.model = None
    self.ss = None
    self.best_score = 0
    self.best_thres = 0.5
    self.thres_chosen = False
    
    
  def fit(self, X, y):
    ss = self.config['scaler']
    X = ss.fit_transform(X)


    # get model
    model = self.get_model_method(self.config)

    if self.config["verbose"]:
      print(model)


    model.fit(X, y)
    
    self.model = model
    self.ss = ss
    
    
    # TODO: train test split
    if not self.thres_chosen:
      self._choose_threshold(X, y)
    
    return model, ss
  
  
  def predict_proba(self, X):
    X_t = self.ss.transform(X)
    y_proba = self.model.predict(X_t)
    mask = np.arange(len(y_proba))[y_proba < 0]
    y_proba[mask] = 0
    
    return y_proba
  
  
  def predict(self, X):
    y_proba = self.predict_proba( X)
    return (y_proba > self.best_thres).astype(int)
  
   
  def _choose_threshold(self, X, y):
    y_proba = self.predict_proba(X).reshape(X.shape[0])
    best_score = 0
    best_thres = 0
    y_true = (y > 0.5).astype(int) # `y` is in [0, 1]??

    for th in np.arange(0.1, 1.01, 0.05):
      y_hat = (y_proba > th).astype(int)

      score = f1_score(y_true, y_hat)

      if score > best_score:
        best_score = score
        best_thres = th
    
    self.best_score, self.best_thres = best_score, best_thres
    return best_score, best_thres
  
  
  def get_thres_score(self):
    return self.best_thres, self.best_score


class dl_classifier:
  def __init__(self, config):
    self.config = config
    self.get_model_method = config['get_model_method']
#     self.threshold = threshold
    self.model = None
    self.ss = None
    self.best_score = 0
    self.best_thres = 0.5
    self.thres_chosen = False
    
    
  def fit(self, X, y):
    ss = self.config['scaler']
    X = ss.fit_transform(X)

    if self.config['reset_session']:
      s = reset_tf_session()

    # get model
    model = self.get_model_method(self.config)

    if self.config["verbose"]:
      model.summary()


    # prepare model for fitting (loss, optimizer, etc)
    model.compile(
        loss=self.config['loss'],
        optimizer=Adam(),
        metrics=self.config['metrics']
    )

    epochs = self.config['epochs']
    batch_size = self.config['batch_size']

    model.fit(X, y, epochs=epochs, batch_size=batch_size, 
              verbose=self.config['verbose'])
    
    self.model = model
    self.ss = ss
    
    
    # TODO: train test split
    if not self.thres_chosen:
      self._choose_threshold(X, y[:, 1])
    
    return model, ss
  
  
  def predict_proba(self, X):
    X_t = self.ss.transform(X)
#     print('keras predict')
#     print(self.model.predict(X_t))
    y_proba = self.model.predict(X_t)[:, 1].flatten()
#     print('Proba')
#     print(y_proba)
    
    return y_proba
  
  
  def predict(self, X):
    y_proba = self.predict_proba(X)
    return (y_proba > self.best_thres).astype(int)
  
   
  def _choose_threshold(self, X, y):
    y_proba = self.predict_proba(X).reshape(X.shape[0])
    best_score = 0
    best_thres = 0
    y_true = (y > 0.5).astype(int) # `y` is in [0, 1]??

    for th in np.arange(0.1, 1.01, 0.05):
      y_hat = (y_proba > th).astype(int)

      score = f1_score(y_true, y_hat)

      if score > best_score:
        best_score = score
        best_thres = th
    
    self.best_score, self.best_thres = best_score, best_thres
    return best_score, best_thres
  
  
  def get_thres_score(self):
    return self.best_thres, self.best_score

  
  
  
  
class dl_regressor:
  def __init__(self, config):
    self.config = config
    self.get_model_method = config['get_model_method']
#     self.threshold = threshold
    self.model = None
    self.ss = None
    self.best_score = 0
    self.best_thres = 0.5
    self.thres_chosen = False
    
    
  def fit(self, X, y):
    ss = self.config['scaler']
    X = ss.fit_transform(X)

    if self.config['reset_session']:
      s = reset_tf_session()

    # get model
    model = self.get_model_method(self.config)

    if self.config["verbose"]:
      model.summary()


    # prepare model for fitting (loss, optimizer, etc)
    model.compile(
        loss=self.config['loss'],
        optimizer=Adam(),
        metrics=self.config['metrics']
    )

    epochs = self.config['epochs']
    batch_size = self.config['batch_size']

    model.fit(X, y, epochs=epochs, batch_size=batch_size, 
              verbose=self.config['verbose'], validation_split=0.1)
    
    self.model = model
    self.ss = ss
    
    
    # TODO: train test split
    if not self.thres_chosen:
      self._choose_threshold(X, y)
    
    return model, ss
  
  
  def predict_proba(self, X):
    X_t = self.ss.transform(X)
    y_proba = self.model.predict(X_t).flatten()
    mask = np.arange(len(y_proba))[y_proba < 0]
    y_proba[mask] = 0
    
    return y_proba
  
  
  def predict(self, X):
    y_proba = self.predict_proba( X)
    return (y_proba > self.best_thres).astype(int)
  
   
  def _choose_threshold(self, X, y):
    y_proba = self.predict_proba(X).reshape(X.shape[0])
    best_score = 0
    best_thres = 0
    y_true = (y > 0.5).astype(int) # `y` is in [0, 1]??

    for th in np.arange(0.1, 1.01, 0.05):
      y_hat = (y_proba > th).astype(int)

      score = f1_score(y_true, y_hat)

      if score > best_score:
        best_score = score
        best_thres = th
    
    self.best_score, self.best_thres = best_score, best_thres
    return best_score, best_thres
  
  
  def get_thres_score(self):
    return self.best_thres, self.best_score