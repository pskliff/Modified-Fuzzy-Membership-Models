import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

import os
import sys
sys.path.append(os.getcwd())

from .modelling import *
from .pipelines import *
from .plotting import *

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

from sklearn.datasets import make_blobs, make_classification, \
                              make_circles, make_moons



def ml_experiment(nn_config, nn1_config, nn2_config, 
                  X, y, lmbd=-0.5, thresholds=[0.5, 0.5, 0.5], verbose=False,
                 left_proba=0.4, right_proba=0.6):
    
    """
    Runs FMM algorithm with machine learning models as approximators
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True, stratify=y,
                                                        random_state=SEED)
    
    # STEP 1
    if nn_config['is_classification']:
        model = ml_classifier(nn_config)
        _, ss = model.fit(X_train, y_train)
    else:
        model = rf_regressor(nn_config)
        _, ss = model.fit(X_train, y_train)
    
    ## plot nn decision boundary
    X_train = ss.transform(X_train)
    
    y_proba = model.predict_proba(X_train)
    thres, score = model.get_thres_score()
    
    plot_bndr(X_train, y_train, model, 
                title=f'nn boundary: thres={thres}, f1_score={score}')
    
    plot_decision_boundary(X_train, y_train, model, 
                title=f'nn boundary: thres={thres}, f1_score={score}')
    
    ## Print NN scores
    y_pred = model.predict(X_test)
    
    print(y_proba)
    print(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'\nNN acc = {acc}, f1 = {f1}')
    
    # STEP 2
    #   y_proba = dl_predict_proba(model, ss, X_train)
    
    
    y_pred = model.predict(X_train)
    
    
    s1_mask = (y_train==0) & (y_proba >= left_proba)
    s2_mask = (y_train==1) & (y_proba < right_proba)
    
    s1_idx = np.arange(len(y_train))[s1_mask]
    s2_idx = np.arange(len(y_train))[s2_mask]
    
    X_train_s1 = X_train[s1_idx, :]
    X_train_s2 = X_train[s2_idx, :]
    
    # STEP 3
    ## TODO
    
    # STEP 4
    y_train_s1 = y_proba[s1_idx]
    y_train_s2 = y_proba[s2_idx]
    
    # STEP 5
    ## already normalized
    
    # STEP 6
    y_train_s1 = y_train_s1
    y_train_s2 = 1 - y_train_s2
    
    
    # STEP 7
    y_train_s1_c = 1 - (1 - y_train_s1)/(1 + lmbd*y_train_s1)
    y_train_s2_c = (1 - y_train_s2)/(1 + lmbd*y_train_s2)
    
    
    # Plot nn1 and nn2 datasets
    plt.scatter(X_train_s1[:, 0], X_train_s1[:, 1], 
                c=(y_train_s1_c > thresholds[1]).astype(int))
    plt.title(f'Misclassification Dataset for NN0, shape={s1_idx.shape}')
    plt.show()
    
    plt.scatter(X_train_s2[:, 0], X_train_s2[:, 1],
                c=(y_train_s2_c > thresholds[2]).astype(int))
    plt.title(f'Misclassification Dataset for NN1, shape={s2_idx.shape}')
    plt.show()
    
    # STEP 8
    ## no need
    
    
    # STEP 9
    ## Train nn1 model on nn1 dataset
    model_nn1 = ml_regressor(nn1_config)
    
    _, ss_nn1 = model_nn1.fit(X_train_s1, y_train_s1_c)
    
    y_proba = model_nn1.predict_proba(X_train_s1)
    thres1, score1 = model_nn1.get_thres_score()
    
    
    # STEP 10
    ## Train nn1 model on nn1 dataset
    model_nn2 = ml_regressor(nn2_config)
    _, ss_nn2 = model_nn2.fit(X_train_s2, y_train_s2_c)
    
    y_proba = model_nn2.predict_proba(X_train_s2)
    thres2, score2 = model_nn2.get_thres_score()
    
    # Plot Results
    ## NN 1
    plot_bndr(X_train_s1, (y_train_s1_c>0.5).astype(int), model_nn1, 
        title=f'NN0 boundary on NN0 dataset: thres={thres1}, f1_score={score1}')
    
    plot_decision_boundary(X_train_s1, (y_train_s1_c>0.5).astype(int), model_nn1, 
        title=f'NN0 boundary on NN0 dataset: thres={thres1}, f1_score={score1}')
    
    ## NN2
    plot_bndr(X_train_s2, (y_train_s2_c>0.5).astype(int), model_nn2, 
        title=f'NN1 boundary on NN1 dataset: thres={thres2}, f1_score={score2}')
    
    plot_decision_boundary(X_train_s2, (y_train_s2_c>0.5).astype(int), model_nn2, 
        title=f'NN1 boundary on NN1 dataset: thres={thres2}, f1_score={score2}')
    
    ## NN both
    plot_bndrs(X_train, y_train, model, model_nn1, model_nn2, 
                title='Crisp Decision with Fuzzy boundaries')
    
    plot_decision_boundaries(X_train, y_train, model, model_nn1, model_nn2, 
                title='Crisp Decision with Fuzzy boundaries')
    
    plot_sigms(X_train, y_train, model, model_nn1, model_nn2, 
                title='Crisp Decision with Fuzzy boundaries 1D')




def dl_experiment(nn_config, nn1_config, nn2_config, 
                  X, y, lmbd=-0.5, thresholds=[0.5, 0.5, 0.5], verbose=False,
                 left_proba=0.4, right_proba=0.6):
    """
    Runs FMM algorithm with deep learning models as approximators
    """
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True, stratify=y,
                                                        random_state=SEED)
    
    # STEP 1
    if nn_config['is_classification']:
        model = dl_classifier(nn_config)
        _, ss = model.fit(X_train, to_categorical(y_train))
    else:
        model = dl_regressor(nn_config)
        _, ss = model.fit(X_train, y_train)
    
    ## plot nn decision boundary
    X_train = ss.transform(X_train)
    
    y_proba = model.predict_proba(X_train)
    thres, score = model.get_thres_score()
    
    plot_bndr(X_train, y_train, model, 
                title=f'nn boundary: thres={thres}, f1_score={score}')
    
    plot_decision_boundary(X_train, y_train, model, 
                title=f'nn boundary: thres={thres}, f1_score={score}')
    
    ## Print NN scores
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'\nNN acc = {acc}, f1 = {f1}')
    
    # STEP 2
    #   y_proba = dl_predict_proba(model, ss, X_train)
    
    
    y_pred = model.predict(X_train)
    
    
    s1_mask = (y_train==0) & (y_proba >= left_proba)
    s2_mask = (y_train==1) & (y_proba < right_proba)
    
    s1_idx = np.arange(len(y_train))[s1_mask]
    s2_idx = np.arange(len(y_train))[s2_mask]
    
    X_train_s1 = X_train[s1_idx, :]
    X_train_s2 = X_train[s2_idx, :]
    
    # STEP 3
    ## TODO
    
    # STEP 4
    y_train_s1 = y_proba[s1_idx]
    y_train_s2 = y_proba[s2_idx]
    
    # STEP 5
    ## already normalized
    
    # STEP 6
    y_train_s1 = y_train_s1
    y_train_s2 = 1 - y_train_s2
    
    
    # STEP 7
    y_train_s1_c = 1 - (1 - y_train_s1)/(1 + lmbd*y_train_s1)
    y_train_s2_c = (1 - y_train_s2)/(1 + lmbd*y_train_s2)
    
    
    # Plot nn1 and nn2 datasets
    plt.scatter(X_train_s1[:, 0], X_train_s1[:, 1], 
                c=(y_train_s1_c > thresholds[1]).astype(int))
    plt.title(f'Misclassification Dataset for NN0, shape={s1_idx.shape}')
    plt.show()
    
    plt.scatter(X_train_s2[:, 0], X_train_s2[:, 1],
                c=(y_train_s2_c > thresholds[2]).astype(int))
    plt.title(f'Misclassification Dataset for NN1, shape={s2_idx.shape}')
    plt.show()
    
    # STEP 8
    ## no need
    
    
    # STEP 9
    ## Train nn1 model on nn1 dataset
    model_nn1 = dl_regressor(nn1_config)
    
    _, ss_nn1 = model_nn1.fit(X_train_s1, y_train_s1_c)
    
    y_proba = model_nn1.predict_proba(X_train_s1)
    thres1, score1 = model_nn1.get_thres_score()
    
    
    # STEP 10
    ## Train nn1 model on nn1 dataset
    model_nn2 = dl_regressor(nn2_config)
    _, ss_nn2 = model_nn2.fit(X_train_s2, y_train_s2_c)
    
    y_proba = model_nn2.predict_proba(X_train_s2)
    thres2, score2 = model_nn2.get_thres_score()
    
    # Plot Results
    ## NN 1
    plot_bndr(X_train_s1, (y_train_s1_c>0.5).astype(int), model_nn1, 
        title=f'NN0 boundary on NN0 dataset: thres={thres1}, f1_score={score1}')
    
    plot_decision_boundary(X_train_s1, (y_train_s1_c>0.5).astype(int), model_nn1, 
        title=f'NN0 boundary on NN0 dataset: thres={thres1}, f1_score={score1}')
    
    ## NN2
    plot_bndr(X_train_s2, (y_train_s2_c>0.5).astype(int), model_nn2, 
        title=f'NN1 boundary on NN1 dataset: thres={thres2}, f1_score={score2}')
    
    plot_decision_boundary(X_train_s2, (y_train_s2_c>0.5).astype(int), model_nn2, 
        title=f'NN1 boundary on NN1 dataset: thres={thres2}, f1_score={score2}')
    
    ## NN both
    plot_bndrs(X_train, y_train, model, model_nn1, model_nn2, 
                title='Crisp Decision with Fuzzy boundaries')
    
    plot_decision_boundaries(X_train, y_train, model, model_nn1, model_nn2, 
                title='Crisp Decision with Fuzzy boundaries')
    
    plot_sigms(X_train, y_train, model, model_nn1, model_nn2, 
                title='Crisp Decision with Fuzzy boundaries 1D')
  
  