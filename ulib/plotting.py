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


from sklearn.datasets import make_blobs, make_classification, \
                              make_circles, make_moons



def plot_decision_boundary(X, y, model, title='decision boundary'):
    """
    Plots decision boundary with coutours
    """
    fig = plt.figure(figsize=(10, 7), dpi=100)
    X1min, X2min = X.min(axis=0)
    X1max, X2max = X.max(axis=0)
    x1, x2 = np.meshgrid(np.linspace(X1min, X1max, 500),
                         np.linspace(X2min, X2max, 500))
    ypred = model.predict_proba(np.c_[x1.ravel(), x2.ravel()])
    ypred = ypred.reshape(x1.shape)
    
#     plt.contourf(x1, x2, ypred, alpha=.4, cmap='RdGy', levels=50)
#     plt.colorbar();


    contours = plt.contour(x1, x2, ypred, 10, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, alpha=0.1)
#     plt.imshow(ypred, extent=[X1min, X1max, X2min, X2max], origin='lower',
#            cmap='RdGy', alpha=.8)
    plt.colorbar()
    plt.axis(aspect='image');
    
    
    
    plt.title(title)
    plt.show()
    
    
def plot_decision_boundaries(X, y, model, model_nn1, model_nn2, title='decision boundary'):
    """
    Plots 3 decision boundaries with coutours
    """
    fig = plt.figure(figsize=(12, 4), dpi=100)
    X1min, X2min = X.min(axis=0)
    X1max, X2max = X.max(axis=0)
    x1, x2 = np.meshgrid(np.linspace(X1min, X1max, 500),
                         np.linspace(X2min, X2max, 500))
    X_grid = np.c_[x1.ravel(), x2.ravel()]
    ypred = model.predict(X_grid)
    ypred = ypred.reshape(x1.shape)
    
    y_hat_1 = model_nn1.predict_proba(X_grid).reshape(x1.shape)
    y_hat_2 = model_nn2.predict_proba(X_grid).reshape(x1.shape)
    
    y_pred_1 = model_nn1.predict(X_grid).reshape(x1.shape)
    y_pred_2 = model_nn2.predict(X_grid).reshape(x1.shape)

    plt.subplot(1, 3, 1)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, alpha=0.1)
    plt.contourf(x1, x2,  y_pred_1, alpha=.5, levels=50, cmap='Greys')
    plt.contour(x1, x2, ypred, alpha=1)
    
    contours_1 = plt.contour(x1, x2, 
                             y_hat_1, 3, colors='black')
    plt.clabel(contours_1, inline=True, fontsize=8)
    
    plt.title('Class 1 countour lines')
    
    
    
    plt.subplot(1, 3, 2)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, alpha=0.1)
    plt.contour(x1, x2, ypred, alpha=1)
    plt.contourf(x1, x2, 1-y_pred_2, alpha=.5, levels=50, cmap='Greys')
    contours_2 = plt.contour(x1, x2, 
                             1-y_hat_2, 3, colors='black')
    plt.clabel(contours_2, inline=True, fontsize=8)
    
    plt.title('Class 2 countour lines')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, alpha=0.1)
    plt.contourf(x1, x2,  y_pred_1, alpha=.5, levels=50, cmap='Greys')
    plt.contour(x1, x2, ypred, alpha=1)
    plt.contourf(x1, x2, 1-y_pred_2, alpha=.5, levels=50, cmap='Greys')
    
    plt.title(title)
    plt.show()


def plot_bndr(X, y, model, title='decision boundary'):
    """
    Plots decision boundary without coutours
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    x0, x1 = np.meshgrid(np.arange(-3, 3, 0.1),
                            np.arange(-3, 3, 0.1))
    xx0, xx1 = x0.ravel(), x1.ravel()

    X_grid = np.c_[xx0, xx1, ]
    
    y_hat_0 = model.predict(X_grid).reshape(x0.shape)

    plt.contour(x0, x1, y_hat_0, levels=[0], cmap='jet')
    
    X_s = X#model.ss.transform(X)
    ax.scatter(X_s[:, 0],
                X_s[:, 1],
                c=y,
                cmap=plt.cm.Paired, alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.show()



def plot_bndrs(X, y, model, model_nn1, model_nn2, title='decision boundary'):
    """
    Plots 3 decision boundaries without coutours
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    x0, x1 = np.meshgrid(np.arange(-3, 3, 0.1),
                            np.arange(-3, 3, 0.1))
    xx0, xx1 = x0.ravel(), x1.ravel()

    X_grid = np.c_[xx0, xx1, ]
    
    y_hat_0 = model.predict(X_grid).reshape(x0.shape)
    y_hat_1 = model_nn1.predict(X_grid).reshape(x0.shape)
    y_hat_2 = model_nn2.predict(X_grid).reshape(x0.shape)

    plt.contour(x0, x1, y_hat_1, levels=[0], cmap='inferno', alpha=0.6, label='nn1')
    plt.contour(x0, x1, y_hat_0, levels=[0], cmap='jet', label='nn')
    plt.contour(x0, x1, y_hat_2, levels=[0], cmap='inferno', alpha=0.6, label='nn2')
    
    plt.legend(['nn1', 'nn', 'nn2'])
    
    ax.scatter(X[:, 0],
                X[:, 1],
                c=y,
                cmap=plt.cm.Paired, alpha=0.5)
    plt.title(title)
    
    plt.show()


def plot_sigms(X, y, model, model_nn1, model_nn2, title='decision boundary'):
    """
    Plots membership functions in 2D
    """
  
    plt.figure(figsize=(6, 6), dpi=100)
    #   fig, ax = plt.subplots(figsize=(7, 7))
    
    x0 = np.arange(-5, 5, 0.1)
    x1 = np.zeros(x0.shape)
    
    #   x0, x1 = np.meshgrid(x0, x1)

    #   x0, x1 = np.meshgrid(np.arange(-3, 3, 0.1),
    #                          np.arange(-3, 3, 0.1))
    #   xx0, xx1 = x0.ravel(), x1.ravel()

    X_grid = np.c_[x0, x1, ]
    
    #   X_grid = ss.transform(X_grid)
    #   X = ss.transform(X)
    
    y_hat_0 = model.predict_proba(X_grid).reshape(x0.shape)
    y_hat_1 = model_nn1.predict_proba(X_grid).reshape(x0.shape)
    y_hat_2 = model_nn2.predict_proba(X_grid).reshape(x0.shape)
    
    print(x0.shape)
    print(y_hat_0.shape)

    plt.plot(x0.ravel(), y_hat_1.ravel(), alpha=1, label='nn1')
    plt.plot(x0.ravel(), y_hat_0.ravel(), alpha=1, label='nn0')
    plt.plot(x0.ravel(), y_hat_2.ravel(), alpha=1, label='nn2')
    plt.legend()
    
    #   ax.scatter(X[:, 0],
    #              X[:, 1],
    #              c=y,
    #              cmap=plt.cm.Paired, alpha=0.5)
    plt.title(title)
    plt.show()



