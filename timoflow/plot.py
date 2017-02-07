"""
A little neural network toolbox

(c) Timo Flesch, 2017

"""

import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_error(modID,data,figHandle,spID,lRate):
    """
    helper function to plot error (1-accuracy)
    INPUTS:
    - modID:    ID of model
    - data:     log file
    - figHandle: ID of figure
    - spID:      ID of subplot
    - lRate:     learning rate
    """
    fig = plt.figure(figHandle,figsize=(15,5))
    fig.patch.set_facecolor('white')
    plt.subplot(1,2,spID)
    plt.plot(1-np.asarray(data['accuracy_train']))#[0::3])
    plt.plot(1-np.asarray(data['accuracy_test']))#[0::3])
    plt.title('Error, Model: %d, LR: %.4f'%(modID,lRate))
    plt.xlabel('Epoch')
    plt.ylabel('1-Accuracy')
    plt.ylim([0,1])
    plt.legend(['Training','Test'])

def plot_xen(modID,data,figHandle,spID,lRate):
    """
    helper function to plot cross entropy
    INPUTS:
    - modID:    ID of model
    - data:     log file
    - figHandle: ID of figure
    - spID:      ID of subplot
    - lRate:     learning rate
    """
    fig = plt.figure(figHandle,figsize=(15,5))
    fig.patch.set_facecolor('white')
    plt.subplot(1,2,spID)
    plt.plot(np.asarray(data['cross_entropy_train']))#[0::3])
    plt.plot(np.asarray(data['cross_entropy_test']))#[0::3])
    plt.title('XEn, Model: %d, LR: %.4f'%(modID,lRate))
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy')
    plt.legend(['Training','Test'])


def plot_confmat(modID,cMat,figHandle,titleString):
    """
    helper function to plot confusion matrix.
    INPUTS:
    - modID:    ID of model
    - cMat:     confusion matrix
    - figHandle: figure id
    - titleString: 'train' or 'test'
    """
    fig = plt.figure(figHandle,figsize=(8,6))
    fig.patch.set_facecolor('white')
    # plot
    plt.imshow(cMat,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title('Confusion Matrix, Model %s, %s'%(modID,titleString))
    # plt.xticks(np.arange(10),[0,1,2,3,4,5,6,7,8,9])
    # plt.yticks(np.arange(10),[0,1,2,3,4,5,6,7,8,9])
    cb = plt.colorbar()
    cb.set_label('# of Classications')

    # add numerical values, color depends on magnitude.
    # This is a litttle snippet I found online
    # (http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
    cm = cMat
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
