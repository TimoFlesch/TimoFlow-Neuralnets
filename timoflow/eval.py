"""
A little neural network toolbox

(c) Timo Flesch, 2017

"""
from sklearn.metrics import confusion_matrix
import numpy as np
def computeLossAcc(xData,yData,myModel):
    """
    computes loss and accuracy
    """
    y_hat = np.zeros(yData.shape)
    loss = np.zeros(xData.shape[0])
    y_hat = myModel.fprop(xData.T)
    muLoss =  myModel.loss.fprop(y_hat,yData)
    muAcc = np.mean(np.equal(np.argmax(y_hat,axis=0),np.argmax(yData,axis=0).astype('int')))
    return [muLoss,muAcc]

def computeConfMat(xData,yData,myModel):
    """
    computes confusion matrix
    """
    yHat = np.argmax(myModel.fprop(xData.T),axis=0)
    yTrue = np.argmax(yData.T,axis=0)

    return confusion_matrix(yTrue,yHat)
