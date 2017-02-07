"""
A little neural network toolbox

(c) Timo Flesch, 2017

"""
import pickle
from timoflow import nnet
def save_model(myModel,modelIDX,modelDir,isPy3):
    """
    saves model on harddisk
    """
    fileName = modelDir + 'model' + str(modelIDX)
    modelFile = {'layers':myModel.layers, 'loss': myModel.loss}

    with open(fileName, 'wb') as f:
        if isPy3:
            pickle.dump(modelFile,f,protocol=2)
        else:
            pickle.dump(modelFile,f)


def save_log(log_dict,modelIDX,logDir,isPy3):
    """
    saves log on harddisk
    """
    fileName = logDir + 'model' + str(modelIDX)
    with open(fileName, 'wb') as f:
        if isPy3:
            pickle.dump(log_dict,f,protocol=2)
        else:
            pickle.dump(log_dict,f)


def load_model(modelIDX,modelDir):
    """
    loads model
    """
    fileName = modelDir + 'model' + str(modelIDX)
    with open(fileName, 'rb') as f:
        modelFile = pickle.load(f)
        myModel = nnet.myNet(modelFile)
        return myModel
