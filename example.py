"""
A simple neural network example
This is just a sample code to explain the syntax
(c) Timo Flesch, 2017

"""
import numpy as np
import random
import timoflow as tif
from sys import version_info

isPy3 = version_info[0] > 2


train_lrate      =                     0.01    # which learning rate?
train_numEpochs  =                       25    # how many epochs?
train_batchSize  =                      100    # size of minibatch
train_numSamples =                    10000    # number of samples
train_mode       =                 'minibatch' # online vs minibatch

def nnet_setup():
        """
        main function for neural network setup. Calls my custom "timoflow" scripts
        OUTPUT:
        - myModel: instance of timoflow ann with desired layer structure and loss
        """

    ## here, I define a sample neural network:
    ## the network provides a function from 256 input nodes to 3 class nodes.
    ## it has two hidden relu layers with 128 & 64 nodes and a
    ## softmax activation function.
    ## as loss, I use cross entropy
    myModel = tif.nnet.myNet({
                            'layers':[tif.nnet.module_linear(256,128),
                                        tif.nnet.module_relu(128),
                                        tif.nnet.module_linear(128,64),
                                        tif.nnet.module_relu(64),
                                        tif.nnet.module_linear(64,3),
                                        tif.nnet.module_softmax(3)],
                            'loss':tif.nnet.module_xent(3)})

    return myModel

def nnet_train(x_train,y_train,x_test,y_test,myModel,train_lrate=5e-3, train_mode='online',train_batchSize=100):
    """
    trains model with SGD
    INPUT
    - x_train:         the inputs
    - y_train:         the labels
    - myModel:         the neural network (list of layer objects)
    - train_lrate:     learning rate of model
    - train_mode:      'online' or 'minibatch'
    - train_batchSize: size of minibatch
    """

    if train_mode=='online':
        for ii_epoch in range(train_numEpochs):
            # obtain shuffle indices
            ii_shuff = np.random.permutation(x_train.shape[0])
            # iterate through sampels
            for ii_iter in range(x_train.shape[0]):
                # perform forward pass through network:
                y_hat = myModel.fprop(x_train[ii_iter,:,None])
                # compute loss
                loss = myModel.loss.fprop(y_hat,y_train[ii_iter,:,None])

                # propagate error backwards (compute gradients) & update weights
                myModel.bprop(y_hat,y_train[ii_iter,:,None],train_lrate)
                # log summary each epoch
                if ii_iter==0:
                    results_train = tif.eval.computeLossAcc(x_train,y_train.T,myModel)
                    results_test = tif.eval.computeLossAcc(x_test,y_test.T,myModel)
                    xenTrain.append(results_train[0])
                    accTrain.append(results_train[1])
                    xenTest.append(results_train[0])
                    accTest.append(results_train[1])
                    print('Epoch %d, AccTrain: %.4f, AccTest: %.4f -- XenTrain: %3.2f, XenTest: %3.2f'%(ii_epoch+1,results_train[1],results_test[1],results_train[0],results_test[0]))



    elif train_mode=='minibatch':
        for ii_epoch in range(train_numEpochs):
            # obtain shuffle indices
            ii_shuff = np.random.permutation(x_train.shape[0])
            # shuffle data
            x_train = x_train[ii_shuff,:]
            y_train = y_train[ii_shuff,:]
            # split data into batches
            minibatches_x = [x_train[k:k+train_batchSize] for k in range(0,x_train.shape[0],train_batchSize)]
            minibatches_y = [y_train[k:k+train_batchSize] for k in range(0,y_train.shape[0],train_batchSize)]
            # now, iterate through all the batches
            for ii_iter in range(len(minibatches_x)):
                # perform forward pass through network:
                y_hat = myModel.fprop(minibatches_x[ii_iter].T)
                # compute loss
                loss = myModel.loss.fprop(y_hat,minibatches_y[ii_iter].T)
                # propagate error backwards (compute gradients) & update weights
                myModel.bprop(y_hat,minibatches_y[ii_iter].T,train_lrate)
                # log summary  each epoch
                if ii_iter==0:
                    results_train = tif.eval.computeLossAcc(x_train,y_train.T,myModel)
                    results_test = tif.eval.computeLossAcc(x_test,y_test.T,myModel)
                    xenTrain.append(results_train[0])
                    accTrain.append(results_train[1])
                    xenTest.append(results_train[0])
                    accTest.append(results_train[1])
                    print('Epoch %d, AccTrain: %.4f, AccTest: %.4f -- XenTrain: %3.2f, XenTest: %3.2f'%(ii_epoch+1,results_train[1],results_test[1],results_train[0],results_test[0]))

    # save trained model:
    tif.io.save_model(myModel,param_modelIDX,param_modelDir,isPy3)
    # compute confusion matrix:
    cMatTrain = tif.eval.computeConfMat(x_train,y_train,myModel)
    cMatTest  = tif.eval.computeConfMat(x_test,y_test,myModel)
    # save log of training:
    log_dict = {'cross_entropy_train':xenTrain,
                'cross_entropy_test':xenTest,
                'accuracy_train':accTrain,
                'accuracy_test':accTest,
                'confmat_train':cMatTrain,
                'confmat_test':cMatTest}
    tif.io.save_log(log_dict,param_modelIDX,param_logDir,isPy3)

## EXPERIMENTS -----------------------------------------------------------------
accTrain = []
accTest  = []
xenTrain = []
xenTest  = []
logName  = ''

myModel = nnet_setup(param_modelIDX)
tif.monitor.controlModelSetup(myModel)
nnet_train(x_train,y_train,x_test,y_test,myModel,
            train_lrate=train_lrate,
            train_mode=train_mode,
            train_batchSize=train_batchSize)

## how to evaluate a saved model:
# # load model
# myModel = tif.io.load_model(modIDX,param_modelDir)
# # compute stuff:
# results_train = tif.eval.computeLossAcc(x_train,y_train.T,myModel)
# results_test = tif.eval.computeLossAcc(x_test,y_test.T,myModel)
# np.set_printoptions(threshold=np.inf)
# cMat = tif.eval.computeConfMat(x_train,y_train,myModel)
# print(cMat)
# cMat = tif.eval.computeConfMat(x_test,y_test,myModel)
# print(cMat)
