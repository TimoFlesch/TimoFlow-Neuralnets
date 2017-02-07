"""
A little neural network toolbox

(c) Timo Flesch, 2017

"""
import numpy
def controlModelSetup(model):
    """
    receives model as input and prints its structure
    """
    print("network architecture:".upper())
    for ii in model.layers:
        print('%s'%ii.type)
        if ii.type=='act_linear':
            print(' Weights: %s \n Bias: %s'% (str(ii.W.shape),str(ii.b.shape)))
        print(' Output: %s'% (str(ii.out.shape)))
    print('\nLoss Function:'.upper())
    print('%s \n Output: %s '%(model.loss.type,str(model.loss.out.shape)))
