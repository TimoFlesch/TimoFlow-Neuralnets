"""
A little neural network toolbox

(c) Timo Flesch, 2017

"""
import numpy as np
import random
from scipy.stats import truncnorm

class myNet(object):
    """
    neural network
    """
    def __init__(self,netParams):
        self.layers = netParams['layers']
        self.loss   = netParams['loss']

    def fprop(self,x_in):
        for layer in self.layers:
            x_in = layer.fprop(x_in)
        return x_in

    def bprop(self,y_hat,y_true,lRate):
        dY = []
        dY = self.loss.bprop(y_hat,y_true)
        for ii in range(len(self.layers)-1,-1,-1):
            # compute bprop and param_grad
            # update weights accordingly
            dX = self.layers[ii].bprop(dY)
            if self.layers[ii].type=='act_linear':
                self.layers[ii].W-lRate*self.layers[ii].param_W_grad(dY)
                self.layers[ii].b-lRate*self.layers[ii].param_b_grad(dY)
            dY = dX


## layer modules:
class module_linear(object):
    """
    linear module
    """
    def __init__(self,dimIn,dimOut):
        self.type = 'act_linear'
        self.W    = truncnorm.rvs(0,0.1,size=(dimOut,dimIn))#np.random.randn(dimOut,dimIn)
        self.b    = np.ones([dimOut,1])*0.1
        self.out  = np.zeros([dimOut,1])
        self.x    = np.zeros([dimIn,1])
    def fprop(self,x):
        self.x  = x
        self.out = np.dot(self.W,x)+self.b
        return self.out

    def bprop(self,dY):
        return np.dot(self.W.T,dY)

    def param_W_grad(self,dY):
        return np.dot(dY,self.x.T)

    def param_b_grad(self,dY):
        return np.sum(dY, axis=1,keepdims=True)

class module_relu(object):
    """
    Rectified Linear Unit module
    """
    def __init__(self,dimOut):
        self.type = 'act_relu'
        self.out = np.zeros([dimOut,1])

    def fprop(self,x):
        self.out = x * (x>0).astype('int')
        return self.out

    def bprop(self,dY):
        return dY * (dY>0).astype('int')

class module_sigmoid(object):
    """
    Sigmoid activation module
    """
    def __init__(self,dimOut):
        self.type = "act_sigm"
        self.out = np.zeros([dimOut,1])

    def fprop(self,x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def bprop(self,dY):
        return dY *(1-self.out)*self.out

class module_tanh(object):
    """
    hyperbolic tangent activation module
    """

    def __init__(self,dimOut):
        self.type = "act_tanh"
        self.out = np.zeros([dimOut,1])

    def fprop(self,x):
        self.out = np.tanh(x)
        return self.out

    def bprop(dY):
        return dY*(1-self.out*self.out)


class module_softmax(object):
    """
    softmax module
    """

    def __init__(self,dimOut):
        self.type = 'act_softmax'
        self.out = np.zeros([dimOut,1])

    def fprop(self,x):
        stable_expx = np.exp(x-np.max(x, axis=0, keepdims=True))
        self.out = stable_expx/np.sum(stable_expx, axis=0, keepdims=True)
        return self.out

    def bprop(self,dY):
        return dY*self.out-np.sum(dY*self.out,axis=0,keepdims=True)

## loss modules
class module_sqeucl(object):
    """
    squared euclidean module
    """

    def __init__(self,dimOut):
        self.type = 'loss_squaredeuclidean'
        self.out = np.zeros([dimOut,1])

    def fprop(self,x,y_true):
        self.out = np.mean(-(np.sum((x-y_true)**2,axis=0)))
        return self.out

    def bprop(self,x,y_true):
        return (x-y_true)*2


class module_xent(object):
    """
    cross entropy module
    """

    def __init__(self,dimOut):
        self.type = 'loss_crossentropy'
        self.out = np.zeros([dimOut,1])

    def fprop(self,x,y_true):
        self.out = np.mean(-np.sum(y_true*np.log(x+1e-10), axis=0, keepdims=True))
        return self.out

    def bprop(self,x,y_true):
        return x/y_true


class module_xentbinary(object):
    """
    cross entropy for sigmoid  (llh)
    """

    def __init__(self,dimOut):
        self.type = 'loss_crossentropy_binary'
        self.out = np.zeros([dimOut,1])

    def fprop(self,x,y_true):
        self.out = np.mean(np.sum(-(y_true*np.log(x+1e-10)+(1-y_true)*np.log(1-x+1e-10)), axis=0, keepdims=True))
        return self.out

    def bprop(self,x,y_true):
        return ((1-y_true)/(1-x))-(y_true/x)



class module_xentlogit(object):
    """
    cross entropy module with logit input
    (softmax -xent compound module)
    """

    def __init__(self,dimOut):
        self.type = 'loss_crossentropy_logit'
        self.out = np.zeros([dimOut,1])

    def fprop(self,x,y_true):
        stable_expx = np.exp(x-np.max(x, axis=0, keepdims=True))
        logits = stable_expx/np.sum(stable_expx, axis=0, keepdims=True)
        self.out = np.mean(-np.sum(y_true*np.log(logits+1e-10), axis=0, keepdims=True))
        return self.out

    def bprop(self,x,y_true):
        return x-y_true
