# Timoflow
I wrote a little modularized neural network toolbox that allows the user to easily implement arbitrary feed-forward MLPs.

## Activation Functions 
The following activation functions are available: 
- relu 
- Sigmoid 
- Softmax 
- TanH

## Loss Functions 
Furthermore, the user can choose among several loss functions:

- Binary Cross-Entropy
- Multiclass Cross-Entropy
- Logit-Cross-Entropy Compound
- L2 Loss


## Optimisation Procedure
The networks are optimised via stochastic gradient descent (minibatch or online)

## Example
An example on how to use the toolbox is provided in example.py.
The neural network is defined as class, and the network architecture is passed as Python dictionary:
```Python
myModel = tif.nnet.myNet({'layers':[tif.nnet.module_linear(256,128),
                                        tif.nnet.module_relu(128),
                                        tif.nnet.module_linear(128,64),
                                        tif.nnet.module_relu(64),
                                        tif.nnet.module_linear(64,3),
                                        tif.nnet.module_softmax(3)],
                            'loss':tif.nnet.module_xent(3)})
```

Once the network is initialised, a forward pass can be carried out to obtain predictions for a given input:
```Python
y_hat = myModel.fprop(x_in)                
```
To compute the loss (e.g. for plotting, call fprop function of the loss module:
```Python
loss = myModel.loss.fprop(y_hat,y_true)
``` 
Finally, to update the weights, perform backpropagation (which computes the loss internally):
```Python
myModel.bprop(y_hat,y_true,lrate)
``` 


<!--
## timoflow/nnet.py
My neural network module
### class myNet
My neural networks are defined as class with "layers" and "loss" variables.
#### layers
This is a list containing layer instances. The actual neural network
#### loss
an instance of a loss function.
#### fprop
function to perform forward propagation of an input through the entire network
#### brop
My bprop function is a compound of backpropagation and weight update. The error is propagated backwards and the weights of each layer are adapted accordingly, online.  

### layer_modules
Here I defined the different modules. The most common nonlinearities were implemented
Each module has the following functions:
#### fprop
propagates the input of a module forward. Basically just feeding it through the function.
#### bprop
backpropagation of error dY
#### param_W_grad, param_b_grad
Only for linear activation function. Contains the derivative after the parameters, necessary for parameter update.

### loss_modules
I implemented cross entropy loss for softmax and sigmoid activation functions, a compound module of softmax+crossentropy and a simple squared euclidean loss.

## timoflow/monitor.py
Here, I defined a little helper function which prints the structure of a defined network (activation function + dimensionalities) on stdout. Just a sanity check :)

## timoflow/io.py
I wrote little helper functions to save and load a model, and to save the log-files of a training session.

## timoflow/eval.py
Helper function to evaluate the performance of a model and a wrapper to compute a confusion matrix (uses external funct.)

## timoflow/plot.py
Some functions to plot error curves and confusion matrices-->
