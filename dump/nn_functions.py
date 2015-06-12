"""
Frequently used Activation, Regularization, and Cost Functions
and their derivatives in NN training/testing.
"""
import numpy as np

#-----------------------------------------------------------------
# Activations
#
# Sigmoid.
# Input: 1-d array x
# Output: 1-d array y
# Since amplitude tends to build up in deep nets, a clipper is added
# to ensure numerical stability.

def sigmoid(x,beta=1.):
    lb=-100.
    ub=-lb
    return 1./(1.+np.exp(-beta*np.clip(x,lb,ub)))

# ReLU.
# Input: 1-d array x
# Output: 1-d array y

def relu(x,scale=1.):
    return np.maximum(0.,scale*x)

# Softmax.
# Input: 1-d array x.
# Output: 1-d array y

def softmax(x,beta=1.):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

# Tanh.
# Input: 1-d array x
# Output: 1-d array y
# Note: This function return negative values.

def tanh(x,scale=1.,amp=1.):
    return amp*np.tanh(scale*x)

aFn={"sigmoid":sigmoid,"relu":relu,"softmax":softmax,"tanh":tanh}

#----------------------------------------------------------------
# Derivatives of Activation Func
#

def dsigmoid(x,beta=1):
    return beta*sigmoid(x,beta)*(1.-sigmoid(x,beta))

def drelu(x,scale=1.):
    return (x>0).astype(float)*scale

# the derivative of softmax returns a symmetric matrix.
def dsoftmax(x,beta=1.):
    return beta*(np.diag(softmax(x,beta))-
                 np.outer(softmax(x,beta),softmax(x,beta)))

def dtanh(x,scale=1.,amp=1.):
    return scale*amp*(1.-np.power(tanh(x,scale,amp),2.))

daFn={"sigmoid":dsigmoid,"relu":drelu,"softmax":dsoftmax,"tanh":dtanh}

#---------------------------------------------------------------
# Cost Functions
#
# a is the learned prediction, t is label.
# a has dimension N_sample X N_feature.
# t has the same dimension as a.

def crossEntropy(a,t):
    a=np.array(a)
    t=np.array(t)
    num_samples=len(a[:,0])
    p=np.array([x/np.sum(x) for x in a]) # normalize output to [0,1]
    return np.sum(-t*np.nan_to_num(np.log(p))-
                  (1-t)*np.nan_to_num(np.log(1.-p)))/num_samples

def leastSquare(a,t):
    a=np.array(a)
    t=np.array(t)
    num_samples=len(a[:,0])
    return 0.5*np.sum((a-t)**2)/num_samples

cFn={'crossEntropy':crossEntropy,'leastSquare':leastSquare}

#---------------------------------------------------------------
# Growth Rate Function at the last layer.
# This is crucial for computing backpropation.
# It composes of 2 derivatives: 1 from Loss func, 1 from Activation Func.
# Unlike the def of cost functions, these derivitives only return
# growth rate for 1 sample.
#
# Input: 1-d array a and t
# Output: 1-d array delta_L

def sigEnt(a,y,t,batch_size,beta=1.):
    return beta*(a-t)/batch_size

def reluEnt(a,y,t,batch_size):
    return drelu(y,scale=1.)*(a-t)*np.nan_to_num(1./(a*(1.-a)))/batch_size

def softEnt(a,y,t,batch_size):
    return np.dot((a-t)*np.nan_to_num(1./(a*(1.-a))),
                  dsoftmax(y,beta=1.))/batch_size

def tanhEnt(a,y,t,batch_size):
    return (a-t)*np.nan_to_num(1./(a*(1.-a)))*dtanh(y,amp=1.,scale=1.)/batch_size

def sigSquare(a,y,t,batch_size):
    return (a-t)*dsigmoid(y,beta=1)/batch_size

def reluSquare(a,y,t,batch_size):
    return (a-t)*drelu(y,scale=1.)/batch_size

def softSquare(a,y,t,batch_size):
    return np.dot(a-t,dsoftmax(y,beta=1.))/batch_size

def tanhSquare(a,y,t,batch_size):
    return (a-t)*dtanh(y,amp=1.,scale=1.)/batch_size

grFn={('sigmoid','crossEntropy'):sigEnt,
      ('relu','crossEntropy'):reluEnt,
      ('softmax','crossEntropy'):softEnt,
      ('tanh','crossEntropy'):tanhEnt,
      ('sigmoid','leastSquare'):sigSquare,
      ('relu','leastSquare'):reluSquare,
      ('softmax','leastSquare'):softSquare,
      ('tanh','leastSquare'):tanhSquare}

#---------------------------------------------------------------
# Regularization function
# L1 reg:
# sum |w|
#
# L2 reg:
# sum w^2*0.5
#
# Lnorm reg:
# sum w^2/(1+w^2)
#
# Input: all weights
# Output: scalar - the penalty

def regL1(w,lmbd,num_samples):
    return np.sum([np.sum(np.abs(x)) for x in w])*lmbd/num_samples

def regL2(w,lmbd,num_samples):
    return np.sum([np.sum(np.power(x,2)) for x in w])*lmbd/num_samples*0.5

def regLnorm(w,lmbd,num_samples):
    return 0.5*np.sum([np.sum(x**2/(1.+x**2)) for x in w])*lmbd/num_sample

regFn={'L1':regL1,'L2':regL2,'Lnorm':regLnorm}

#---------------------------------------------------------------
# Derivitives of regularization

# Input: all weights
# Output: derivitives of regFn w.r.t weights
def dregL1(w,lmbd,num_samples):
    return [2.*((x>0.).astype(float)-0.5)/num_samples for x in w]

def dregL2(w,lmbd,num_samples):
    return [lmbd*x/num_samples for x in w]

def dregLnorm(w,lmbd,num_samples):
    return [lmbd*(x/(1+x**2)**2)/num_samples for x in w]

dregFn={"L1":dregL1,"L2":dregL2,"Lnorm":dregLnorm}
