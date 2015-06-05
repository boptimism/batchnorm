import numpy as np

#-----------------------------------------------------
# activation function

def donothing(x):
    return x

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def relu(x):
    return x[x<0]=0.

def softplus(x):
    return np.log(1.+np.exp(x))

def softmax(x):
    return (np.exp(x.T)/np.sum(np.exp(x),1)).T

actFn={'pass':donothing,
       'sigmoid':sigmoid,
       'relu':relu,
       'softplus':softplus,
       'softmax':softmax}

#----------------------------------------------------
# derivatives

def ddono(x):
    return np.ones(x.shape)

def dsig(x):
    return sigmoid(x)*(1.-sigmoid(x))

def drelu(x):
    x[x>0]=1.
    x[x<0]=0.
    return x

def dsplus(x):
    return sigmoid(x)

def dsmax(x):
    return np.array([np.diag(x_s)-np.outer(x_s,x_s) for x_s in x])

dactFn={'pass':ddono,
        'sigmoid':dsig,
        'relu':drelu,
        'softplus':dsplus,
        'softmax':dsmax}

#-----------------------------------------------------
# loss function
# phi is the label, psi is the prediciton from model

def loss_ent(phi,psi):
    return -np.sum(phi*np.log(psi))/len(phi)

def loss_sqr(phi,psi):
    return np.sum((phi-psi)**2)/len(phi)

lossFn={'cross_entropy':loss_ent,'square':loss_sqr}

#-----------------------------------------------------
# loss function
# phi is the label, psi is the prediciton from model

def dlent(phi,psi):
    return -phi/psi/len(phi)

def dlsqr(phi,psi):
    return 2.*(psi-phi)/len(phi)

dlossFn={'cross_entropy':dlent,'square':dlsqr}

#-----------------------------------------------------
# prep function

def donothing(inputs):
    return inputs

def sample_center(inputs):
    return inputs-np.mean(inputs,0)

def node_center(inputs):
    return (inputs.T-np.mean(inputs,1)).T

def sample_norm(inputs):
    return (inputs-np.mean(inputs,0))/(np.std(inputs,0)+1.e-15)

def node_norm(inputs):
    tmp=inputs.T
    return ((tmp-np.mean(tmp,0))/(np.std(tmp,0)+1.e-15)).T

prepFn={'pass':donothing,
        'node_center':node_center,
        'sample_center':sample_center,
        'node_norm':node_norm,
        'sample_norm':sample_norm}

