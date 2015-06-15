"""
load data
"""

import numpy as np
import gzip
import struct
import sys, os
from os.path import expanduser, sep
#MNIST_PATH = expanduser("~") + sep + "modules" + sep + "MNIST" + sep
MNIST_PATH = expanduser("~") + sep + "Projects" + sep + "BatchNorm" + sep + "MNIST" + sep


"""
Read in the training and test sets.
Training sets are splited into Training+Validation sets.
All the images' pixel value range from 0-255. This may give trouble to 
training of NN. Instead, these values are normalized to 1.
"""

def training_load():
    with gzip.open(MNIST_PATH + 'train-images-idx3-ubyte.gz','rb') as fp:
        header=fp.read(16)
        mgn,nimg,nrow,ncol=struct.unpack('>llll',header)
        data=np.fromstring(fp.read(),dtype='uint8')
    fp.close()
    train_img=np.reshape(data.astype(float)/255,(nimg,ncol*nrow))
    with gzip.open(MNIST_PATH + 'train-labels-idx1-ubyte.gz','rb') as fp:
        header=fp.read(8)
        mgn,nimg=struct.unpack('>ll',header)
        data=np.fromstring(fp.read(),dtype='uint8')
    fp.close()
    train_label=np.zeros((nimg,10))
    for i,j in zip(np.arange(nimg),data):
        train_label[i,j]=1.0
    return (train_img,train_label)

def test_load():
    with gzip.open(MNIST_PATH + 't10k-images-idx3-ubyte.gz','rb') as fp:
        header=fp.read(16)
        mgn,nimg,nrow,ncol=struct.unpack('>llll',header)
        data=np.fromstring(fp.read(),dtype='uint8')
    fp.close()
    test_img=np.reshape(data.astype(float)/255,(nimg,ncol*nrow))
    with gzip.open(MNIST_PATH + 't10k-labels-idx1-ubyte.gz','rb') as fp:
        header=fp.read(8)
        mgn,nimg=struct.unpack('>ll',header)
        data=np.fromstring(fp.read(),dtype='uint8')
    fp.close()
    test_label=np.zeros((nimg,10))
    for i,j in zip(np.arange(nimg),data):
        test_label[i,j]=1.0
    return (test_img,test_label)

