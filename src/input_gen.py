'''
Generate input files for each calculation
This example shows how to generate sequence of learning rate
'''
import data_loader as dl
import numpy as np
import cPickle as pkl
import random as rnd

def inputs_gen():

    input_data=[]
    
    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=60000
    totaltrains=len(t_img[:,0])
    idx_train=np.array(rnd.sample(range(totaltrains),numoftrains))
    
    numoftests=10000
    totaltest=len(s_img[:,0])
    idx_test=np.array(rnd.sample(range(totaltest),numoftests))

    layers=[28*28,100,100,100,10]
    ubd=np.sqrt(3.)
    lbd=-np.sqrt(3.)
    weights=[np.random.uniform(lbd,ubd,[x,y])
             for x,y in zip(layers[:-1],layers[1:])]
    bias=[np.random.uniform(lbd,ubd,x) for x in layers[1:]]
    bias.insert(0,np.zeros(layers[0]))
    gammas=[np.random.uniform(lbd,ubd,x) for x in layers[1:]]
    gammas.insert(0,np.ones(layers[0]))
    
    data={}
    data['layers']=layers
    data['weights']=weights
    data['bias']=bias
    data['training_index']=idx_train
    data['testing_index']=idx_test
    data['gammas']=gammas


    data['batchsize']=60
    data['epochs']=20

    data['test_check']=True
    data['train_check']=False
    data['init']='constVar1'
    data['lrate_decay']=0.0
    data['lrate']=0.1
 
    return pkl.dumps(data)
