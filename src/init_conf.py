'''
Generate random
0)indices for training and testing data
1)weights, 
2)bias, 
3)gammas (for BN only)
and initial network topology. 
Used for algorithm comparison.
'''
import data_loader as dl
import numpy as np
import cPickle as pkl
import random as rnd

if __name__=="__main__":

    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=60000
    totaltrains=len(t_img[:,0])
    idx_train=np.array(rnd.sample(range(totaltrains),numoftrains))
    
    numoftests=10000
    totaltest=len(s_img[:,0])
    idx_test=np.array(rnd.sample(range(totaltest),numoftests))

    layers=[28*28,100,100,100,10]
    weights=[np.random.randn(x,y)
            for x,y in zip(layers[:-1],layers[1:])]
    bias=[np.random.randn(x) for x in layers[1:]]
    bias.insert(0,np.zeros(layers[0]))
    gammas=[np.random.randn(x) for x in layers[1:]]
    gammas.insert(0,np.ones(layers[0]))
    learnrate=0.1

    # weights=[np.random.uniform(-1.,1.,[x,y])
    #         for x,y in zip(layers[:-1],layers[1:])]
    # bias=[np.random.uniform(-1.,1.,x) for x in layers[1:]]
    # bias.insert(0,np.zeros(layers[0]))
    # gammas=[np.random.uniform(-1.,1.,x) for x in layers[1:]]
    # gammas.insert(0,np.ones(layers[0]))
    
    data={}
    data['layers']=layers
    data['weights']=weights
    data['bias']=bias
    data['training_index']=idx_train
    data['testing_index']=idx_test
    data['gammas']=gammas
    data['learnrate']=learnrate
    data['stop_at']=1  # once accuracy reaches this level stop the run.
    data['train_check']=True
    data['test_check']=True
    data['save_file']=True
    data['dbrec']=1

    with open('initial_conf.pickle','wb') as fout:
        pkl.dump(data,fout,protocol=-1)
    
