"""
BN-V1
"""
import data_loader as dl
import network_bn_v1 as bnv1
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl

if __name__=="__main__":

    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    with open('initial_conf.pickle','rb') as f_init:
        data=pkl.load(f_init)

    training_index=data['training_index']
    testing_index=data['testing_index']
    train_input=np.array(t_img[training_index])
    train_label=np.array(t_label[training_index])
    test_input=np.array(s_img[testing_index])
    test_label=np.array(s_label[testing_index])

    layers=data['layers']
    weights=data['weights']
    bias=data['bias']

    num_of_trains=len(train_input)
    num_of_tests=len(test_input)
    
    learnrate=data['learnrate']
    batchsize=60
    epochs=50

    test_check=True
    train_check=False
    
    network=bnv1.BNv1(layers,learnrate,batchsize,epochs,weights,bias)

    network.sgd(train_input,train_label,test_input,test_label,
                test_check=test_check,train_check=train_check)

    if test_check:
        test_accu=np.array(network.test_accu)
        test_cost=np.array(network.test_cost)
        print 'accuracy:'
        print test_accu
        #--------------------------------
        xaxis=np.arange(epochs)

        fig=plt.figure(1)
        plt.suptitle('TestSet, BN-V1')
        plt.subplot(2,1,1)
        plt.plot(xaxis,test_accu,'r-o')
        plt.grid()
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')

        plt.subplot(2,1,2)
        plt.plot(xaxis,test_cost,'r-o')
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')

        plt.savefig('../results/bnv1_TestSet.png')
        plt.show()

    if train_check:
        accu_train=np.array(network.accu_train)
        cost_train=np.array(network.cost_train)
        print 'accuracy:'
        print accu_train
        #--------------------------------
        xaxis=np.arange(epochs)
        
        fig=plt.figure(2)
        plt.suptitle('TrainSets, BN-V1')
        plt.subplot(2,1,1)
        plt.plot(xaxis,accu_train,'r-o')
        plt.grid()
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')

        plt.subplot(2,1,2)
        plt.plot(xaxis,cost_train,'r-o')
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')

        plt.savefig('../results/bnv1_TrainSet.png')
        plt.show()
        
    data={"number_of_trains":num_of_trains,
          "number_of_tests":num_of_tests,
          "layers":layers,
          "learnrate":learnrate,
          "mini-batch size":batchsize,
          "epochs":epochs,
          "test_accu":test_accu,
          "test_cost":test_cost
      }
    with open("../results/bnv1_accuracy.pickle",'w') as frec:
        pkl.dump(data,frec)
