"""
Baseline
"""
import data_loader as dl
import network_baseline as bl
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import sys

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
    batchsize=data['batchsize']
    epochs=data['epochs']
    test_check=data['test_check']
    train_check=data['train_check']
    
    lrate_decay=data['lrate_decay']
    init_type=data['init']
    
    dbrec=bool(int(sys.argv[1]))

    rec_check=bool(int(sys.argv[2]))
    
    network=bl.Baseline(layers,learnrate,lrate_decay,batchsize,epochs,num_of_trains,num_of_tests,
                        weights,bias,dbrec=dbrec)

    network.sgd(train_input,train_label,test_input,test_label,init_type,
                test_check=test_check,train_check=train_check,rec_check=rec_check)

    #------------------------------------------
    data={"number_of_trains":num_of_trains,
          "number_of_tests":num_of_tests,
          "layers":layers,
          "learnrate":learnrate,
          "mini-batch size":batchsize,
          "epochs":epochs,
      }

    if test_check:
        test_accu=np.array(network.test_accu)
        test_cost=np.array(network.test_cost)
        print 'accuracy:'
        print test_accu
        data['test_accu']=test_accu
        data['test_cost']=test_cost
        
        #--------------------------------
        xaxis=np.arange(epochs)

        fig=plt.figure(1)
        plt.suptitle('TestSet,BaseLine')
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

        plt.savefig('../results/baseline_TestSet.png')
        plt.show()


    if train_check:
        train_accu=np.array(network.train_accu)
        train_cost=np.array(network.train_cost)
        print 'accuracy:'
        print train_accu
        data['train_accu']=train_accu
        data['train_cost']=train_cost
        
        #--------------------------------
        xaxis=np.arange(epochs)
        
        fig=plt.figure(2)
        plt.suptitle('TrainSets')
        plt.subplot(2,1,1)
        plt.plot(xaxis,train_accu,'r-o')
        plt.grid()
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')

        plt.subplot(2,1,2)
        plt.plot(xaxis,train_cost,'r-o')
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')

        plt.savefig('../results/baseline_TrainSet.png')
        plt.show()
        
    with open("../results/baseline_accuracy.pickle",'w') as frec:
        pkl.dump(data,frec)

