"""
Baseline
"""
import data_loader as dl
import network_baseline as bl
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import sys
import argparse


def run_baseline(params={}):

    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    with open('initial_conf.pickle','rb') as f_init:
        data=pkl.load(f_init)

    # This is where we overwrite the defaults with the passed in paramters.    
    for k,v in params.items():
        data[k]=v

    training_index=data['training_index']
    testing_index=data['testing_index']
    train_input=np.array(t_img[training_index])
    train_label=np.array(t_label[training_index])
    test_input=np.array(s_img[testing_index])
    test_label=np.array(s_label[testing_index])

    layers=data['layers']
    weights=data['weights']
    bias=data['bias']
    learnrate=data['learnrate']
    stop_at=data['stop_at']

    test_check=bool(data['test_check'])
    train_check=bool(data['train_check'])
    save_file=bool(data['save_file'])
    dbrec=data['dbrec']

    num_of_trains=len(train_input)
    num_of_tests=len(test_input)

    batchsize=60
    epochs=50

    
    network=bl.Baseline(layers,learnrate,batchsize,epochs,num_of_trains,num_of_tests,
                        weights,bias,dbrec=dbrec,stop_at=stop_at,comment=data['comment'])

    network.sgd(train_input,train_label,test_input,test_label,
                test_check=test_check,train_check=train_check)


    test_check=data['test_check']
    train_check=data['train_check']
    save_file=data['save_file']
    #------------------------------------------
    if test_check:
        test_accu=np.array(network.test_accu)
        test_cost=np.array(network.test_cost)
        print 'accuracy:'
        print test_accu
        #--------------------------------

        if plot_flag:
            xaxis=np.arange(epochs)

            fig=plt.figure(1)
            plt.suptitle('TestSet')
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
        accu_train=np.array(network.accu_train)
        cost_train=np.array(network.cost_train)
        print 'accuracy:'
        print accu_train
        #--------------------------------
        if plot_flag:
            xaxis=np.arange(epochs)
            
            fig=plt.figure(2)
            plt.suptitle('TrainSets')
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

            plt.savefig('../results/baseline_TrainSet.png')
            plt.show()
            
    if save_file:    
        data={"number_of_trains":num_of_trains,
              "number_of_tests":num_of_tests,
              "layers":layers,
              "learnrate":learnrate,
              "mini-batch size":batchsize,
              "epochs":epochs,
              "test_accu":test_accu,
              "test_cost":test_cost,
              "comment":comment
          }
        with open("../results/baseline_accuracy.pickle",'w') as frec:
            pkl.dump(data,frec)


if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Learn mNIST with baseline backprop')
    parser.add_argument('--lr', dest='learnrate', type=float, default=0.1,
                       help='The learning rate of the network. default=0.1')
    parser.add_argument('--db', dest='dbrec', type=int, default=0,
                       help='Record results to a database. default=0, set to 1 to record. Must configure ~/.dbconf to use')
    parser.add_argument('--test_check', dest='test_check', type=int, default=1,
                       help='Check test samples and make a plot. default=1.')
    parser.add_argument('--train_check', dest='train_check', type=int, default=1,
                       help='Check test samples and make a plot. default=1 (set to 0 to not check).')
    parser.add_argument('--save_file', dest='save_file', type=int, default=1,
                       help='save output to file. default=1 (set to 0 to not save).')
    parser.add_argument('--make_plots', dest='plot_flag', type=int, default=0,
                       help='plot results output to file. default=0 (set to 0 to not save).')
    parser.add_argument('--comment', dest='comment', type=str, default='',
                       help='A comment that will get saved to the DB')



    params = vars(parser.parse_args())
    run_baseline(params)