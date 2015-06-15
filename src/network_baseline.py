import numpy as np
import time
import scipy.stats as sstats
import cPickle as pkl
import get_code_ver as codever
import multiprocessing
import sys, os
from os.path import expanduser, sep
sys.path.append(expanduser("~") + sep + "modules")
from helpers.DBUtilsClass import Connection
from random import random


class Baseline:
    def __init__(self,layers,learnrate,batchsize,epochs,num_trains,num_tests,
                 weights,bias,stop_at=1,comment='',dbrec=0):
        
        self.layers=layers
        self.learnrate=learnrate
        self.batchsize=batchsize
        self.epochs=epochs
        self.weights=weights
        self.bias=bias
        self.stop_at=stop_at
        
        self.us=[np.zeros([batchsize,l]) for l in self.layers]
        self.ys=[np.zeros([batchsize,l]) for l in self.layers]
        self.deltas=[np.zeros([batchsize,l]) for l in self.layers[1:]]
        
        self.test_accu=[]
        self.test_cost=[]
        self.train_accu=[]
        self.train_cost=[]
        self.dbrec=dbrec

        if self.dbrec:
            self.con=Connection()
            self.con.use('ann3')

            rec_runs={'neural_network_type':'baseline',
                      'layers':str(self.layers),
                      'database':'MNIST',
                      'learning_rate_i':0.0,
                      'learning_rate':self.learnrate,
                      'batch_size':self.batchsize,
                      'total_epochs':self.epochs,
                      'train_size':num_trains,
                      'test_size':num_tests,
                      'code_file':'baseline.py',
                      'code_version':codever.git_version(),
                      'comment':comment}
            self.con.saveToDB('runs',rec_runs)
            self.runid = self.con.lastInsertID()
            
    def sgd(self,train_inputs,train_labels,test_inputs,test_labels,
            test_check=True,train_check=False):

        num_of_trains=len(train_labels)
        batch_per_epoch=num_of_trains/self.batchsize
        idx_epoch=np.arange(num_of_trains)
        
        for p in np.arange(self.epochs):
            # For each epoch
            tstart=time.clock()
            np.random.shuffle(idx_epoch)
            # SQL--------------------------------------
            if self.dbrec:
                rec_epochs={'num':int(p),'runid':self.runid}
                self.con.saveToDB('epochs',rec_epochs)
                epochid = self.con.lastInsertID()
                self.con.saveToDB('epochdata',{'epochid':epochid, 'permidx':idx_epoch})

            for q in np.arange(batch_per_epoch):
                # For each batch

                ts=time.clock()
                
                idx_batch=idx_epoch[q*self.batchsize:(q+1)*self.batchsize]
                batch_data=train_inputs[idx_batch]
                batch_label=train_labels[idx_batch]

                self.feedforward(batch_data)

                dw,db=self.bp(batch_label)
                
                self.batch_update(dw,db)

                te=time.clock()

                if self.dbrec and ((p==0 and (random()<0.15 or q<100)) or random()<0.02):

                    accu,cost=self.inference(test_inputs,test_labels)

                    rec_batches={'num':int(q),
                                 'epochid':epochid,
                                 'runid':self.runid,
                                 'runtime':te-ts,
                                 'acc': float(accu),
                                 'cost': float(cost)}
                    self.con.saveToDB('minibatches',rec_batches)
                    mbid = self.con.lastInsertID()


                   
                    if self.dbrec>1:
                    
                        w_mu=[np.mean(self.weights[i-1],0) for i,x in enumerate(self.layers)]
                        w_sig=[np.std(self.weights[i-1],0) for i,x in enumerate(self.layers)]
                        b_mu=[np.mean(self.bias[i-1]) for i,x in enumerate(self.layers)]
                        b_sig=[np.std(self.bias[i-1]) for i,x in enumerate(self.layers)]
                        err_mu=[np.mean(delta_s,0) for delta_s in self.deltas]
                        err_sig=[np.std(delta_s,0) for delta_s in self.deltas]
                        err_skew=[sstats.skew(delta_s,0) for delta_s in self.deltas]
                        err_kurtosis=[sstats.kurtosis(delta_s,0) for delta_s in self.deltas]
                        act_mu=[np.mean(u_s,0) for u_s in self.us]
                        act_sig=[np.std(u_s,0) for u_s in self.us]
                        act_skew=[sstats.skew(u_s,0) for u_s in self.us]
                        act_kurtosis=[sstats.kurtosis(u_s,0) for u_s in self.us]
                
                        rec_params={'W_mu':pkl.dumps(w_mu),
                                    'W_sig':pkl.dumps(w_sig),
                                    'bias_mu':pkl.dumps(b_mu),
                                    'bias_sig':pkl.dumps(b_sig),
                                    'mbid':mbid}
                        # self.con.saveToDB('mbparams',rec_params)
                        # This tables doesn't exist.

                        rec_samples={'error_mu':pkl.dumps(err_mu),
                                     'error_sig':pkl.dumps(err_sig),
                                     'error_skew':pkl.dumps(err_skew),
                                     'error_kurtosis':pkl.dumps(err_kurtosis),
                                     'activation_mu':pkl.dumps(act_mu),
                                     'activation_sig':pkl.dumps(act_sig),
                                     'activation_skew':pkl.dumps(act_skew),
                                     'activation_kurtosis':pkl.dumps(act_kurtosis),
                                     'mbid':mbid}
                    
                        self.con.saveToDB('mbsamples',rec_samples)

                        rec_mbdata={'mbid':mbid,
                                    'W':pkl.dumps(self.weights),
                                    'bias':pkl.dumps(self.bias),
                                    'error':pkl.dumps(self.deltas),
                                    'activation':pkl.dumps(self.us)}
                    
                      #  self.con.saveToDB('mb_data',rec_mbdata)

                
            if test_check:
                accu,cost=self.inference(test_inputs,test_labels)
                self.test_accu.append(accu)
                self.test_cost.append(cost)
            else:
                self.test_accu.append(-1.0)
                self.test_cost.append(-1.0)
            if train_check:
                accu,cost=self.inference(train_inputs,train_labels)
                self.train_accu.append(accu)
                self.train_cost.append(cost)
            else:
                self.train_accu.append(-1.0)
                self.train_cost.append(-1.0)



            tend=time.clock()                

            if self.dbrec:
                sqlstr = 'update_epoch_accuracy({0},{1},{2},{3},{4},{5})'           
                self.con.call(sqlstr.format(epochid, self.train_accu[-1],self.train_cost[-1],self.test_accu[-1], self.test_cost[-1],tend-tstart))

            print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)
            try: 
                if self.test_accu[-1]>self.stop_at:
                    return
            except:
                if p==0:
                    print "Need test_check enabled to use stop_at"

    def feedforward(self,batch_data):
        self.us[0]=batch_data
        for l in np.arange(1,len(self.layers)):
            self.ys[l]=np.dot(self.us[l-1],self.weights[l-1])+self.bias[l]
            self.us[l]=sigmoid(self.ys[l])

    def bp(self,batch_label):
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_b=[np.zeros(b.shape) for b in self.bias]
        
        self.deltas[-1]=(self.us[-1]-batch_label)/self.batchsize
        adj_b[-1]=np.sum(self.deltas[-1],0)
        adj_w[-1]=np.dot(self.us[-2].T,self.deltas[-1])

        for l in np.arange(2,len(self.layers)):
            self.deltas[-l]=sigmoid(self.ys[-l])*(1.-sigmoid(self.ys[-l]))*\
                        np.dot(self.deltas[-l+1],self.weights[-l+1].T)
            adj_b[-l]=np.sum(self.deltas[-l])
            adj_w[-l]=np.dot(self.us[-l-1].T,self.deltas[-l])

        return adj_w,adj_b
            
    def batch_update(self,dw,db):
        self.weights=[w-self.learnrate*dnw for w,dnw in zip(self.weights,dw)]
        self.bias=[b-self.learnrate*dnb for b,dnb in zip(self.bias,db)]
            

    def inference(self,test_inputs,test_labels):
        labels=np.array([np.argmax(x) for x in test_labels])
        labels_inf=[]
        for sample in test_inputs:
            us_inf=sample
            for l in np.arange(1,len(self.layers)):
                ys_inf=np.dot(us_inf,self.weights[l-1])+self.bias[l]
                us_inf=sigmoid(ys_inf)
            labels_inf.append(us_inf)
        # us_inf=[np.zeros()]
        # res_state=np.array([self.feedforward(x)[-1] for x in test_input])
        # res_nn=np.array([np.argmax(self.feedforward(x)[-1]) for x in test_input])
        # #res_nn=np.array([np.argmax(x) for x in res_state])
        # label=np.array([np.argmax(x) for x in test_label])
        labels_inf=np.array(labels_inf)
        cost=costFn(labels_inf,test_labels)
        labels_inf=[np.argmax(x) for x in labels_inf]
        hits=sum(labels_inf==labels)*1./len(test_inputs)
        if cost>1e30:
            cost=1e30

        return hits,cost
        
def costFn(labels_inf,labels):
    p=np.array([x/np.sum(x) for x in labels_inf])
    num_tests=len(labels)
    return np.sum(-labels*np.nan_to_num(np.log(p))-
                  (1.-labels)*np.nan_to_num(np.log(1.-p)))/num_tests

def sigmoid(x):
    return 1./(1.+np.exp(-np.clip(x,-100,100)))


