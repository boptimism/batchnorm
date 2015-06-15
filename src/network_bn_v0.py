"""
BN-V0
"""
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

class BNv0:
    def __init__(self,layers,learnrate,batchsize,epochs,num_trains,num_tests,weights,bias,gammas,stop_at=1,comment='',dbrec=0):

        self.layers=layers
        self.learnrate=learnrate
        self.batchsize=batchsize
        self.epochs=epochs
        self.weights=weights
        self.bias=bias
        self.gammas=gammas
        self.stop_at=stop_at

        
        self.us=[np.zeros([batchsize,l]) for l in self.layers]
        self.xhats=[np.zeros([batchsize,l]) for l in self.layers]
        self.ys=[np.zeros([batchsize,l]) for l in self.layers]
        self.deltas=[np.zeros([batchsize,l]) for l in self.layers[1:]]
        self.stds=[np.zeros(l) for l in self.layers]
        
        # non moving average
        self.means_inf=[np.zeros(l) for l in self.layers]
        self.vars_inf=[np.zeros(l) for l in self.layers]
        
        self.test_accu=[]
        self.test_cost=[]
        self.train_accu=[]
        self.train_cost=[]
        self.dbrec=dbrec

        if self.dbrec:
            self.con=Connection()
            self.con.use('ann3')

            rec_runs={'neural_network_type':'bn_v0',
                      'layers':str(self.layers),
                      'database':'MNIST',
                      'learning_rate_i':0.0,
                      'learning_rate':self.learnrate,
                      'batch_size':self.batchsize,
                      'total_epochs':self.epochs,
                      'train_size':num_trains,
                      'test_size':num_tests,
                      'code_file':'bn_v0.py',
                      'code_version':codever.git_version(),
                      'comment':comment}
            #self.con=self.connect()
            self.con.saveToDB('runs',rec_runs)
            self.runid = self.con.lastInsertID()
        
    def sgd(self,train_inputs,train_labels,test_inputs,test_labels,
            test_check=True,train_check=False):
        
        num_of_trains=len(train_labels)
        batch_per_epoch=num_of_trains/self.batchsize
        idx_epoch=np.arange(num_of_trains)
        

        for p in np.arange(self.epochs):
            tstart=time.clock()
            np.random.shuffle(idx_epoch)
            if self.dbrec:
                rec_epochs={'num':int(p),'runid':self.runid}
                self.con.saveToDB('epochs',rec_epochs)
                epochid = self.con.lastInsertID()
                self.con.saveToDB('epochdata',{'epochid':epochid, 'permidx':idx_epoch})


            for q in np.arange(batch_per_epoch):
                ts=time.clock()
                idx_batch=idx_epoch[q*self.batchsize:(q+1)*self.batchsize]
                batch_data=train_inputs[idx_batch]
                batch_label=train_labels[idx_batch]

                self.feedforward(batch_data)
                
                dw,db,dg=self.bp(batch_label)
                
                self.batch_update(dw,db,dg)
                te=time.clock()

                if self.dbrec and ((p==0 and (random()<0.15 or q<100)) or random()<0.02) :

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
                        g_mu=[np.mean(self.gammas[i-1]) for i,x in enumerate(self.layers)]
                        g_sig=[np.std(self.gammas[i-1]) for i,x in enumerate(self.layers)]
                        
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
                                    'gamma_mu':pkl.dumps(b_mu),
                                    'gamma_sig':pkl.dumps(b_sig),
                                    'mbid':mbid}
                        # self.con.saveToDB('mbparams',rec_params)
                        # this table is not made 

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
                    
                        #self.con.saveToDB('mb_data',rec_mbdata)


            if test_check:
                bs_inf=60
                ep_inf=3
                self.pop_stats(train_inputs,ep_inf,bs_inf)
                accu,cost=self.inference(test_inputs,test_labels)
                self.test_accu.append(accu)
                self.test_cost.append(cost)
            if train_check:
                accu,cost=self.inference(train_inputs,train_labels)
                self.train_accu.append(accu)
                self.train_cost.append(cost)
                    
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
        eps=1.e-15
        self.us[0]=batch_data
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(self.us[l-1],self.weights[l-1])
            self.xhats[l]=(wu-np.mean(wu,0))/(np.std(wu,0)+eps)
            self.stds[l]=np.std(wu,0)
            self.ys[l]=self.gammas[l]*self.xhats[l]+self.bias[l]
            self.us[l]=sigmoid(self.ys[l])

    def bp(self,batch_label):
        eps=1.e-15
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_b=[np.zeros(b.shape) for b in self.bias]
        adj_g=[np.zeros(g.shape) for g in self.gammas]
        coeff=[g/(sigma+eps) for g,sigma in zip(self.gammas,self.stds)]
        
        self.deltas[-1]=(self.us[-1]-batch_label)/self.batchsize
        adj_b[-1]=np.sum(self.deltas[-1],0)
        adj_g[-1]=np.sum(self.xhats[-1]*self.deltas[-1],0)
        u_shift=self.us[-2]-np.mean(self.us[-2],0)
        t3=np.dot(u_shift.T,self.deltas[-1])
        t1=np.mean(self.deltas[-1]*self.xhats[-1],0)
        t2=np.dot(self.us[-2].T,self.xhats[-1])
        adj_w[-1]=(t3-t2*t1)*coeff[-1]
        
        for l in np.arange(2,len(self.layers)):
            delta_shift=(self.deltas[-l+1]-np.mean(self.deltas[-l+1],0)-\
                         self.xhats[-l+1]*\
                         np.mean(self.deltas[-l+1]*self.xhats[-l+1],0))*coeff[-l+1]
            self.deltas[-l]=sigmoid(self.ys[-l])*(1.-sigmoid(self.ys[-l]))*\
                        np.dot(delta_shift,self.weights[-l+1].T)
            adj_b[-l]=np.sum(self.deltas[-l])
            adj_g[-l]=np.sum(self.deltas[-l]*self.xhats[-l],0)
            u_shift=self.us[-l-1]-np.mean(self.us[-l-1],0)
            t3=np.dot(u_shift.T,self.deltas[-l])
            t1=np.mean(self.deltas[-l]*self.xhats[-l],0)
            t2=np.dot(self.us[-l-1].T,self.xhats[-l])
            adj_w[-l]=(t3-t2*t1)*coeff[-l]

        return adj_w,adj_b,adj_g
            
    def batch_update(self,dw,db,dg):
        self.weights=[w-self.learnrate*dnw for w,dnw in zip(self.weights,dw)]
        self.bias=[b-self.learnrate*dnb for b,dnb in zip(self.bias,db)]
        self.gammas=[g-self.learnrate*dng for g,dng in zip(self.gammas,dg)]
    
    def pop_stats(self,train_inputs,ep_inf,bs_inf):
        eps=1.e-15
        num_of_trains=len(train_inputs)
        bs_per_ep_inf=num_of_trains/bs_inf
        num_of_batches=ep_inf*bs_per_ep_inf
        idx_inf=np.arange(num_of_trains)

        self.means_inf=[np.zeros(l) for l in self.layers]
        self.vars_inf=[np.zeros(l) for l in self.layers]
        
        for p in np.arange(ep_inf):
            np.random.shuffle(idx_inf)
            for q in np.arange(bs_per_ep_inf):
                idx_batch=idx_inf[q*bs_inf:(q+1)*bs_inf]
                batch_data=train_inputs[idx_batch]
                us_inf=batch_data
                for l in np.arange(1,len(self.layers)):
                    wu=np.dot(us_inf,self.weights[l-1])
                    wumean=np.mean(wu,0)
                    wuvar=np.var(wu,0)
                    self.means_inf[l]=self.means_inf[l]+wumean/num_of_batches
                    self.vars_inf[l]=self.vars_inf[l]+wuvar/(num_of_batches-1)
                    xhats_inf=(wu-wumean)/(np.sqrt(wuvar)+eps)
                    ys_inf=self.gammas[l]*xhats_inf+self.bias[l]
                    us_inf=sigmoid(ys_inf)
        
    def inference(self,test_inputs,test_labels):
        eps=1.e-15
        # sample by sample
        # labels=np.array([np.argmax(x) for x in test_labels])
        # labels_inf=[]
        # for sample in test_inputs:
        #     us_inf=sample
        #     for l in np.arange(1,len(self.layers)):
        #         wu=np.dot(us_inf,self.weights[l-1])
        #         xhats_inf=(wu-self.means_inf[l])/(np.sqrt(self.vars_inf[l])+eps)
        #         ys_inf=self.gammas[l]*xhats_inf+self.bias[l]
        #         us_inf=sigmoid(ys_inf)
        #     labels_inf.append(us_inf)
        # labels_inf=np.array(labels_inf)
        # cost=costFn(labels_inf,test_labels)
        # labels_inf=[np.argmax(x) for x in labels_inf]
        # hits=sum(labels_inf==labels)*1./len(test_inputs)

        # whole batch
        us_inf=test_inputs
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(us_inf,self.weights[l-1])
            xhats_inf=(wu-self.means_inf[l])/(np.sqrt(self.vars_inf[l])+eps)
            ys_inf=self.gammas[l]*xhats_inf+self.bias[l]
            us_inf=sigmoid(ys_inf)
        cost=costFn(us_inf,test_labels)
        label_inf=np.array([np.argmax(s) for s in us_inf])
        label_giv=np.array([np.argmax(s) for s in test_labels])
        hits=sum(label_inf==label_giv)*1./len(test_inputs)
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


