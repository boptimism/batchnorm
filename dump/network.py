"""
Vanilla NN. 
CostFn: CrossEntropy
ActivationFn: Sigmoid
"""

import numpy as np
import time
import cPickle as pkl
import json
import copy

class Vnn:
    def __init__(self,layers,learnrate,minibatch,epoch):
        self.layers=layers
        self.learnrate=learnrate
        self.minibatch=minibatch
        self.epoch=epoch
        # self.weights=[np.random.randn(x,y)/np.sqrt(x+1)
        #               for x,y in zip(self.layers[:-1],self.layers[1:])]
        with open('weights.json','r') as fw:
            data=json.load(fw)
        self.weights=[np.array(x) for x in data['weights']]
        
        self.bias=[np.random.randn(x) for x in layers[1:]]
        self.accuracy=[]
        self.cost=[]
        
        self.avgu=[np.zeros(l) for l in self.layers]
        self.avgdelta=copy.deepcopy(self.avgu)
        self.recavgu=[]
        self.recavgdelta=[]
        
    def sgd(self,inputs,outputs,tests,labels,check=False,check_freq=1):
        num_minibatch,res=divmod(len(inputs[:,0]),self.minibatch)
        if res!=0:
            print "Mini-batch can't be divided by total number of tasks."
            raise SystemExit
        inouts=zip(inputs,outputs)

        # data={'epoch':self.epoch,'batch_per_epoch':num_minibatch,
        #       'check_freq':check_freq}
        # wdata={'epoch':self.epoch,'batch_per_epoch':num_minibatch,
        #       'check_freq':check_freq}
        
        for p in np.arange(self.epoch):
            tstart=time.clock()
            np.random.shuffle(inouts)
            rand_trdata=np.array(zip(*inouts)[0])
            rand_trlabel=np.array(zip(*inouts)[1])
            for q in np.arange(num_minibatch):
                batch_data=rand_trdata[q*self.minibatch:(q+1)*self.minibatch]
                batch_label=rand_trlabel[q*self.minibatch:(q+1)*self.minibatch]
                # Whitening only the inputs to VNN
                #batch_data=(batch_data-np.mean(batch_data,0))/(np.std(batch_data,0)+1.e-15)
                # self.avgu=[np.zeros(l) for l in self.layers]
                # self.avgdelta=[np.zeros(l) for l in self.layers[1:]]
                
                # self.batch_update(batch_data,batch_label)

                # self.recavgu.append(self.avgu)
                # self.recavgdelta.append(self.avgdelta)
                
                num_of_batches=p*num_minibatch+q+1
                if check and not(num_of_batches%check_freq):
                    accu,cost=self.pred(tests,labels)
                    self.accuracy.append(accu)
                    self.cost.append(cost)
                # Check the distribution
                # 1) between adjcent layers
                # 2) between Normal Distri. and a random node in hidden layer.
                # if check and not(num_of_batches%check_freq) and len(self.layers)>=4:
                #     fwd_test=[self.feedforward(x) for x in tests]
                #     hidden_last_in=[h[-3] for h in fwd_test]
                #     hidden_2ndlast_in=[h[-4] for h in fwd_test]
                #     hidden_z_last_in=np.array([np.dot(self.weights[-2].T,x)+self.bias[-2]
                #                                for x in hidden_last_in])
                #     hidden_z_2ndlast_in=np.array([np.dot(self.weights[-3].T,x)+self.bias[-3]
                #                                   for x in hidden_2ndlast_in])
                #     key='p'+str(p)+'q'+str(q)+'hl-1'
                #     data[key]=hidden_z_last_in
                #     #wdata[key]=copy.deepcopy(self.weights[-1])
                #     key='p'+str(p)+'q'+str(q)+'hl-2'
                #     #data[key]=hidden_z_2ndlast_in
                #     key='p'+str(p)+'q'+str(q)
                #     wdata[key]=copy.deepcopy(self.weights)
                    
            tend=time.clock()
            
            print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)

        # with open('vanilla-dist.pickle','wb') as fout_pickle:
        #     pkl.dump(data,fout_pickle,protocol=-1)

        # with open('baseline_weights.pickle','wb') as fout:
        #      pkl.dump(wdata,fout,protocol=-1)
            
        # with open('beerbet.pickle','wb') as fout1:
        #     pkl.dump([self.recavgu,self.recavgdelta],fout1,protocol=-1)
             
    def batch_update(self,inputs,outputs):
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.bias]

        for x,y in zip(inputs,outputs):
            dnw,dnb=self.bp(x,y)

            for i in xrange(len(nabla_w)):
                nabla_w[i]+=dnw[i]
                nabla_b[i]+=dnb[i]
        
        for i in xrange(len(self.weights)):
            self.weights[i]-=self.learnrate*nabla_w[i]#/self.minibatch
            self.bias[i]-=self.learnrate*nabla_b[i]#/self.minibatch
            
    def bp(self,sinput,slabel):
        adata=self.feedforward(sinput)

        self.avgu=[a+b for a,b in zip(adata,self.avgu)]
        
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_b=[np.zeros(b.shape) for b in self.bias]
        # adj_b[-1]=-(slabel-adata[-1])*\
        #     dsig(np.dot(adata[-2],self.weights[-1])+self.bias[-1])
        adj_b[-1]=-(slabel-adata[-1])/self.minibatch
        adj_w[-1]=np.outer(adata[-2],adj_b[-1])
        for i in xrange(2,len(self.layers)):
            sigmaprime=dsig(np.dot(adata[-i-1],self.weights[-i])+self.bias[-i])
            adj_b[-i]=sigmaprime*np.dot(adj_b[-i+1],self.weights[-i+1].T)
            adj_w[-i]=np.outer(adata[-i-1],adj_b[-i])
        
        self.avgdelta=[a+b for a,b in zip(adj_b,self.avgdelta)]

        return (adj_w,adj_b)
            
    def feedforward(self,sinput):
        fwd_data=[np.zeros(x) for x in self.layers]
        #First layer activates based on input. Typically this brings down performance.
        #fwd_data[0]=sigmoid(sinput)
        #First layer (the same dimension as input) simply let the signal
        #passing through. 

        fwd_data[0]=sinput
        for j in np.arange(len(self.layers)-1):
            fwd_data[j+1]=sigmoid(np.dot(fwd_data[j],self.weights[j])+self.bias[j])
        return fwd_data

    def pred(self,test_input,test_label):
        res_state=np.array([self.feedforward(x)[-1] for x in test_input])
        res_nn=np.array([np.argmax(self.feedforward(x)[-1]) for x in test_input])
        #res_nn=np.array([np.argmax(x) for x in res_state])
        label=np.array([np.argmax(x) for x in test_label])
        hits=sum(res_nn==label)*1./len(test_input)
        cost=costFn(res_state,test_label)
        return hits,cost
        
def costFn(a,t):
    p=np.array([x/np.sum(x) for x in a])
    num_samples=len(a[:,0])
    return np.sum(-t*np.nan_to_num(np.log(p))-
                  (1-t)*np.nan_to_num(np.log(1.-p)))/num_samples

def sigmoid(x):
    return 1./(1.+np.exp(-np.clip(x,-100,100)))

def dsig(x):
    return (1.-sigmoid(x))*sigmoid(x)

