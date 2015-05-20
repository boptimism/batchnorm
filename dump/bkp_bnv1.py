"""
Bn-V1
"""

import numpy as np
import copy
import time
import nn_functions as fn
import cPickle as pkl
import json

#---------------------------------------------------------------------
class bnN:

    def __init__(self,layers,learnrate,batchsize,epochs,
                 costFn='crossEntropy',actFn='sigmoid'):

        # hyperparameters
        self.layers=layers
        self.learnrate=learnrate
        self.batchsize=batchsize
        self.epochs=epochs

        # Cost funciton,activation function and its derivitive
        self.costfn=fn.cFn[costFn]
        self.actfn=fn.aFn[actFn]
        self.dactfn=fn.daFn[actFn]
        self.deltaLFn=fn.grFn[(actFn,costFn)]
        
        # parameter initialization.
        # self.weights=[np.random.randn(x,y)/np.sqrt(x)
        #               for x,y in zip(self.layers[:-1],self.layers[1:])]
        with open('weights.json','r') as fw:
            data=json.load(fw)
        self.weights=[np.array(x) for x in data['weights']]
        self.betas=[np.random.randn(x) for x in layers]
        self.betas[0]=np.zeros(layers[0])
        
        self.means=[np.zeros(l) for l in layers]
        self.xs=[np.zeros((self.batchsize,l)) for l in layers]
        self.ys=copy.deepcopy(self.xs)
        self.us=copy.deepcopy(self.xs)
        self.deltas=copy.deepcopy(self.xs)

        # model_check is to keep track of NN's performance on every E epochs
        # It records accuracy, baseline cost (crossEntropy).
        self.ys_inf=[]
        self.us_inf=[]
        self.model_check=[]

    #---------------------------------------------------------------------
    def sgd(self,inputs,outputs,test_input,test_labels,
            eval_timing=True,inf_check=False,check_freq=1):

        total_num_train=len(inputs[:,0])
        num_minibatch,res=divmod(total_num_train,self.batchsize)
        if res!=0:
            print "Mini-batch can't be divided by total number of tasks."
            raise SystemExit

        # pop_trains is used to record the trained mini-batches. needed when computing
        # population stats - means that feed to inference.
        pop_trains=[]
        # wdata={'epoch':self.epochs,'batch_per_epoch':num_minibatch,
        #       'check_freq':check_freq}

        for p in np.arange(self.epochs):
            if eval_timing:
                tstart=time.clock()

            dataindex=np.arange(total_num_train)
            np.random.shuffle(dataindex)
            rand_trdata=inputs[dataindex]
            rand_trlabel=outputs[dataindex]
            pop_trains.append(dataindex)
            
            for q in np.arange(num_minibatch):
                batch_data=rand_trdata[q*self.batchsize:(q+1)*self.batchsize]
                batch_label=rand_trlabel[q*self.batchsize:(q+1)*self.batchsize]
                self.batch_update(batch_data,batch_label)
                num_batches=q+p*num_minibatch+1

                if inf_check and (not num_batches%check_freq):
                    pop_mean=self.pop_stats(inputs,np.array(pop_trains),p,q)
                    test_results=self.inference(test_input,test_labels,pop_mean)
                    self.model_check.append(test_results) 
                    res=self.inf_fwd(test_input,pop_mean)

                    # key='p'+str(p)+'q'+str(q)
                    # wdata[key]=copy.deepcopy(self.weights)
                    
            if eval_timing:
                tend=time.clock()
                print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)

        # if inf_check:
        #     with open('bn_v1_weights.pickle','wb') as fout:
        #         pkl.dump(wdata,fout,protocol=-1)

    #---------------------------------------------------------------------
    def batch_update(self,inputs,labels):
        self.feedforward(inputs)
        dw,dbeta=self.bp(labels)
        self.weights=[w-self.learnrate*w1 for w,w1 in zip(self.weights,dw)]
        self.betas=[b-self.learnrate*b1 for b,b1 in zip(self.betas,dbeta)]

    #---------------------------------------------------------------------
    def feedforward(self,batch_inputs):
        # normalize initial inputs.
        data_in=copy.deepcopy(batch_inputs)
        self.means[0]=np.mean(data_in,0)
        self.xs[0]=data_in-self.means[0]
        self.ys[0]=self.xs[0]+self.betas[0]
        self.us[0]=np.array([y for y in self.ys[0]])
        
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(self.us[l-1],self.weights[l-1])
            self.means[l]=np.mean(wu,0)
            self.xs[l]=wu-self.means[l]
            self.ys[l]=self.xs[l]+self.betas[l]
            self.us[l]=np.array([self.actfn(y) for y in self.ys[l]])

    #---------------------------------------------------------------------        
    def bp(self,labels):
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_beta=[np.zeros(b.shape) for b in self.betas]

        self.deltas[-1]=np.array([self.deltaLFn(a,y,t,self.batchsize)
                    for a,y,t in zip(self.us[-1],self.ys[-1],labels)])
        adj_beta[-1]=np.sum(self.deltas[-1],0)
        
        ushift=self.us[-2]-np.mean(self.us[-2],0)
        adj_w[-1]=sum([np.outer(a,b) for a,b in zip(ushift,self.deltas[-1])])

        for i in xrange(2,len(self.layers)+1):
            delta_shift=self.deltas[-i+1]-np.mean(self.deltas[-i+1],0)
            dgdy=np.array([self.dactfn(y) for y in self.ys[-i]])
            self.deltas[-i]=dgdy*np.dot(delta_shift,self.weights[-i+1].T)
            adj_beta[-i]=np.sum(self.deltas[-i],0)
            if i<len(self.layers):
                ushift=self.us[-i-1]-np.mean(self.us[-i-1],0)
                adj_w[-i]=sum([np.outer(a,b) for a,b in zip(ushift,self.deltas[-i])])

        return (adj_w,adj_beta)
            
    #---------------------------------------------------------------------        
    # inference function return accuracy,cost
    def inference(self,test_input,test_label,gmean):
        tests=copy.deepcopy(test_input)
        results=self.inf_fwd(tests,gmean)
        res_nn=np.array([np.argmax(x) for x in results])
        label=np.array([np.argmax(x) for x in test_label])
        hits=sum(res_nn==label)
        accuracy=1.0*hits/len(test_input)
        cost=self.costfn(results,test_label)
        return accuracy,cost

    #---------------------------------------------------------------------
    # FeedForwards at inference. Using population stats

    def inf_fwd(self,tests,gmean):
        eps=1.e-15
        self.xs_inf=[np.zeros((len(tests[:,0]),l)) for l in self.layers]
        self.ys_inf=copy.deepcopy(self.xs_inf)
        self.us_inf=copy.deepcopy(self.xs_inf)
        
        self.xs_inf[0]=tests-gmean[0]
        self.ys_inf[0]=self.xs_inf[0]+self.betas[0]
        #self.us[0]=[self.actfn(y) for y in self.ys[0]]
        self.us_inf[0]=[y for y in self.ys_inf[0]]
        
        for l in np.arange(1,len(self.layers)):
            # wx is W*U, e.g. the weight multiply inputs
            wu=np.dot(self.us_inf[l-1],self.weights[l-1])
            self.xs_inf[l]=wu-gmean[l]
            self.ys_inf[l]=self.xs_inf[l]+self.betas[l]
            self.us_inf[l]=np.array([self.actfn(y) for y in self.ys_inf[l]])

        return self.us_inf[-1]

    def pop_stats(self,inputs,p_trains,p,q):
        bnum_per_epoch=len(p_trains[0,:])/self.batchsize
        p_mean=[np.zeros(l) for l in self.layers]
        bstrp_steps=bnum_per_epoch*3
        sample_size=len(p_trains[0,:])
        for n in np.arange(bstrp_steps):
            subsample_idx=np.random.randint(0,sample_size,self.batchsize)
            data=inputs[subsample_idx]
            p_mean=self.pop_fwd(data,p_mean,bstrp_steps)
        return p_mean

    #----------------------------------------------------------------------
    # feedforward to compute population stats.
    def pop_fwd(self,batch_inputs,pop_mean,num_batches):
        u_in=copy.deepcopy(batch_inputs)
        self.means[0]=np.mean(u_in,0)
        self.xs[0]=u_in-self.means[0]
        self.ys[0]=self.xs[0]+self.betas[0]
        #self.us[0]=np.array([self.actfn(y) for y in self.ys[0]])
        self.us[0]=np.array([y for y in self.ys[0]])
        
        pop_mean[0]+=self.means[0]/num_batches
        
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(self.us[l-1],self.weights[l-1])
            self.means[l]=np.mean(wu,0)
            self.xs[l]=wu-self.means[l]
            self.ys[l]=self.xs[l]+self.betas[l]
            self.us[l]=np.array([self.actfn(y) for y in self.ys[l]])
            pop_mean[l]+=self.means[l]/num_batches

        return pop_mean
