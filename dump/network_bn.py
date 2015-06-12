"""
Batch Normalization

Following Ioffe & Szegedy's 2015 paper
***
Important:
The Batch-Normalization is applied to whatever coming in 
to the hidden layer. Not the output of the layer. For example:
u=g(y) being an activation, W is the weights linking two neighbouring layers,
BN applies to the following quantity: BN(Wu), not BN(u).

Whitening also happens to the initial input. W can be viewed as 1 for the
1st layer of NN. At this layer:
gamma = 1.
beta = 1.
gamma and beta is the var and mean of the Normalized input for each layer.
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
        # growth rate delta at the last layer.
        self.deltaLFn=fn.grFn[(actFn,costFn)]
        
        # parameter initialization.
        # self.weights=[np.random.randn(x,y)/np.sqrt(x)
        #               for x,y in zip(self.layers[:-1],self.layers[1:])]
        with open('weights.json','r') as fw:
            data=json.load(fw)
        self.weights=[np.array(x) for x in data['weights']]
        self.gammas=[np.random.randn(x) for x in layers]
        self.betas=[np.random.randn(x) for x in layers]
        self.gammas[0]=np.ones(layers[0])
        self.betas[0]=np.zeros(layers[0])
        
        # useful quantities when performing BP. the notions follow
        # Ioffe&Szegedy's paper.
        self.stds=[np.zeros(l) for l in layers]
        self.means=copy.deepcopy(self.stds)
        self.xhats=[np.zeros((self.batchsize,l)) for l in layers]
        self.ys=copy.deepcopy(self.xhats)
        self.us=copy.deepcopy(self.xhats)
        self.deltas=copy.deepcopy(self.xhats)

        # model_check is to keep track of NN's performance on every E epochs
        # It records accuracy, baseline cost (crossEntropy).
        self.xhats_inf=[]
        self.ys_inf=[]
        self.us_inf=[]
        self.model_check=[]

        #check the insights of math derv
        self.idi=2
        self.idj=6
        self.idl=-2
        self.recstdErr=[]
        self.recstdU=[]
        self.recstdX=[]
        self.recadjW=[]
        self.recgamma=[]
        self.t1cos=[]
        self.t2cos=[]
        self.t3cos=[]
    #---------------------------------------------------------------------
    def sgd(self,inputs,outputs,test_input,test_labels,
            eval_timing=True,inf_check=False,check_freq=1):

        total_num_train=len(inputs[:,0])
        num_minibatch,res=divmod(total_num_train,self.batchsize)
        if res!=0:
            print "Mini-batch can't be divided by total number of tasks."
            raise SystemExit

        # pop_trains is used to record the trained mini-batches. needed when computing
        # population stats - means and vars that feed to inference.
        pop_trains=[]

        data={'epoch':self.epochs,'batch_per_epoch':num_minibatch,
              'check_freq':check_freq}
        wdata={'epoch':self.epochs,'batch_per_epoch':num_minibatch,
              'check_freq':check_freq}

        data_gammavar=[]
        
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
                    pop_mean,pop_var=self.pop_stats(inputs,np.array(pop_trains),p,q)
                    # when there is more than 1 test sample, one can normalize each layer
                    # just using test data. For online testing, the receipt from the paper
                    # is followed.
                    # use population stats:
                    test_results=self.inference(test_input,test_labels,
                                                 pop_mean,pop_var)
                    # no population stats:
                    # test_results=self.inference(test_input,test_label)
                    self.model_check.append(test_results) 
                    res=self.inf_fwd(test_input,pop_mean,pop_var)

                    key='p'+str(p)+'q'+str(q)+'hl-1'
                    data[key]=self.ys_inf[-2]
                    #wdata[key]=copy.deepcopy(self.weights[-1])
                    key='p'+str(p)+'q'+str(q)+'hl-2'
                    data[key]=self.ys_inf[-3]
                    #wdata[key]=copy.deepcopy(self.weights[-2])
                    key='p'+str(p)+'q'+str(q)
                    wdata[key]=copy.deepcopy(self.weights)
                    lrate_scale=[x/y for x,y in zip(self.gammas[1:-1],self.stds[1:-1])]
                    data_gammavar.append(lrate_scale)
                    
            if eval_timing:
                tend=time.clock()
                print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)

        if inf_check:
            with open('bn-dist.pickle','wb') as fout_pickle:
                pkl.dump(data,fout_pickle,protocol=-1)
            with open('lrate_adj.pickle','wb') as fout:
                pkl.dump(data_gammavar,fout,protocol=-1)
            with open('batchnorm_weights.pickle','wb') as fout:
                pkl.dump(wdata,fout,protocol=-1)

        with open('insights.pickle','wb') as fout:
            data=[np.array(self.recgamma),
                  np.array(self.recstdErr),
                  np.array(self.recstdU),
                  np.array(self.recstdX),
                  np.array(self.t3cos),
                  np.array(self.t1cos),
                  np.array(self.t2cos),
                  np.array(self.recadjW)]
            pkl.dump(data,fout,protocol=-1)
            
    #---------------------------------------------------------------------
    def batch_update(self,inputs,labels):
        self.feedforward(inputs)
        dw,dgamma,dbeta=self.bp(labels)
        self.weights=[w-self.learnrate*w1 for w,w1 in zip(self.weights,dw)]
        self.gammas=[g-self.learnrate*g1 for g,g1 in zip(self.gammas,dgamma)]
        self.betas=[b-self.learnrate*b1 for b,b1 in zip(self.betas,dbeta)]

    #---------------------------------------------------------------------
    def feedforward(self,batch_inputs):
        eps=1.e-15
        # normalize initial inputs.
        u_in=copy.deepcopy(batch_inputs)
        self.means[0]=np.mean(u_in,0)
        self.stds[0]=np.std(u_in,0)
        self.xhats[0]=(u_in-self.means[0])/(self.stds[0]+eps)
        self.ys[0]=self.gammas[0]*self.xhats[0]+self.betas[0]
        #self.us[0]=np.array([self.actfn(y) for y in self.ys[0]])
        # First layer just perform whitening. It doesn't activate input
        # With activation at first layer, the performance is worse than no activation.
        self.us[0]=np.array([y for y in self.ys[0]])
        
        for l in np.arange(1,len(self.layers)):
            # wu=W*U, e.g. the "activation" mentioned in the paper
            wu=np.dot(self.us[l-1],self.weights[l-1])
            self.means[l]=np.mean(wu,0)
            self.stds[l]=np.std(wu,0)
            self.xhats[l]=(wu-self.means[l])/(self.stds[l]+eps)
            self.ys[l]=self.gammas[l]*self.xhats[l]+self.betas[l]
            self.us[l]=np.array([self.actfn(y) for y in self.ys[l]])

    #---------------------------------------------------------------------        
    def bp(self,labels):
        eps=1.e-15
        # deltas is the growth rate in backpropagation.
        # delta=d[Cost]/d[y=Wu]
        #deltas=[np.zeros((self.batchsize,l)) for l in self.layers]
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_gamma=[np.zeros(r.shape) for r in self.gammas]
        adj_beta=[np.zeros(b.shape) for b in self.betas]

        # the last layer may be specially treated
        # e.g. using softmax as activiation func for last layer.
        # This code needs to be modified if Softmax is used due to
        # its matrix output. (Other functions' output being 1-d array)
        self.deltas[-1]=np.array([self.deltaLFn(a,y,t,self.batchsize)
                    for a,y,t in zip(self.us[-1],self.ys[-1],labels)])
        adj_gamma[-1]=np.sum(self.deltas[-1]*self.xhats[-1],0)
        adj_beta[-1]=np.sum(self.deltas[-1],0)
        
        # adjusting weigths
        coeff=self.deltas[-1]*self.gammas[-1]/(self.stds[-1]+eps)
        term1=sum([np.outer(a,b) for a,b in zip(self.us[-2],coeff)])
        term2=np.outer(np.mean(self.us[-2],0),np.sum(coeff,0))
        coeff=self.xhats[-1]*np.mean(coeff*self.xhats[-1],0)
        term3=sum([np.outer(a,b) for a,b in zip(self.us[-2],coeff)])
        adj_w[-1]=term1-term2-term3


        # Don't apply whitening to 1st layer's input, e.g. gamma=1, beta=0
        # for 1st layer.
        # for i in xrange(2,len(self.layers)):
        #     term1=self.gammas[-i+1]/(self.stds[-i+1]+eps)*\
        #     (deltas[-i+1]-np.mean(deltas[-i+1],0)-self.xhats[-i+1]*\
        #      np.mean(deltas[-i+1]*self.xhats[-i+1],0))
        #     term2=np.array([self.dactfn(y) for y in self.ys[-i]])
        #     deltas[-i]=term2*np.dot(term1,self.weights[-i+1].T)
        #     adj_gamma[-i]=np.sum(deltas[-i]*self.xhats[-i],0)
        #     adj_beta[-i]=np.sum(deltas[-i])
        #     coeff=self.gammas[-i]/(self.stds[-i]+eps)*deltas[-i]
        #     term1=sum([np.outer(a,b) for a,b in zip(self.us[-i-1],coeff)])
        #     term2=np.outer(np.mean(self.us[-i-1],0),np.sum(coeff,0))
        #     coeff=self.xhats[-i]*np.mean(self.xhats[-i]*coeff,0)
        #     term3=sum([np.outer(a,b) for a,b in zip(self.us[-i-1],coeff)])
        #     adj_w[-i]=term1-term2-term3

        # Adjusting the initial input's gamma and beta too.
        for i in xrange(2,len(self.layers)+1):
            term1=self.gammas[-i+1]/(self.stds[-i+1]+eps)*\
            (self.deltas[-i+1]-np.mean(self.deltas[-i+1],0)-self.xhats[-i+1]*\
             np.mean(self.deltas[-i+1]*self.xhats[-i+1],0))
            term2=np.array([self.dactfn(y) for y in self.ys[-i]])
            self.deltas[-i]=term2*np.dot(term1,self.weights[-i+1].T)
            adj_gamma[-i]=np.sum(self.deltas[-i]*self.xhats[-i],0)
            adj_beta[-i]=np.sum(self.deltas[-i],0)
            # adjusting weights
            if i<len(self.layers):
                coeff=self.gammas[-i]/(self.stds[-i]+eps)*self.deltas[-i]
                term1=sum([np.outer(a,b) for a,b in zip(self.us[-i-1],coeff)])
                term2=np.outer(np.mean(self.us[-i-1],0),np.sum(coeff,0))
                coeff=self.xhats[-i]*np.mean(self.xhats[-i]*coeff,0)
                term3=sum([np.outer(a,b) for a,b in zip(self.us[-i-1],coeff)])
                adj_w[-i]=term1-term2-term3

        # insights
        self.recadjW.append(adj_w[self.idl][self.idj,self.idi])
        err=self.deltas[self.idl][:,self.idi]
        xx=np.std(err)
        self.recstdErr.append(xx)
        errhat=(err-np.mean(err))/xx
        #-----------
        uu=self.us[self.idl-1][:,self.idj]
        xx=np.std(uu)
        self.recstdU.append(xx)
        uhat=(uu-np.mean(uu))/xx
        #----------
        xhat=self.xhats[self.idl][:,self.idi]
        theta1cos=np.dot(errhat,xhat)/self.batchsize
        theta2cos=np.dot(xhat,uhat)/self.batchsize
        theta3cos=np.dot(errhat,uhat)/self.batchsize
        self.t1cos.append(theta1cos)
        self.t2cos.append(theta2cos)
        self.t3cos.append(theta3cos)
        self.recstdX.append(self.stds[self.idl][self.idi])
        self.recgamma.append(self.gammas[self.idl][self.idi])
        
        return (adj_w,adj_gamma,adj_beta)
            
    #---------------------------------------------------------------------        
    # inference function return accuracy,cost
    def inference(self,test_input,test_label,gmean,gvar):
        tests=copy.deepcopy(test_input)
        results=self.inf_fwd(tests,gmean,gvar)
        res_nn=np.array([np.argmax(x) for x in results])
        label=np.array([np.argmax(x) for x in test_label])
        hits=sum(res_nn==label)
        accuracy=1.0*hits/len(test_input)
        cost=self.costfn(results,test_label)
        return accuracy,cost

    #---------------------------------------------------------------------
    # FeedForwards at inference. Using population stats

    def inf_fwd(self,tests,gmean,gvar):
        eps=1.e-15
        self.xhats_inf=[np.zeros((len(tests[:,0]),l)) for l in self.layers]
        self.ys_inf=copy.deepcopy(self.xhats_inf)
        self.us_inf=copy.deepcopy(self.xhats_inf)
        
        self.xhats_inf[0]=(tests-gmean[0])/np.sqrt(gvar[0]+eps)
        self.ys_inf[0]=self.gammas[0]*self.xhats_inf[0]+self.betas[0]
        #self.us[0]=[self.actfn(y) for y in self.ys[0]]
        self.us_inf[0]=[y for y in self.ys_inf[0]]
        
        for l in np.arange(1,len(self.layers)):
            # wx is W*U, e.g. the weight multiply inputs
            wu=np.dot(self.us_inf[l-1],self.weights[l-1])
            self.xhats_inf[l]=(wu-gmean[l])/np.sqrt(gvar[l]+eps)
            self.ys_inf[l]=self.gammas[l]*self.xhats_inf[l]+self.betas[l]
            self.us_inf[l]=[self.actfn(y) for y in self.ys_inf[l]]

        return self.us_inf[-1]

    #---------------------------------------------
    # Normalize test data without population stats
    #     eps=1.e-15
    #     # normalize initial inputs.
    #     u_in=copy.deepcopy(tests)
    #     self.means[0]=np.mean(u_in,0)
    #     self.stds[0]=np.std(u_in,0)# T- array/list
    #     self.xhats[0]=(u_in-self.means[0])/(self.stds[0]+eps)
    #     self.ys[0]=self.gammas[0]*self.xhats[0]+self.betas[0]
    #     self.us[0]=np.array([self.actfn(y) for y in self.ys[0]])
        
    #     for l in np.arange(1,len(self.layers)):
    #         wu=np.dot(self.us[l-1],self.weights[l-1])
    #         self.means[l]=np.mean(wu,0)
    #         self.stds[l]=np.std(wu,0)
    #         self.xhats[l]=(wu-self.means[l])/(self.stds[l]+eps)
    #         self.ys[l]=self.gammas[l]*self.xhats[l]+self.betas[l]
    #         self.us[l]=np.array([self.actfn(y) for y in self.ys[l]])

    #     return self.us[-1]

    # Compute the means and vars for inference. However, it is not clear how many training
    # mini-batches need to be included in
    # Var(x)<-m/(m-1)*E[var(B)]
    # I think the purpose is to using subsamples (mini-batch) do approximate the real data distr.
    # This may be viewed as a boostrap process. 

    def pop_stats(self,inputs,p_trains,p,q):
        bnum_per_epoch=len(p_trains[0,:])/self.batchsize
        p_mean=[np.zeros(l) for l in self.layers]
        p_var=[np.zeros(l) for l in self.layers]
        #num_batches=p*bnum_per_epoch+q+1
        # This includes all the previous epochs.
        # for i in np.arange(p+1):
        #     for j in np.arange(q+1):
        #         data=inputs[p_trains[i,j*self.batchsize:(j+1)*self.batchsize]]
        #         p_mean,p_var=self.pop_fwd(data,p_mean,p_var,num_batches)

        # This just computes the total training sample. No previous epoches etc.
        # num_batches=bnum_per_epoch
        # for j in np.arange(bnum_per_epoch):
        #     data=inputs[p_trains[-1,j*self.batchsize:(j+1)*self.batchsize]]
        #     p_mean,p_var=self.pop_fwd(data,p_mean,p_var,num_batches)
            
        # Bootstrap: get estimate of data's mean and var from subsampling.
        bstrp_steps=bnum_per_epoch*3
        sample_size=len(p_trains[0,:])
        for n in np.arange(bstrp_steps):
            subsample_idx=np.random.randint(0,sample_size,self.batchsize)
            data=inputs[subsample_idx]
            p_mean,p_var=self.pop_fwd(data,p_mean,p_var,bstrp_steps)
        return p_mean,p_var

    #----------------------------------------------------------------------
    # feedforward to compute population stats.
    def pop_fwd(self,batch_inputs,pop_mean,pop_var,num_batches):
        eps=1.e-15
        u_in=copy.deepcopy(batch_inputs)
        self.means[0]=np.mean(u_in,0)
        self.stds[0]=np.std(u_in,0)
        self.xhats[0]=(u_in-self.means[0])/(self.stds[0]+eps)
        self.ys[0]=self.gammas[0]*self.xhats[0]+self.betas[0]
        #self.us[0]=np.array([self.actfn(y) for y in self.ys[0]])
        self.us[0]=np.array([y for y in self.ys[0]])
        
        pop_mean[0]+=self.means[0]/num_batches
        pop_var[0]+=np.power(self.stds[0],2.)/num_batches*\
                     (self.batchsize/(self.batchsize-1.))
        
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(self.us[l-1],self.weights[l-1])
            self.means[l]=np.mean(wu,0)
            self.stds[l]=np.std(wu,0)
            self.xhats[l]=(wu-self.means[l])/(self.stds[l]+eps)
            self.ys[l]=self.gammas[l]*self.xhats[l]+self.betas[l]
            self.us[l]=np.array([self.actfn(y) for y in self.ys[l]])
            pop_mean[l]+=self.means[l]/num_batches
            pop_var[l]+=np.power(self.stds[l],2.)/num_batches*\
                         (self.batchsize/(self.batchsize-1.))

        return pop_mean,pop_var
