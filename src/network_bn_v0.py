"""
BN-V0
Last layer: Softmax
Loss Func: Cross-Entropy with 1 term: C=-Sum[Label*Log(Output)]
"""
import numpy as np
import time

class BNv0:
    def __init__(self,layers,learnrate,batchsize,epochs,weights,bias,gammas):

        self.layers=layers
        self.learnrate=learnrate
        self.batchsize=batchsize
        self.epochs=epochs
        self.weights=weights
        self.bias=bias
        self.gammas=gammas
        
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
        
    def sgd(self,train_inputs,train_labels,test_inputs,test_labels,
            test_check=True,train_check=False):
        
        num_of_trains=len(train_labels)
        batch_per_epoch=num_of_trains/self.batchsize
        idx_epoch=np.arange(num_of_trains)
        
        for p in np.arange(self.epochs):
            tstart=time.clock()
            np.random.shuffle(idx_epoch)

            for q in np.arange(batch_per_epoch):
                idx_batch=idx_epoch[q*self.batchsize:(q+1)*self.batchsize]
                batch_data=train_inputs[idx_batch]
                batch_label=train_labels[idx_batch]

                self.feedforward(batch_data)
                
                dw,db,dg=self.bp(batch_label)
                
                self.batch_update(dw,db,dg)

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
            
            print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)

    def feedforward(self,batch_data):
        eps=1.e-15
        self.us[0]=batch_data
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(self.us[l-1],self.weights[l-1])
            self.xhats[l]=(wu-np.mean(wu,0))/(np.std(wu,0)+eps)
            self.stds[l]=np.std(wu,0)
            self.ys[l]=self.gammas[l]*self.xhats[l]+self.bias[l]
            if l<len(self.layers)-1:
                self.us[l]=sigmoid(self.ys[l])
            else:
                self.us[l]=softmax(self.ys[l])
            

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
            adj_b[-l]=np.sum(self.deltas[-l],0)
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
                    if l<len(self.layers)-1:
                        us_inf=sigmoid(ys_inf)
                    else:
                        us_inf=softmax(ys_inf)
        
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
            if l<len(self.layers)-1:
                us_inf=sigmoid(ys_inf)
            else:
                us_inf=softmax(ys_inf)
        cost=costFn(us_inf,test_labels)
        label_inf=np.array([np.argmax(s) for s in us_inf])
        label_giv=np.array([np.argmax(s) for s in test_labels])
        hits=sum(label_inf==label_giv)*1./len(test_inputs)
        
        return hits,cost
        
def costFn(labels_inf,labels):
    p=np.array([x/np.sum(x) for x in labels_inf])
    num_tests=len(labels)
    return np.sum(-labels*np.nan_to_num(np.log(p)))/num_tests

def sigmoid(x):
#    return 1./(1.+np.exp(-np.clip(x,-100,100)))
    return 1./(1.+np.exp(-x))

def softmax(x):#input is a matrix with dimension n_samples x n_nodes
    return (np.exp(x).T/np.sum(np.exp(x),1)).T

