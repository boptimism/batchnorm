"""
BN-V1
"""
import numpy as np
import time
import cPickle as pkl

class BNv1:
    def __init__(self,layers,learnrate,lrate_decay,batchsize,epochs,weights,bias):

        self.layers=layers
        self.learnrate=learnrate
        self.lrate_decay=lrate_decay
        
        self.batchsize=batchsize
        self.epochs=epochs
        self.weights=weights
        self.bias=bias
        
        self.us=[np.zeros([batchsize,l]) for l in self.layers]
        self.xs=[np.zeros([batchsize,l]) for l in self.layers]
        self.ys=[np.zeros([batchsize,l]) for l in self.layers]
        self.deltas=[np.zeros([batchsize,l]) for l in self.layers[1:]]
        self.means=[np.zeros(l) for l in self.layers]
        
        # moving average
        #self.moving_means=[np.zeros(l) for l in self.layers]
        # non moving average
        self.means_inf=[np.zeros(l) for l in self.layers]
        
        self.test_accu=[]
        self.test_cost=[]
        self.train_accu=[]
        self.train_cost=[]
        
    def sgd(self,train_inputs,train_labels,test_inputs,test_labels,init_type,rec_freq,
            test_check=True,train_check=False,rec_check=False):
        
        num_of_trains=len(train_labels)
        batch_per_epoch=num_of_trains/self.batchsize
        idx_epoch=np.arange(num_of_trains)
        if rec_check:
            rec_ud=[]
            rec_u=[]
            rec_d=[]
            rec_dw=[]
            
        for p in np.arange(self.epochs):
            tstart=time.clock()
            np.random.shuffle(idx_epoch)

            for q in np.arange(batch_per_epoch):
                idx_batch=idx_epoch[q*self.batchsize:(q+1)*self.batchsize]
                batch_data=train_inputs[idx_batch]
                batch_label=train_labels[idx_batch]

                self.feedforward(batch_data)
                
                dw,db=self.bp(batch_label)
                num_of_batches=p*batch_per_epoch+q+1
                self.batch_update(dw,db,num_of_batches)

                if rec_check and not (q+1)%rec_freq:
                    rec_tmp=[np.dot(x.T,y)/self.batchsize
                             for x,y in zip(self.us[:-1],self.deltas)]
                    rec_ud.append(rec_tmp)
                    rec_tmp=[np.mean(x,0) for x in self.us[:-1]]
                    rec_u.append(rec_tmp)
                    rec_tmp=[np.mean(x,0) for x in self.deltas]
                    rec_d.append(rec_tmp)
                    rec_dw.append(dw)
                # mini-batch update performance
                if test_check and not (q+1)%rec_freq:
                    bs_inf=60
                    ep_inf=3
                    self.pop_stats(train_inputs,ep_inf,bs_inf)
                    accu,cost=self.inference(test_inputs,test_labels)
                    self.test_accu.append(accu)
                    self.test_cost.append(cost)
                if train_check and not (q+1)%rec_freq:
                    accu,cost=self.inference(train_inputs,train_labels)
                    self.train_accu.append(accu)
                    self.train_cost.append(cost)


            tend=time.clock()
            
            print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)

        if rec_check:
            data={'u_delta_avg':rec_ud,
                  'u_avg':rec_u,
                  'delta_avg':rec_d,
                  'dw':rec_dw
            }
            fname='rec_bnv1_'+init_type+'.pickle'
            with open(fname,'wb') as frec:
                pkl.dump(data,frec,protocol=-1)
                    
    def feedforward(self,batch_data):
        self.us[0]=batch_data
        for l in np.arange(1,len(self.layers)):
            self.xs[l]=np.dot(self.us[l-1],self.weights[l-1])
            self.means[l]=np.mean(self.xs[l],0)
            self.ys[l]=self.xs[l]-self.means[l]+self.bias[l]
            if l<len(self.layers)-1:
                self.us[l]=sigmoid(self.ys[l])
            else:
                self.us[l]=softmax(self.ys[l])

    def bp(self,batch_label):
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_b=[np.zeros(b.shape) for b in self.bias]
        
        self.deltas[-1]=(self.us[-1]-batch_label)/self.batchsize
        adj_b[-1]=np.sum(self.deltas[-1],0)
        u_shift=self.us[-2]-np.mean(self.us[-2],0)
        adj_w[-1]=np.dot(u_shift.T,self.deltas[-1])

        for l in np.arange(2,len(self.layers)):
            delta_shift=self.deltas[-l+1]-np.mean(self.deltas[-l+1],0)
            self.deltas[-l]=sigmoid(self.ys[-l])*(1.-sigmoid(self.ys[-l]))*\
                        np.dot(delta_shift,self.weights[-l+1].T)
            adj_b[-l]=np.sum(self.deltas[-l],0)
            u_shift=self.us[-l-1]-np.mean(self.us[-l-1],0)
            adj_w[-l]=np.dot(u_shift.T,self.deltas[-l])

        return adj_w,adj_b
            
    def batch_update(self,dw,db,t):
        self.weights=[w-self.learnrate/(1.+self.lrate_decay*t)*dnw for w,dnw in zip(self.weights,dw)]
        self.bias=[b-self.learnrate/(1.+self.lrate_decay*t)*dnb for b,dnb in zip(self.bias,db)]
            
    def pop_stats(self,train_inputs,ep_inf,bs_inf):
        num_of_trains=len(train_inputs)
        bs_per_ep_inf=num_of_trains/bs_inf
        num_of_batches=ep_inf*bs_per_ep_inf
        idx_inf=np.arange(num_of_trains)

        self.means_inf=[np.zeros(l) for l in self.layers]
        
        for p in np.arange(ep_inf):
            np.random.shuffle(idx_inf)
            for q in np.arange(bs_per_ep_inf):
                idx_batch=idx_inf[q*bs_inf:(q+1)*bs_inf]
                batch_data=train_inputs[idx_batch]
                us_inf=batch_data
                for l in np.arange(1,len(self.layers)):
                    xs_inf=np.dot(us_inf,self.weights[l-1])
                    wumean=np.mean(xs_inf,0)
                    self.means_inf[l]=self.means_inf[l]+wumean/num_of_batches
                    ys_inf=xs_inf-wumean+self.bias[l]
                    if l<len(self.layers)-1:
                        us_inf=sigmoid(ys_inf)
                    else:
                        us_inf=softmax(ys_inf)
                        
    def inference(self,test_inputs,test_labels):
        us_inf=test_inputs
        for l in np.arange(1,len(self.layers)):
            wu=np.dot(us_inf,self.weights[l-1])
            ys_inf=wu-self.means_inf[l]+self.bias[l]
            if l<len(self.layers)-1:
                us_inf=sigmoid(ys_inf)
            else:
                us_inf=softmax(ys_inf)
        cost=costFn(us_inf,test_labels)
        label_inf=np.array([np.argmax(s) for s in us_inf])
        label_giv=np.array([np.argmax(s) for s in test_labels])
        hits=sum(label_inf==label_giv)*1./len(test_inputs)

        return hits,cost
         
        # labels=np.array([np.argmax(x) for x in test_labels])
        # labels_inf=[]

        # for sample in test_inputs:
        #     us_inf=sample
        #     for l in np.arange(1,len(self.layers)):
        #         xs_inf=np.dot(us_inf,self.weights[l-1])
        #         #ys_inf=xs_inf-self.moving_means[l]+self.bias[l]
        #         ys_inf=xs_inf-self.means_inf[l]+self.bias[l]
        #         us_inf=sigmoid(ys_inf)
        #     labels_inf.append(us_inf)
        # # us_inf=[np.zeros()]
        # # res_state=np.array([self.feedforward(x)[-1] for x in test_input])
        # # res_nn=np.array([np.argmax(self.feedforward(x)[-1]) for x in test_input])
        # # #res_nn=np.array([np.argmax(x) for x in res_state])
        # # label=np.array([np.argmax(x) for x in test_label])
        # labels_inf=np.array(labels_inf)
        # cost=costFn(labels_inf,test_labels)
        # labels_inf=[np.argmax(x) for x in labels_inf]
        # hits=sum(labels_inf==labels)*1./len(test_inputs)
        # return hits,cost
        
def costFn(labels_inf,labels):
    return np.sum(-labels*np.nan_to_num(np.log(labels_inf)))/len(labels)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x),1)).T
