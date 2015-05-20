import numpy as np
import time

class Baseline:
    def __init__(self,layers,learnrate,batchsize,epochs,weights,bias):

        self.layers=layers
        self.learnrate=learnrate
        self.batchsize=batchsize
        self.epochs=epochs
        self.weights=weights
        self.bias=bias
        
        self.us=[np.zeros([batchsize,l]) for l in self.layers]
        self.ys=[np.zeros([batchsize,l]) for l in self.layers]
        self.deltas=[np.zeros([batchsize,l]) for l in self.layers[1:]]
        
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

                dw,db=self.bp(batch_label)
                
                self.batch_update(dw,db)

            if test_check:
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
        return hits,cost
        
def costFn(labels_inf,labels):
    p=np.array([x/np.sum(x) for x in labels_inf])
    num_tests=len(labels)
    return np.sum(-labels*np.nan_to_num(np.log(p))-
                  (1.-labels)*np.nan_to_num(np.log(1.-p)))/num_tests

def sigmoid(x):
    return 1./(1.+np.exp(-np.clip(x,-100,100)))


