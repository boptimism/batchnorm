import numpy as np
import time
import scipy.stats as sstats
import cPickle as pkl
import mysql.connector as sqlconn
import dbutils 
import get_code_ver as codever
import multiprocessing

class Baseline:
    def __init__(self,layers,learnrate,batchsize,epochs,num_trains,num_tests,
                 weights,bias,dbrec=False):
        
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
        self.dbrec=dbrec

        self.dbdict={'host':'erlichfs', 'user':'bo', 'password':'mayr2000','database':'ann'}
        if dbrec:
            self.pool[0] = sqlconn.pooling.MySQLConnectionPool(pool_name = "mypool0",
                                                      pool_size = 32,
                                                      **self.dbdict)
            self.pool[1] = sqlconn.pooling.MySQLConnectionPool(pool_name = "mypool1",
                                                      pool_size = 32,
                                                      **self.dbdict)
            self.pool[2] = sqlconn.pooling.MySQLConnectionPool(pool_name = "mypool2",
                                                      pool_size = 32,
                                                      **self.dbdict)
            self.pool[3] = sqlconn.pooling.MySQLConnectionPool(pool_name = "mypool3",
                                                      pool_size = 32,
                                                      **self.dbdict)
            self.pool[4] = sqlconn.pooling.MySQLConnectionPool(pool_name = "mypool4",
                                                      pool_size = 32,
                                                      **self.dbdict)



    
        #------------------------------------------
        # Save to DB

        if self.dbrec:
            self.con=sqlconn.connect(host='erlichfs',
                                     user='bo',password='mayr2000',
                                     database='ann')

        if dbrec:

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
                      'code_version':codever.git_version()}
            self.con=self.connect()
            dbutils.saveToDB(self.con,'ann.runs',rec_runs)
            self.runid = dbutils.lastInsertID(self.con)
     
    def connect(self):
        con = 1
        while con==1:
            try:
                con = self.pool[np.random.randint(5)].get_connection()
            except:
                time.sleep(1)

        return con
            
    def sgd(self,train_inputs,train_labels,test_inputs,test_labels,
            test_check=True,train_check=False):

        num_of_trains=len(train_labels)
        batch_per_epoch=num_of_trains/self.batchsize
        idx_epoch=np.arange(num_of_trains)
        
        for p in np.arange(self.epochs):

            # SQL--------------------------------------
            if self.dbrec:
                rec_epochs={'num':int(p),'runid':self.runid}

                dbutils.saveToDB(self.con,'ann.epochs',rec_epochs)
                epochid = dbutils.lastInsertID(self.con)

            tstart=time.clock()
            np.random.shuffle(idx_epoch)

            for q in np.arange(batch_per_epoch):

                ts=time.clock()
                
                idx_batch=idx_epoch[q*self.batchsize:(q+1)*self.batchsize]
                batch_data=train_inputs[idx_batch]
                batch_label=train_labels[idx_batch]

                self.feedforward(batch_data)

                dw,db=self.bp(batch_label)
                
                self.batch_update(dw,db)

                te=time.clock()

                if self.dbrec:
                
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
                
                    rec_batches={'num':int(q),
                                 'epochid':epochid,
                                 'runid':self.runid,
                                 'runtime':te-ts}
                    dbutils.saveToDB(self.con,'ann.minibatches',rec_batches)
                    mbid = dbutils.lastInsertID(self.con)

                    rec_params={'W_mu':pkl.dumps(w_mu),
                                'W_sig':pkl.dumps(w_sig),
                                'bias_mu':pkl.dumps(b_mu),
                                'bias_sig':pkl.dumps(b_sig),
                                'mbid':mbid}
                    dbutils.saveToDB_m(self.connect(),'ann.mbparams',rec_params)
                    
                                
                    rec_samples={'error_mu':pkl.dumps(err_mu),
                                 'error_sig':pkl.dumps(err_sig),
                                 'error_skew':pkl.dumps(err_skew),
                                 'error_kurtosis':pkl.dumps(err_kurtosis),
                                 'activation_mu':pkl.dumps(act_mu),
                                 'activation_sig':pkl.dumps(act_sig),
                                 'activation_skew':pkl.dumps(act_skew),
                                 'activation_kurtosis':pkl.dumps(act_kurtosis),
                                 'mbid':mbid}
                
                    dbutils.saveToDB_m(self.connect(),'ann.mbsamples',rec_samples)

                    rec_mbdata={'mbid':mbid,
                                'W':pkl.dumps(self.weights),
                                'bias':pkl.dumps(self.bias),
                                'error':pkl.dumps(self.deltas),
                                'activation':pkl.dumps(self.us)}
                
                    dbutils.saveToDB_m(self.connect(),'ann.mb_data',rec_mbdata)

                
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
                sqlstr = 'call update_epoch_accuracy({0},{1},{2},{3},{4},{5})'
                cur=self.con.cursor()

                cur.execute(sqlstr.format(epochid, self.train_accu[-1],
                                          self.train_cost[-1],self.test_accu[-1],
                                          self.test_cost[-1],tend-tstart))
            tend=time.clock()                
# update_epoch_accuracy`(in id int, in train_acc float, in train_loss float,in test_acc float, in test_loss float, in rt float)
        

            if self.dbrec:
                sqlstr = 'call update_epoch_accuracy({0},{1},{2},{3},{4},{5})'           
                dbutils.execute_m(self.connect(),sqlstr.format(epochid, self.train_accu[-1],self.train_cost[-1],self.test_accu[-1], self.test_cost[-1],tend-tstart))
            

            print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)
            
        if self.dbrec:
            self.con.close()
            
    def feedforward(self,batch_data):
        self.us[0]=batch_data
        for l in np.arange(1,len(self.layers)):
            self.ys[l]=np.dot(self.us[l-1],self.weights[l-1])+self.bias[l]
            if l<len(self.layers)-1:
                self.us[l]=sigmoid(self.ys[l])
            else:
                self.us[l]=softmax(self.ys[l])

    def bp(self,batch_label):
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_b=[np.zeros(b.shape) for b in self.bias]
        
        self.deltas[-1]=((self.us[-1].T*np.sum(batch_label,1)).T-batch_label)/self.batchsize
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
        us_inf=test_inputs
        for l in np.arange(1,len(self.layers)):
            ys_inf=np.dot(us_inf,self.weights[l-1])+self.bias[l]
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
    return np.sum(-labels*np.nan_to_num(np.log(labels_inf)))/len(labels)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x),1)).T
