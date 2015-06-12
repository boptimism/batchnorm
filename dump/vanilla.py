import data_loader as dl
import network as network
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__=="__main__":
    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=60000
    totaltrains=len(t_img[:,0])
    idx_train=np.random.randint(0,totaltrains,numoftrains)
    t_in=np.array(t_img[idx_train])
    t_la=np.array(t_label[idx_train])
    
    
    numoftests=10000
    totaltest=len(s_img[:,0])
    idx_test=np.random.randint(0,totaltest,numoftests)
    s_in=np.array(s_img[idx_test])
    s_la=np.array(s_label[idx_test])


    layers=[28*28,100,100,100,10]
    weights=[np.random.randn(x,y)/np.sqrt(x+1)
            for x,y in zip(layers[:-1],layers[1:])]

    idx_sets={'idx_train':idx_train.tolist(),'idx_test':idx_test.tolist()}
    ws={'weights':[x.tolist() for x in weights]}
    with open('idx_file.json','w') as fout1:
        json.dump(idx_sets,fout1)
    with open('weights.json','w') as fout2:
        json.dump(ws,fout2)
        
    learnrate=0.1
    batchsize=60
    epochs=50
    check_freq=1000
    checknum=np.arange(epochs*numoftrains/batchsize/check_freq)+1
    checks_per_epoch=numoftrains/batchsize/check_freq
    

    ann=network.Vnn(layers,learnrate,batchsize,epochs)
    ann.sgd(t_in,t_la,s_in,s_la,check=True,check_freq=check_freq)

    accu=np.array(ann.accuracy)
    print accu
    cost=np.array(ann.cost)

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(checknum,accu,'ro',checknum,accu,'b',
            checknum[checks_per_epoch-1::checks_per_epoch],
                     accu[checks_per_epoch-1::checks_per_epoch],'go')
    plt.ylabel('Accuracy')
    plt.xlabel('# of minibatches (Unit:{0})'.format(check_freq))
    plt.title('Baseline NN.')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(checknum,cost,'ro',checknum,cost,'b')
    plt.ylabel('crossEntropy loss')
    plt.grid()
    plt.savefig('./dump/nn-vanilla-check.png')
    plt.show()
                                    

    data={"number_of_trains":numoftrains,
          "number_of_tests":numoftests,
          "layers":layers,
          "learnrate":learnrate,
          "mini-batch size":batchsize,
          "epochs":epochs,
          "weights":[x.tolist() for x in ann.weights],
          "bias":[x.tolist() for x in ann.bias],
          "accuracy_epoch":accu.tolist(),
          "crossEntropy_epoch":cost.tolist(),
              }
    with open("./dump/nn-vanilla-check.json",'w') as f0:
                json.dump(data,f0)
