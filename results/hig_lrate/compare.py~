import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

if __name__=='__main__':
    with open('baseline_accuracy.pickle','r') as f_bl:
        bl_data=pkl.load(f_bl)
    with open('bnv0_accuracy.pickle','r') as f_bn_v0:
        bn_v0_data=pkl.load(f_bn_v0)
    with open('bnv1_accuracy.pickle','r') as f_bn_v1:
        bn_v1_data=pkl.load(f_bn_v1)

    bl_test_accu=np.array(bl_data['test_accu'])
    bnv0_test_accu=np.array(bn_v0_data['test_accu'])
    bnv1_test_accu=np.array(bn_v1_data['test_accu'])

    bl_test_loss=np.array(bl_data['test_cost'])
    bnv0_test_loss=np.array(bn_v0_data['test_cost'])
    bnv1_test_loss=np.array(bn_v1_data['test_cost'])

    bl_train_accu=np.array(bl_data['train_accu'])
    bnv0_train_accu=np.array(bn_v0_data['train_accu'])
    bnv1_train_accu=np.array(bn_v1_data['train_accu'])

    bl_train_loss=np.array(bl_data['train_cost'])
    bnv0_train_loss=np.array(bn_v0_data['train_cost'])
    bnv1_train_loss=np.array(bn_v1_data['train_cost'])

    # print "Baseline: ",bl_data['learnrate']
    # print "Baseline: ",bn_v0_data['learnrate']
    # print "Baseline: ",bn_v1_data['learnrate']

    lrate=2.0
    lrate_decay=0.002
    fig=plt.figure()
    
    plt.suptitle('learning decay:{0}/(1+{1}*t)'.format(lrate,lrate_decay))
    plt.subplot(221)
    plt.plot(bl_test_accu,'g-o',bnv0_test_accu,'r-o',bnv1_test_accu,'b-o')
    plt.legend(['Baseline','BN','BN-V1'],loc='lower right')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test')
    
    plt.subplot(222)
    plt.plot(bl_test_loss,'g-o',bnv0_test_loss,'r-o',bnv1_test_loss,'b-o')
    plt.legend(['Baseline','BN','BN-v1'],loc='upper right')
    plt.grid()
    plt.ylabel('CrossEntropy Loss')
    plt.xlabel('Epochs')
    plt.title('Test')

    plt.subplot(223)
    plt.plot(bl_train_accu,'g-o',bnv0_train_accu,'r-o',bnv1_train_accu,'b-o')
    plt.legend(['Baseline','BN','BN-V1'],loc='lower right')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train')
    
    plt.subplot(224)
    plt.plot(bl_train_loss,'g-o',bnv0_train_loss,'r-o',bnv1_train_loss,'b-o')
    plt.legend(['Baseline','BN','BN-v1'],loc='upper right')
    plt.grid()
    plt.ylabel('CrossEntropy Loss')
    plt.xlabel('Epochs')
    plt.title('Train')

    # plt.subplot(223)
    # plt.plot(bnv0accu,'r-o',bnv1accu,'b-o')
    # plt.legend(['BN','BN-V1'],loc=4)
    # plt.grid()
    # plt.xlabel('Epochs')
    # plt.title('Accuracy')
    
    # plt.subplot(224)
    # plt.plot(bnv0loss,'r-o',bnv1loss,'b-o')
    # plt.legend(['BN','BN-v1'])
    # plt.grid()
    # plt.title('CrossEntropy Loss')
    # plt.xlabel('Epochs')

    # with open('beerbet.pickle','rb') as fin:
    #     data=pkl.load(fin)
    # avgu=data[0]
    # avgdelta=data[1]
    # l_prev=-3
    # l_post=-2
    # idi=5
    # idj=4

    # pavgu=np.array([ut[l_prev][idj] for ut in avgu])/50.
    # pavgdelta=np.array([deltat[l_post][idi] for deltat in avgdelta])/50.

    # ns=0
    # ne=len(pavgu)
    # dn=10

    # plotu=pavgu[ns:ne:dn]
    # plotd=pavgdelta[ns:ne:dn]
    
    # fig=plt.figure()
    # ax=fig.add_subplot(211)
    # ax.plot(plotu,'r-o')
    # ax.grid()
    # ax.set_title("<u>,layer {0},node {1}".format(l_prev,idj))
    # ax=fig.add_subplot(212)
    # ax.plot(plotd,'r-o')
    # ax.grid()
    # ax.set_title(r'<$\Delta$>,layer {0},node {1}'.format(l_post,idi))

    
    plt.show()
    
