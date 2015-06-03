import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

if __name__=='__main__':
    with open('./uni_l/baseline_accuracy.pickle','r') as f_bl:
        bl_data=pkl.load(f_bl)
    with open('./uni_l/bnv0_accuracy.pickle','r') as f_bn_v0:
        bn_v0_data=pkl.load(f_bn_v0)
    with open('./uni_l/bnv1_accuracy.pickle','r') as f_bn_v1:
        bn_v1_data=pkl.load(f_bn_v1)

    ul_bl_test_accu=np.array(bl_data['test_accu'])
    ul_bnv0_test_accu=np.array(bn_v0_data['test_accu'])
    ul_bnv1_test_accu=np.array(bn_v1_data['test_accu'])
    ul_bl_test_loss=np.array(bl_data['test_cost'])
    ul_bnv0_test_loss=np.array(bn_v0_data['test_cost'])
    ul_bnv1_test_loss=np.array(bn_v1_data['test_cost'])

    with open('./uni_t/baseline_accuracy.pickle','r') as f_bl:
        bl_data=pkl.load(f_bl)
    with open('./uni_t/bnv0_accuracy.pickle','r') as f_bn_v0:
        bn_v0_data=pkl.load(f_bn_v0)
    with open('./uni_t/bnv1_accuracy.pickle','r') as f_bn_v1:
        bn_v1_data=pkl.load(f_bn_v1)

    ut_bl_test_accu=np.array(bl_data['test_accu'])
    ut_bnv0_test_accu=np.array(bn_v0_data['test_accu'])
    ut_bnv1_test_accu=np.array(bn_v1_data['test_accu'])
    ut_bl_test_loss=np.array(bl_data['test_cost'])
    ut_bnv0_test_loss=np.array(bn_v0_data['test_cost'])
    ut_bnv1_test_loss=np.array(bn_v1_data['test_cost'])

    
    with open('./g_l/baseline_accuracy.pickle','r') as f_bl:
        bl_data=pkl.load(f_bl)
    with open('./g_l/bnv0_accuracy.pickle','r') as f_bn_v0:
        bn_v0_data=pkl.load(f_bn_v0)
    with open('./g_l/bnv1_accuracy.pickle','r') as f_bn_v1:
        bn_v1_data=pkl.load(f_bn_v1)

    gl_bl_test_accu=np.array(bl_data['test_accu'])
    gl_bnv0_test_accu=np.array(bn_v0_data['test_accu'])
    gl_bnv1_test_accu=np.array(bn_v1_data['test_accu'])
    gl_bl_test_loss=np.array(bl_data['test_cost'])
    gl_bnv0_test_loss=np.array(bn_v0_data['test_cost'])
    gl_bnv1_test_loss=np.array(bn_v1_data['test_cost'])

    with open('./g_t/baseline_accuracy.pickle','r') as f_bl:
         bl_data=pkl.load(f_bl)
    with open('./g_t/bnv0_accuracy.pickle','r') as f_bn_v0:
        bn_v0_data=pkl.load(f_bn_v0)
    with open('./g_t/bnv1_accuracy.pickle','r') as f_bn_v1:
        bn_v1_data=pkl.load(f_bn_v1)

    gt_bl_test_accu=np.array(bl_data['test_accu'])
    gt_bnv0_test_accu=np.array(bn_v0_data['test_accu'])
    gt_bnv1_test_accu=np.array(bn_v1_data['test_accu'])
    gt_bl_test_loss=np.array(bl_data['test_cost'])
    gt_bnv0_test_loss=np.array(bn_v0_data['test_cost'])
    gt_bnv1_test_loss=np.array(bn_v1_data['test_cost'])

    ##---------------------------------------------------
    ## plots
    ##--------------------------------------------------
    ## Same model, diff config
    ## BL-------------------------
    # legends=['Uniform_Wide','Uniform_Tight','Norm_Wide','Norm_Tight']
    # fig=plt.figure(1)
    # ax=plt.subplot(111)
    # ax.set_ylim(0.,1.)
    # xx=np.arange(len(ul_bl_test_accu))
    # ax.plot(xx,ul_bl_test_accu,'m-o',
    #          xx,ut_bl_test_accu,'b-o',
    #          xx,gl_bl_test_accu,'g-o',
    #          xx,gt_bl_test_accu,'r-o',
    #          ms=2.)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Baseline Performance')
    # ax.legend(legends,loc='center right',prop={'size':10})
    # ax.grid()
    # ax.set_xticks([xx[0],xx[-1]],minor=False)
    # ax.set_xticks(xx[20:-1:20],minor=True)
    # xtk=ax.get_xticks().tolist()
    # xtk[0]=0
    # xtk[-1]=1
    # ax.set_xticklabels(xtk)
    # ax.xaxis.grid(True,which='major')
    # ax.xaxis.grid(True,which='minor')
    # plt.show()
    
    ## BN-V1------------------------------------------
    # legends=['Uniform_Wide','Uniform_Tight','Norm_Wide','Norm_Tight']
    # fig=plt.figure(2)
    # ax=plt.subplot(111)
    # ax.set_ylim(0.,1.)
    # xx=np.arange(len(ul_bl_test_accu))
    # ax.plot(xx,ul_bnv1_test_accu,'m-o',
    #          xx,ut_bnv1_test_accu,'b-o',
    #          xx,gl_bnv1_test_accu,'g-o',
    #          xx,gt_bnv1_test_accu,'r-o',
    #          ms=2.)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('BN-V1 Performance')
    # ax.legend(legends,loc='center right',prop={'size':10})
    # ax.grid()
    # ax.set_xticks([xx[0],xx[-1]],minor=False)
    # ax.set_xticks(xx[20:-1:20],minor=True)
    # xtk=ax.get_xticks().tolist()
    # xtk[0]=0
    # xtk[-1]=1
    # ax.set_xticklabels(xtk)
    # ax.xaxis.grid(True,which='major')
    # ax.xaxis.grid(True,which='minor')
    # plt.show()

    ## BN-V0------------------------------------------
    # legends=['Uniform_Wide','Uniform_Tight','Norm_Wide','Norm_Tight']
    # fig=plt.figure(3)
    # ax=plt.subplot(111)
    # ax.set_ylim(0.,1.)
    # xx=np.arange(len(ul_bl_test_accu))
    # ax.plot(xx,ul_bnv0_test_accu,'m-o',
    #          xx,ut_bnv0_test_accu,'b-o',
    #          xx,gl_bnv0_test_accu,'g-o',
    #          xx,gt_bnv0_test_accu,'r-o',
    #          ms=3.)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('BN-V0 Performance')
    # ax.legend(legends,loc='center right',prop={'size':10})
    # ax.grid()
    # ax.set_xticks([xx[0],xx[-1]],minor=False)
    # ax.set_xticks(xx[20:-1:20],minor=True)
    # xtk=ax.get_xticks().tolist()
    # xtk[0]=0
    # xtk[-1]=1
    # ax.set_xticklabels(xtk)
    # ax.xaxis.grid(True,which='major')
    # ax.xaxis.grid(True,which='minor')
    # plt.show()

    ##-------------------------------
    ## Same init, diff model
    ## Uni_Wide
    # legends=['BL','BN-V1','BN-V0']
    # fig=plt.figure(4)
    # ax=plt.subplot(111)
    # ax.set_ylim(0.,1.)
    # xx=np.arange(len(ul_bl_test_accu))
    # ax.plot(xx,ul_bl_test_accu,'r-o',
    #          xx,ul_bnv1_test_accu,'g-o',
    #          xx,ul_bnv0_test_accu,'b-o',
    #          ms=3.)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Uniform-Wide')
    # ax.legend(legends,loc='center right',prop={'size':10})
    # ax.grid()
    # ax.set_xticks([xx[0],xx[-1]],minor=False)
    # ax.set_xticks(xx[20:-1:20],minor=True)
    # xtk=ax.get_xticks().tolist()
    # xtk[0]=0
    # xtk[-1]=1
    # ax.set_xticklabels(xtk)
    # ax.xaxis.grid(True,which='major')
    # ax.xaxis.grid(True,which='minor')
    # plt.show()

    ## Uni-tight
    # legends=['BL','BN-V1','BN-V0']
    # fig=plt.figure(5)
    # ax=plt.subplot(111)
    # ax.set_ylim(0.,1.)
    # xx=np.arange(len(ul_bl_test_accu))
    # ax.plot(xx,ut_bl_test_accu,'r-o',
    #          xx,ut_bnv1_test_accu,'g-o',
    #          xx,ut_bnv0_test_accu,'b-o',
    #          ms=3.)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Uniform-Tight')
    # ax.legend(legends,loc='center right',prop={'size':10})
    # ax.grid()
    # ax.set_xticks([xx[0],xx[-1]],minor=False)
    # ax.set_xticks(xx[20:-1:20],minor=True)
    # xtk=ax.get_xticks().tolist()
    # xtk[0]=0
    # xtk[-1]=1
    # ax.set_xticklabels(xtk)
    # ax.xaxis.grid(True,which='major')
    # ax.xaxis.grid(True,which='minor')
    # plt.show()
    
    ## Norm-Wide
    # legends=['BL','BN-V1','BN-V0']
    # fig=plt.figure(6)
    # ax=plt.subplot(111)
    # ax.set_ylim(0.,1.)
    # xx=np.arange(len(ul_bl_test_accu))
    # ax.plot(xx,gl_bl_test_accu,'r-o',
    #          xx,gl_bnv1_test_accu,'g-o',
    #          xx,gl_bnv0_test_accu,'b-o',
    #          ms=3.)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Norm-Wide')
    # ax.legend(legends,loc='center right',prop={'size':10})
    # ax.grid()
    # ax.set_xticks([xx[0],xx[-1]],minor=False)
    # ax.set_xticks(xx[20:-1:20],minor=True)
    # xtk=ax.get_xticks().tolist()
    # xtk[0]=0
    # xtk[-1]=1
    # ax.set_xticklabels(xtk)
    # ax.xaxis.grid(True,which='major')
    # ax.xaxis.grid(True,which='minor')
    # plt.show()

    ## Norm-tight
    legends=['BL','BN-V1','BN-V0']
    fig=plt.figure(6)
    ax=plt.subplot(111)
    ax.set_ylim(0.,1.)
    xx=np.arange(len(ul_bl_test_accu))
    ax.plot(xx,gt_bl_test_accu,'r-o',
             xx,gt_bnv1_test_accu,'g-o',
             xx,gt_bnv0_test_accu,'b-o',
             ms=3.)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Norm-Tight')
    ax.legend(legends,loc='center right',prop={'size':10})
    ax.grid()
    ax.set_xticks([xx[0],xx[-1]],minor=False)
    ax.set_xticks(xx[20:-1:20],minor=True)
    xtk=ax.get_xticks().tolist()
    xtk[0]=0
    xtk[-1]=1
    ax.set_xticklabels(xtk)
    ax.xaxis.grid(True,which='major')
    ax.xaxis.grid(True,which='minor')
    plt.show()
    
    # bl_train_accu=np.array(bl_data['train_accu'])
    # bnv0_train_accu=np.array(bn_v0_data['train_accu'])
    # bnv1_train_accu=np.array(bn_v1_data['train_accu'])

    # bl_train_loss=np.array(bl_data['train_cost'])
    # bnv0_train_loss=np.array(bn_v0_data['train_cost'])
    # bnv1_train_loss=np.array(bn_v1_data['train_cost'])

    # print "Baseline: ",bl_data['learnrate']
    # print "Baseline: ",bn_v0_data['learnrate']
    # print "Baseline: ",bn_v1_data['learnrate']

    # lrate=2.0
    # lrate_decay=0.002
    # fig=plt.figure()
    
    # plt.suptitle('learning decay:{0}/(1+{1}*t)'.format(lrate,lrate_decay))
    # plt.subplot(221)
    # plt.plot(bl_test_accu,'g-o',bnv0_test_accu,'r-o',bnv1_test_accu,'b-o')
    # plt.legend(['Baseline','BN','BN-V1'],loc='lower right')
    # plt.grid()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Test')
    
    # plt.subplot(222)
    # plt.plot(bl_test_loss,'g-o',bnv0_test_loss,'r-o',bnv1_test_loss,'b-o')
    # plt.legend(['Baseline','BN','BN-v1'],loc='upper right')
    # plt.grid()
    # plt.ylabel('CrossEntropy Loss')
    # plt.xlabel('Epochs')
    # plt.title('Test')

    # plt.subplot(223)
    # plt.plot(bl_train_accu,'g-o',bnv0_train_accu,'r-o',bnv1_train_accu,'b-o')
    # plt.legend(['Baseline','BN','BN-V1'],loc='lower right')
    # plt.grid()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Train')
    
    # plt.subplot(224)
    # plt.plot(bl_train_loss,'g-o',bnv0_train_loss,'r-o',bnv1_train_loss,'b-o')
    # plt.legend(['Baseline','BN','BN-v1'],loc='upper right')
    # plt.grid()
    # plt.ylabel('CrossEntropy Loss')
    # plt.xlabel('Epochs')
    # plt.title('Train')

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

    
    #plt.show()
    
