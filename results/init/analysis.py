import cPickle as pkl
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kstest,ks_2samp
if __name__=='__main__':

    #-----------------------------------
    # Uniform_loose

    data={}

    init1='uniform_loose'

    mt1='bl'
    mt2='bnv1'
    mt3='bnv0'

    fname='rec_'+mt1+'_'+init1+'.pickle'
    with open(fname,'rb') as frec11:
        blul=pkl.load(frec11)

    fname='rec_'+mt2+'_'+init1+'.pickle'
    with open(fname,'rb') as frec21:
        bnv1ul=pkl.load(frec21)

    fname='rec_'+mt3+'_'+init1+'.pickle'
    with open(fname,'rb') as frec31:
        bnv0ul=pkl.load(frec31)

    blul_ud=blul['u_delta_avg']
    blul_u=blul['u_avg']
    blul_d=blul['delta_avg']
    blul_dw=blul['dw']
    
    bnv1ul_ud=bnv1ul['u_delta_avg']
    bnv1ul_u=bnv1ul['u_avg']
    bnv1ul_d=bnv1ul['delta_avg']
    bnv1ul_dw=bnv1ul['dw']
    
    bnv0ul_ud=bnv0ul['u_delta_avg']
    bnv0ul_u=bnv0ul['u_avg']
    bnv0ul_d=bnv0ul['delta_avg']
    bnv0ul_dx=bnv0ul['dx_avg']
    bnv0ul_ux=bnv0ul['ux_avg']
    bnv0ul_dw=bnv0ul['dw']
    
    b_per_e=1000
    rec_freq=10
    ep=2
    ep_ins=1
    
    ids=(ep_ins-1)*b_per_e/rec_freq
    ide=ep_ins*b_per_e/rec_freq
    
    # compare the dw with uniform distr
    bl_dw_l1=np.array([x[0] for x in blul_dw[ids:ide]])
    bl_dw_l2=np.array([x[1] for x in blul_dw[ids:ide]])
    bl_dw_l3=np.array([x[2] for x in blul_dw[ids:ide]])
    bl_dw_l4=np.array([x[3] for x in blul_dw[ids:ide]])

    bnv0_dw_l1=np.array([x[0] for x in bnv0ul_dw[ids:ide]])
    bnv0_dw_l2=np.array([x[1] for x in bnv0ul_dw[ids:ide]])
    bnv0_dw_l3=np.array([x[2] for x in bnv0ul_dw[ids:ide]])
    bnv0_dw_l4=np.array([x[3] for x in bnv0ul_dw[ids:ide]])

    bnv1_dw_l1=np.array([x[0] for x in bnv1ul_dw[ids:ide]])
    bnv1_dw_l2=np.array([x[1] for x in bnv1ul_dw[ids:ide]])
    bnv1_dw_l3=np.array([x[2] for x in bnv1ul_dw[ids:ide]])
    bnv1_dw_l4=np.array([x[3] for x in bnv1ul_dw[ids:ide]])

    d1,d2,d3=bl_dw_l1.shape
    bl_dw_l1=bl_dw_l1.reshape(d1,d2*d3)
    bnv0_dw_l1=bnv0_dw_l1.reshape(d1,d2*d3)
    bnv1_dw_l1=bnv1_dw_l1.reshape(d1,d2*d3)

    d1,d2,d3=bl_dw_l2.shape
    bl_dw_l2=bl_dw_l2.reshape(d1,d2*d3)
    bnv0_dw_l2=bnv0_dw_l2.reshape(d1,d2*d3)
    bnv1_dw_l2=bnv1_dw_l2.reshape(d1,d2*d3)

    d1,d2,d3=bl_dw_l3.shape
    bl_dw_l3=bl_dw_l3.reshape(d1,d2*d3)
    bnv0_dw_l3=bnv0_dw_l3.reshape(d1,d2*d3)
    bnv1_dw_l3=bnv1_dw_l3.reshape(d1,d2*d3)

    d1,d2,d3=bl_dw_l4.shape
    bl_dw_l4=bl_dw_l4.reshape(d1,d2*d3)
    bnv0_dw_l4=bnv0_dw_l4.reshape(d1,d2*d3)
    bnv1_dw_l4=bnv1_dw_l4.reshape(d1,d2*d3)

    #-----------------------------------------------------------
    # The orthogonality between consecutive weights update 
    # bl_dw_l1_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bl_dw_l1[1:],bl_dw_l1[:-1])])
    # bl_dw_l2_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bl_dw_l2[1:],bl_dw_l2[:-1])])
    # bl_dw_l3_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bl_dw_l3[1:],bl_dw_l3[:-1])])
    # bl_dw_l4_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bl_dw_l4[1:],bl_dw_l4[:-1])])

    # bnv1_dw_l1_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv1_dw_l1[1:],bnv1_dw_l1[:-1])])
    # bnv1_dw_l2_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv1_dw_l2[1:],bnv1_dw_l2[:-1])])
    # bnv1_dw_l3_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv1_dw_l3[1:],bnv1_dw_l3[:-1])])
    # bnv1_dw_l4_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv1_dw_l4[1:],bnv1_dw_l4[:-1])])

    # bnv0_dw_l1_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv0_dw_l1[1:],bnv0_dw_l1[:-1])])
    # bnv0_dw_l2_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv0_dw_l2[1:],bnv0_dw_l2[:-1])])
    # bnv0_dw_l3_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv0_dw_l3[1:],bnv0_dw_l3[:-1])])
    # bnv0_dw_l4_cos=np.array([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    #                        for x,y in zip(bnv0_dw_l4[1:],bnv0_dw_l4[:-1])])

    # pltdata=[bl_dw_l1_cos,bl_dw_l2_cos,bl_dw_l3_cos,bl_dw_l4_cos,
    #          bnv1_dw_l1_cos,bnv1_dw_l2_cos,bnv1_dw_l3_cos,bnv1_dw_l4_cos,
    #          bnv0_dw_l1_cos,bnv0_dw_l2_cos,bnv0_dw_l3_cos,bnv0_dw_l4_cos]

    # xx=np.arange(20,1001,10)
    
    # n_rows = 3
    # n_cols = 4
    # fig, axes = plt.subplots(n_rows, n_cols, sharey = True)
    # plt.suptitle('Orthogonality between consecutive updates of W')
    # axes[0][0].set_ylim(-1.,1.)

    # for i in range(n_cols):
    #     l_label='layer-{0}'.format(i+1)
    #     #axes[0][i].text(x = 0., y = -1., s = 'num_of_batch', ha = "center")
    #     axes[n_rows-1][i].set_xlabel(l_label)

    # t_label=['BL','BN_V1','BN_V0']
    # for i in range(n_rows):
    #     #axes[i][0].text(x = -0.8, y = 0, s = t_label[i], rotation = 90, va = "center")
    #     axes[i][0].set_ylabel(t_label[i]+r": $cos[\lambda]$")

    # for n,ax in enumerate(axes.flat):
    #     ax.plot(xx,pltdata[n],'-o',ms=1.5)
    #     ax.set_yticks([-1.,0.,1.],minor=False)
    #     ax.set_yticks([-0.5,0.5],minor=True)
    #     ax.yaxis.grid(True, which='major')
    #     ax.yaxis.grid(True, which='minor')

    #     ax.set_xticks([xx[0],xx[-1]],minor=False)
    #     ax.set_xticks(xx[20:-1:20],minor=True)
    #     ax.xaxis.grid(True, which='major')
    #     ax.xaxis.grid(True, which='minor')

    # plt.show()


    #--------------------------------------------------------
    # KS test . Tenative quatifying the randomness of weights update
    # layers=[784,100,100,100,10]
    # rnd_base=[np.random.uniform(-1.,1.,[x,y]).flatten() for x,y in zip(layers[1:],layers[:-1])]
    # bl_dw_l1_ks=np.array([ks_2samp(x,rnd_base[0]) for x in bl_dw_l1])
    # bl_dw_l2_ks=np.array([ks_2samp(x,rnd_base[1]) for x in bl_dw_l2])
    # bl_dw_l3_ks=np.array([ks_2samp(x,rnd_base[2]) for x in bl_dw_l3])
    # bl_dw_l4_ks=np.array([ks_2samp(x,rnd_base[3]) for x in bl_dw_l4])
    # bnv1_dw_l1_ks=np.array([ks_2samp(x,rnd_base[0]) for x in bnv1_dw_l1])
    # bnv1_dw_l2_ks=np.array([ks_2samp(x,rnd_base[1]) for x in bnv1_dw_l2])
    # bnv1_dw_l3_ks=np.array([ks_2samp(x,rnd_base[2]) for x in bnv1_dw_l3])
    # bnv1_dw_l4_ks=np.array([ks_2samp(x,rnd_base[3]) for x in bnv1_dw_l4])
    # bnv0_dw_l1_ks=np.array([ks_2samp(x,rnd_base[0]) for x in bnv0_dw_l1])
    # bnv0_dw_l2_ks=np.array([ks_2samp(x,rnd_base[1]) for x in bnv0_dw_l2])
    # bnv0_dw_l3_ks=np.array([ks_2samp(x,rnd_base[2]) for x in bnv0_dw_l3])
    # bnv0_dw_l4_ks=np.array([ks_2samp(x,rnd_base[3]) for x in bnv0_dw_l4])

    # pltdata=[bl_dw_l1_ks,bl_dw_l2_ks,bl_dw_l3_ks,bl_dw_l4_ks,
    #          bnv1_dw_l1_ks,bnv1_dw_l2_ks,bnv1_dw_l3_ks,bnv1_dw_l4_ks,
    #          bnv0_dw_l1_ks,bnv0_dw_l2_ks,bnv0_dw_l3_ks,bnv0_dw_l4_ks]
    # plt_p=np.array([[pd[1] for pd in x] for x in pltdata])
    # plt_d=np.array([[pd[0] for pd in x] for x in pltdata])
    
    # xx=np.arange(10,1001,10)

    # n_rows = 3
    # n_cols = 4
    # fig, axes = plt.subplots(n_rows, n_cols, sharey = True)
    # plt.suptitle('d_Value udner KS_Test of W update')
    # #axes[0][0].set_ylim(0.,1.)

    # for i in range(n_cols):
    #     l_label='layer-{0}'.format(i+1)
    #     axes[0][i].text(x = 0.5, y = 12, s = l_label, ha = "center")
    #     axes[n_rows-1][i].set_xlabel(l_label)

    # t_label=['BL','BN_V1','BN_V0']
    # for i in range(n_rows):
    #     #axes[i][0].text(x = -0.8, y = 0, s = t_label[i], rotation = 90, va = "center")
    #     axes[i][0].set_ylabel(t_label[i])

    # for n,ax in enumerate(axes.flat):
    #     ax.plot(xx,plt_d[n],'-o',ms=1.5)
    #     ax.grid()
    #     # ax.set_yticks([-1.,0.,1.],minor=False)
    #     # ax.set_yticks([-0.5,0.5],minor=True)
    #     # ax.yaxis.grid(True, which='major')
    #     # ax.yaxis.grid(True, which='minor')

    #     # ax.set_xticks([xx[0],xx[-1]],minor=False)
    #     # ax.set_xticks(xx[20:-1:20],minor=True)
    #     # ax.xaxis.grid(True, which='major')
    #     # ax.xaxis.grid(True, which='minor')

    # plt.show()

    #--------------------------------
    # simple variance
    bl_dw_l1_var=np.array([np.var(x) for x in bl_dw_l1])
    bl_dw_l2_var=np.array([np.var(x) for x in bl_dw_l2])
    bl_dw_l3_var=np.array([np.var(x) for x in bl_dw_l3])
    bl_dw_l4_var=np.array([np.var(x) for x in bl_dw_l4])
    bnv1_dw_l1_var=np.array([np.var(x) for x in bnv1_dw_l1])
    bnv1_dw_l2_var=np.array([np.var(x) for x in bnv1_dw_l2])
    bnv1_dw_l3_var=np.array([np.var(x) for x in bnv1_dw_l3])
    bnv1_dw_l4_var=np.array([np.var(x) for x in bnv1_dw_l4])
    bnv0_dw_l1_var=np.array([np.var(x) for x in bnv0_dw_l1])
    bnv0_dw_l2_var=np.array([np.var(x) for x in bnv0_dw_l2])
    bnv0_dw_l3_var=np.array([np.var(x) for x in bnv0_dw_l3])
    bnv0_dw_l4_var=np.array([np.var(x) for x in bnv0_dw_l4])

    pltdata=[bl_dw_l1_var,bl_dw_l2_var,bl_dw_l3_var,bl_dw_l4_var,
             bnv1_dw_l1_var,bnv1_dw_l2_var,bnv1_dw_l3_var,bnv1_dw_l4_var,
             bnv0_dw_l1_var,bnv0_dw_l2_var,bnv0_dw_l3_var,bnv0_dw_l4_var]
    
    xx=np.arange(10,1001,10)

    n_rows = 3
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, sharey = True)
    plt.suptitle('Variance of W update')
    #axes[0][0].set_ylim(0.,1.)

    for i in range(n_cols):
        l_label='layer-{0}'.format(i+1)
        axes[0][i].text(x = 0.5, y = 12, s = l_label, ha = "center")
        axes[n_rows-1][i].set_xlabel(l_label)

    t_label=['BL','BN_V1','BN_V0']
    for i in range(n_rows):
        #axes[i][0].text(x = -0.8, y = 0, s = t_label[i], rotation = 90, va = "center")
        axes[i][0].set_ylabel(t_label[i]+' Var(dW)')

    for n,ax in enumerate(axes.flat):
        ax.plot(xx,pltdata[n],'-o',ms=1.5)
        ax.grid()
        # ax.set_yticks([-1.,0.,1.],minor=False)
        # ax.set_yticks([-0.5,0.5],minor=True)
        # ax.yaxis.grid(True, which='major')
        # ax.yaxis.grid(True, which='minor')

        ax.set_xticks([xx[0],xx[-1]],minor=False)
        ax.set_xticks(xx[20:-1:20],minor=True)
        ax.xaxis.grid(True, which='major')
        ax.xaxis.grid(True, which='minor')

    plt.show()

    
    # bl_corr_ul=[[xl-np.outer(yl,zl) for xl,yl,zl in zip(x,y,z)]
    #             for x,y,z in zip(blul_ud[ids:ide],blul_u[ids:ide],blul_d[ids:ide])]
    # bl_beer_ul=[[np.outer(yl,zl) for yl,zl in zip(y,z)]
    #             for y,z in zip(blul_u[ids:ide],blul_d[ids:ide])]

    
    # bnv1_corr_ul=[[xl-np.outer(yl,zl) for xl,yl,zl in zip(x,y,z)]
    #               for x,y,z in zip(
    #                       bnv1ul_ud[ids:ide],bnv1ul_u[ids:ide],bnv1ul_d[ids:ide])]

    # bnv0_corr_ul=[[xl-np.outer(yl,zl) for xl,yl,zl in zip(x,y,z)]
    #               for x,y,z in zip(
    #                       bnv0ul_ud[ids:ide],bnv0ul_u[ids:ide],bnv0ul_d[ids:ide])]
    # bnv0_t2t3_ul=[[yl*zl for yl,zl in zip(y,z)]
    #               for y,z in zip(bnv0ul_ux[ids:ide],bnv0ul_dx[ids:ide])]


    

    
