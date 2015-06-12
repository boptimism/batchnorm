import numpy as np
import cPickle as pkl
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

with open('insights.pickle','rb') as fin:
    data=pkl.load(fin)
gammas=data[0]
stderr=data[1]
stdu=data[2]
stdx=data[3]
t3cos=data[4]
t1cos=data[5]
t2cos=data[6]
wij=data[7]

# Angles
#-----------------------
# ns=0
# ne=wij.shape[-1]
# dn=25
# xx=np.arange(ne)
# x=xx[ns:ne:dn]

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(x,np.arccos(t3cos[ns:ne:dn])/np.pi,'r-o',
#          x,np.arccos(t1cos[ns:ne:dn])/np.pi,'b-o',
#          x,np.arccos(t2cos[ns:ne:dn])/np.pi,'g-o')
# y_tick=np.arange(0.,1.01,0.25)
# y_label=[r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$']
# ax.set_yticks(y_tick)
# ax.set_yticklabels(y_label,fontsize=20)

# plt.legend([r'$\theta_3(\hat{\Delta},\hat{u})$',
#             r'$\theta_1(\hat{\Delta},\hat{x})$',
#             r'$\theta_2(\hat{x},\hat{u})$'])
# plt.xlabel('Iter')
# plt.title(r'Angles between $\hat{u}$,$\hat{\Delta}$,$\hat{x}$')
# plt.grid()
# plt.show()
#-------------------------

# STDs
#-------------------------
# ns=0
# ne=wij.shape[-1]
# dn=25
# xx=np.arange(ne)
# x=xx[ns:ne:dn]

# fig=plt.figure()
# ax=fig.add_subplot(311)
# ax.plot(x,stderr[ns:ne:dn],'-o')
# plt.grid()
# plt.title(r'$\sigma(\Delta)$')

# ax=fig.add_subplot(312)
# ax.plot(x,stdu[ns:ne:dn],'-o')
# plt.grid()
# plt.title(r'$\sigma(u)$')

# ax=fig.add_subplot(313)
# ax.plot(x,stdx[ns:ne:dn],'-o')
# plt.grid()
# plt.title(r'$\sigma(x)$')

# plt.show()
#-------------------------------


# Gamma
#-------------------------------
ns=0
ne=wij.shape[-1]
dn=25
xx=np.arange(ne)
x=xx[ns:ne:dn]

fig=plt.figure()
ax=fig.add_subplot(111)
#xnew=np.delete(x,[0])
#adjgamma=np.delete(50.*stderr[ns:ne:dn]*t1cos[ns:ne:dn],[0])
ax.plot(x,gammas[ns:ne:dn]/stdx[ns:ne:dn],'-o')
plt.xlabel('iteration')
plt.ylabel(r'$\gamma/\sigma(x)$')
plt.grid()

# ax=fig.add_subplot(312)
# ax.plot(x,stderr[ns:ne:dn],'-o')
# plt.title(r'$\sigma(\Delta)$')
# plt.grid()

# ax=fig.add_subplot(313)
# ax.plot(x,t1cos[ns:ne:dn],'-o')
# plt.title(r'$cos(\theta_1)$')
# plt.grid()

plt.show()
#-------------------------------


# Stats for BN - exmaine the moments
# def fitfunc(x,a,b,c):
#     return a*np.log(x+b)+c

# with open('baseline_weights.pickle','rb') as fin:
#     data=pkl.load(fin)
# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# blw=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)
#             blw.append(data[key])
# wbl=np.zeros(len(blw[0])).tolist()
# for i in np.arange(len(wbl)):
#     wbl[i]=np.array([x[i].flatten() for x in blw])
    
# with open('batchnorm_weights.pickle','rb') as fin:
#     data=pkl.load(fin)
# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# bnw=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)
#             bnw.append(data[key])
# wbn=np.zeros(len(bnw[0])).tolist()
# for i in np.arange(len(wbn)):
#     wbn[i]=np.array([x[i].flatten() for x in bnw])

# num_train=len(wbl[0][:,0])

# #-----------------------------------------------------------
# # Check different W
# layer=0
# trange=np.arange(num_train)+1

# fig=plt.figure()
# fig.suptitle('stats of weights linking layer {0},{1}'.format(layer,layer+1))
# ax=fig.add_subplot(221)
# bl_m1=np.mean(wbl[layer],1)
# bn_m1=np.mean(wbn[layer],1)
# ax.plot(trange,bl_m1,'r-o',trange,bn_m1,'b-o')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('mean of weights')
# #ax.set_title('W(L{0},L{1})'.format(layer,layer+1))
# ax.legend(['baseline','batchnorm'],loc=7)
# ax.grid()

# ax=fig.add_subplot(222)
# bl_m2=np.std(wbl[layer],1)
# bn_m2=np.std(wbn[layer],1)

# para0=np.array([1.,0.,0.])
# paras=opt.curve_fit(fitfunc,trange,bn_m2,para0)
# pred=fitfunc(trange,paras[0][0],paras[0][1],paras[0][2])
# print paras[0]

# ax.plot(trange,bl_m2,'r-o',trange,bn_m2,'b-o',trange,pred,'k-',lw=3.)
# ax.set_xlabel('Iteration')
# ax.set_ylabel('std of weights')
# #ax.set_title('W(L{0},L{1})'.format(layer,layer+1))
# ax.legend(['baseline','batchnorm'],loc=4)
# ax.annotate('{0:.3f}*log(x+{1:.3f})+{2:.3f}'.format(paras[0][0],paras[0][1],paras[0][2]),
#             xy=(trange[35],pred[35]),xytext=(trange[25]-3,pred[25]-0.1),
#             arrowprops=dict(facecolor='black',shrink=0.05))
# ax.grid()

# ax=fig.add_subplot(223)
# bl_m3=stats.skew(wbl[layer],1)
# bn_m3=stats.skew(wbn[layer],1)
# ax.plot(trange,bl_m3,'r-o',trange,bn_m3,'b-o')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('skew of weights')
# #ax.set_title('W(L{0},L{1})'.format(layer,layer+1))
# ax.legend(['baseline','batchnorm'],loc=4)
# ax.grid()

# ax=fig.add_subplot(224)
# bl_m4=stats.kurtosis(wbl[layer],1)
# bn_m4=stats.kurtosis(wbn[layer],1)
# ax.plot(trange,bl_m4,'r-o',trange,bn_m4,'b-o')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('kurtosis of weights')
# #ax.set_title('W(L{0},L{1})'.format(layer,layer+1))
# ax.legend(['baseline','batchnorm'],loc=1)
# ax.grid()

# fig2=plt.figure()
# ax=fig2.add_subplot(211)
# plotrange=np.arange(0,len(wbl[3][0,:]),10)
# for i in plotrange:
#     ax.plot(trange,wbl[3][:,i],'-o',ms=2.)
# ax.set_xlabel('iteration')
# ax.set_ylabel('weights of last layer')
# ax.set_title('BaseLine')
# ax.grid()

# ax=fig2.add_subplot(212)
# plotrange=np.arange(0,len(wbn[3][0,:]),10)
# for i in plotrange:
#     ax.plot(trange,wbn[3][:,i],'-o',ms=2.)
# ax.set_xlabel('iteration')
# ax.set_ylabel('weights of last layer')
# ax.set_title('BatchNorm')
# ax.grid()

#plt.show()















# figW1=plt.figure()
# ax=figW1.add_subplot(111,projection='3d')
# trange=np.arange(num_train)
# nrange=np.arange(1,num_weights,5)
# X,Y=np.meshgrid(trange,nrange)
# data=blw1[trange,:][:,nrange]
# #ax.plot_wireframe(X,Y,blw1.T,rstride=10,cstride=10)
# for x,y,z in zip(X,Y,data.T):
#     ax.plot(x,y,z,'-o',ms=2.)
# ax.set_ylabel('weights')
# ax.set_xlabel('iteration')
# ax.set_title('Baseline weights layer [L-1,L]')
#BN

# num_train=len(bnw1[:,0])
# num_weights=len(bnw1[0,:])

# ax=fig.add_subplot(212)
# trange=np.arange(num_train)
# wrange=np.arange(1,num_weights,5)
# for y in bnw1[:,wrange].T:
#     ax.plot(trange,y,'-o',ms=2.)
# ax.set_xlabel('Iteration')
# ax.set_ylabel('weights')
# ax.set_title('batchnorm weights between last 2 layers')
# ax.grid()
# num_train=len(bnw1[:,0])
# num_weights=len(bnw1[0,:])
# figW2=plt.figure()
# ax=figW2.add_subplot(111,projection='3d')
# trange=np.arange(num_train)
# nrange=np.arange(1,num_weights,5)
# X,Y=np.meshgrid(trange,nrange)
# data=bnw1[trange,:][:,nrange]
# #ax.plot_wireframe(X,Y,bnw1.T,rstride=10,cstride=10)
# for x,y,z in zip(X,Y,data.T):
#     ax.plot(x,y,z,'-o',ms=2.)
# ax.set_ylabel('weights')
# ax.set_xlabel('iteration')
# ax.set_title('Batchnorm weights layer[L-2,L-1]')


# fig1=plt.figure()
# plt.suptitle('BaseLine')
# ax=plt.subplot(121)
# im=plt.imshow(blw1.T,cmap='hot',origin='lower')
# plt.colorbar(im,ax=ax)
# ax.set_ylabel('weights [L-1,L]')
# ax.set_xlabel('iterations')
# plt.grid()

# ax=plt.subplot(122)
# im=plt.imshow(blw2.T,cmap='hot',origin='lower')
# plt.colorbar(im,ax=ax)
# ax.set_ylabel('weights [L-2,L-1]')
# ax.set_xlabel('iterations')
# plt.grid()

#---------------------
# BN
#---------------------

# with open('batchnorm_weights.pickle','rb') as fin:
#     data=pkl.load(fin)
# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# bnw1=[]
# bnw2=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)+'hl-1'
#             bnw1.append(data[key])
#             key='p'+str(p)+'q'+str(q)+'hl-2'
#             bnw2.append(data[key])
# bnw1=np.array([x.flatten() for x in bnw1])
# bnw2=np.array([x.flatten() for x in bnw2])

# num_train=len(blw1[:,0])
# num_nodes=len(blw1[0,:])

# fig2=plt.figure()
# plt.suptitle('BatchNorm')
# ax=plt.subplot(121)
# im=plt.imshow(bnw1.T,cmap='hot',origin='lower')
# plt.colorbar(im,ax=ax)
# ax.set_ylabel('weights [L-1,L]')
# ax.set_xlabel('iterations')
# plt.grid()

# ax=plt.subplot(122)
# im=plt.imshow(bnw2.T,cmap='hot',origin='lower')
# plt.colorbar(im,ax=ax)
# ax.set_ylabel('weights [L-2,L-1]')
# ax.set_xlabel('iterations')
# plt.grid()


#plt.show()

# x=np.arange(num_train)+1
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,blw1.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('Same Init. Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))


# with open('batchnorm_weights.pickle','rb') as fin:
#     data=pkl.load(fin)
# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# bnw1=[]
# bnw2=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)+'hl-1'
#             bnw1.append(data[key])
#             key='p'+str(p)+'q'+str(q)+'hl-2'
#             bnw2.append(data[key])



            
# with open('vanilla-dist.pickle','rb') as fin:
#     data=pkl.load(fin)

# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# layer_1=[]
# layer_2=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)+'hl-1'
#             layer_1.append(data[key])
#             key='p'+str(p)+'q'+str(q)+'hl-2'
#             layer_2.append(data[key])

# h_1=np.array(layer_1)
# h_2=np.array(layer_2)
# num_train=len(h_1[:,0,0])
# num_sample=len(h_1[0,:,0])
# num_nodes=len(h_1[0,0,:])

# Temporal evolution of correlation between 1 node in last layer
# to all the nodes in the 2nd-to-last layer
# thenode=1
# corr_layernode=[]
# for nt in np.arange(num_train):
#     xdata=h_1[nt,:,thenode-1]
#     qq_reg_layernode=[stats.linregress(xdata,h_2[nt,:,y]) for y in np.arange(num_nodes)]
#     corr_layernode.append(qq_reg_layernode)
# corr_layernode=np.array(corr_layernode)
# r_bl=corr_layernode[:,:,2]
# stderr=corr_layernode[:,:-1]

# figc=plt.figure()
# x=np.arange(num_train)+1
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r_bl.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('Same Init. Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))

# rdata=r2[10,:]
# p_rdata=np.sort(rdata[rdata>0.])
# n_rdata=np.sort(rdata[rdata<=0.])
# rdata=np.sort(rdata)
# print len(p_rdata),len(n_rdata)
# figt_bl=plt.figure()
# plt.suptitle('Correlations of Inputs to Sigmoid between Node[-1,1] to Node[-2,:]')
# prange=np.arange(5,21,3)
# plegend=[]
# for i in prange:
#     plegend.append('Iter={0}'.format(i))

# plt.subplot(2,2,1)
# for i in prange:
#     plt.plot(np.sort(r_bl[i,:]))
# plt.legend(plegend,loc='upper left')
# plt.grid()
# plt.xlabel('Nodes')
# plt.ylabel('r')
# plt.title('Baseline')
# Temporal evolution of correlation of input distribution of last hidden layer.
# The correlation is computed between 2 successive recordings.
# corr_input=[]
# for nt in np.arange(num_train-1):
#     qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
#     corr_input.append(qq_node_corr)
# corr_input=np.array(corr_input)
# r2=corr_input[:,:,2]
# stderr_nodes=corr_input[:,:,-1]

# figc2=plt.figure()
# x=np.arange(num_train-1)+2
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral)
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('Same Init. Input (HL[-1]) Correlation for successive recordings')


# #-----------------------------------------------------------------------
# # BN
# #-----------------------------------------------------------------------
# with open('bn-dist.pickle','rb') as fin:
#     data=pkl.load(fin)

# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# layer_1=[]
# layer_2=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)+'hl-1'
#             layer_1.append(data[key])
#             key='p'+str(p)+'q'+str(q)+'hl-2'
#             layer_2.append(data[key])

# h_1=np.array(layer_1)
# h_2=np.array(layer_2)
# num_train=len(h_1[:,0,0])
# num_sample=len(h_1[0,:,0])
# num_nodes=len(h_1[0,0,:])

# Temporal evolution of correlation between 1 node in last layer
# to all the nodes in the 2nd-to-last layer
# thenode=1
# corr_layernode=[]
# for nt in np.arange(num_train):
#     xdata=h_1[nt,:,thenode-1]
#     qq_reg_layernode=[stats.linregress(xdata,h_2[nt,:,y]) for y in np.arange(num_nodes)]
#     corr_layernode.append(qq_reg_layernode)
# corr_layernode=np.array(corr_layernode)
# r_bn=corr_layernode[:,:,2]
# stderr=corr_layernode[:,:-1]

# figc3=plt.figure()
# x=np.arange(num_train)+1
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r_bn.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('BN(Same Init): Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))

# plt.subplot(2,2,2)
# for i in prange:
#     plt.plot(np.sort(r_bn[i,:]))
# plt.legend(plegend,loc='upper left')
# plt.grid()
# plt.xlabel('Nodes')
# plt.ylabel('r')
# plt.title('BN')

# chkpnt1=5
# chkpnt2=10
# plt.subplot(2,2,3)
# plt.plot(np.sort(r_bl[chkpnt1-1,:]),'ro-',
#          np.sort(r_bn[chkpnt1-1,:]),'bo-')
# plt.xlabel('Sorted Nodes')
# plt.ylabel('r')
# plt.grid()
# plt.legend(['Baseline','BN'],loc='upper left')
# plt.title('{0} batches'.format(chkpnt1))

# plt.subplot(2,2,4)
# plt.plot(np.sort(r_bl[chkpnt2-1,:]),'ro-',
#          np.sort(r_bn[chkpnt2-1,:]),'bo-')
# plt.xlabel('Sorted Nodes')
# plt.ylabel('r')
# plt.grid()
# plt.legend(['Baseline','BN'],loc='upper left')
# plt.title('{0} batches'.format(chkpnt2))

# # Temporal evolution of correlation of input distribution of last hidden layer.
# # The correlation is computed between 2 successive recordings.
# corr_input=[]
# for nt in np.arange(num_train-1):
#     qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
#     corr_input.append(qq_node_corr)
# corr_input=np.array(corr_input)
# r2=corr_input[:,:,2]**2
# stderr_nodes=corr_input[:,:,-1]

# figc4=plt.figure()
# x=np.arange(num_train-1)+2
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('BN(Same Init): Input (HL[-1]) Correlation for successive recordings')
# plt.show()


#plt.show()

# figc5=plt.figure()

# plt.subplot(131)
# y=np.arange(numrec)+1
# x=np.arange(numnodeshl1)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,hl1_adj,cmap=mpl.cm.spectral,edgecolors='k')
# #ax.set_ylabel('Iteration') 
# #ax.set_xlabel('Nodes')
# plt.title('HL1')
# plt.colorbar(orientation='horizontal')
# #plt.show()

# plt.subplot(132)
# y=np.arange(numrec)+1
# x=np.arange(numnodeshl2)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,hl2_adj,cmap=mpl.cm.spectral,edgecolors='k')
# #plt.colorbar(orientation='horizontal')
# #ax.set_ylabel('Iteration') 
# #ax.set_xlabel('Nodes')
# plt.title('HL2')
# plt.colorbar(orientation='horizontal')

# plt.subplot(133)
# y=np.arange(numrec)+1
# x=np.arange(numnodeshl3)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,hl3_adj,cmap=mpl.cm.spectral,edgecolors='k')
# #plt.colorbar(orientation='horizontal')
# #ax.ylabel('Iteration') 
# #ax.set_xlabel('Nodes')
# plt.title('HL3')
# plt.colorbar(orientation='horizontal')
# plt.show()





