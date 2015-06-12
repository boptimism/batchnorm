Aimport numpy as np
from funcs import *

class Layer(object):

    def __init__(self,num_of_nodes,actfn,prepfn):
        self._layertype=prepfn
        self._dim=num_of_nodes
        self._actFn=actFn[actfn]
        self._dactFn=dactFn[actfn]
        self._prepFn=prepFn[prepfn]

        self.bias=np.random.uniform(-1.,1.,len(self._dim))
        self.gammas=np.ones(len(self._dim))

        self.deltas=np.array([])
        self.xhats=np.array([])
        self.ys=np.array([])
        self.us=np.array([])
        
    def activate(self):
        self.us=self._actFn(self.ys)

    def dudy(self):
        return self._dactFn(self.ys)

    def preprocess(self,inputs):
        self.xhats=self._prepFn(inputs)
        self.ys=self.gammas*self.xhats+self.bias

class InputLayer(Layer):
    def __init__(self,num_of_nodes,actfn='pass',prepfn='pass'):
        super(InputLayer,self).__init__(num_of_nodes,actfn,prepfn)
        self.bias=np.zeros(num_of_nodes)
    
class Network(object):
    def __init__(self,lnodes,lossfn,lactfn,lprepfn):
        self._types=lactfn
        self._lossFn=lossFn[lossfn]
        self._dlossFn=dlossFn[lossfn]
        self.layers=[Layer(n,afn,pfn) for n,afn,pfn in
                zip(lnodes[1:],lactfn,lprepfn)]
        self.layers.insert(0,InputLayer(lnodes[0]))
        self.weights=[np.random.uniform(-1.,1.,[x,y])
                      for x,y in zip(lnodes[:-1],lnodes[1:])]
        self.backprop={}
        
        
    def forward(self,inputs):
        self.layer[0].preprocess(inputs)
        self.layer[0].activate()
        l_outs=layer[0].us
        
        for i,layer in enumerate(self.layers[1:]):
            l_ins=np.dot(self.weights[i],l_outs)
            layer.preprocess(l_ins)
            layer.activate()
            l_outs=layer.us
    
    def backward(self,labels):
        dw=[]
        db=[]
        dg=[]
        if self.types[-1]!='softmax':
            dudy=self.layers[-1].dudy()
            self.layers[-1].deltas=dudy*self._dlossFn(labels,self.layers[-1].us)
        elif self.types[-1]=='softmax':
            dudy=self.layers[-1].dudy()
            dcdu=self._dlossFn(labels,self.layers[-1].us)
            self.layers[-1].deltas=np.array([np.dot(x,y)
                                             for x,y in zip(dcdu,dudy)])
        for l0,l1 in zip(reversed(self.layers[:-1]),reversed(self.layers[1:])):
            if l1._layertype==''
            dw,db,dg=self.backprop[l1._layertype](l0,l1)

    

            
    def bp_baseline(self,l0,l1):
        
    def bp_sample_norm(self):

    def bp_sample_center(self):

    def bp_layer_norm(self):

    def bp_layer_center(self):
            
        
