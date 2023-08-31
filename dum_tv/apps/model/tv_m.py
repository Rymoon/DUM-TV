
# Will import:
# * xxx_h
# * xxx_opr
# Contents:
# * Deep unfolding networks

import pytorch_lightning as pl



    
# Model
from torch.nn import Module
from .tv_opr import *
from typing import List, Tuple,Optional,Dict
import torch
from torch import Tensor
from torch.nn import ParameterDict,ParameterList,ModuleList
from ren_utils.rennet import BufferDict
from collections import namedtuple
class TVNet(Module):
    """
    #NOTICE:TVNet: Currently, control_model not work; Always None.
    controled_init = [(controled_name, init_tensor)]
    controle_model.forward -> change_tensor
    controled_args = controled_init + change_tensor
    
    If an arg in controled_init freezed, then:
    * self.contorled_init[idx_layer][name_arg].requires_grad(False)
    
    If an arg in controled_init itself a leanabel param, then
    * control_model.forward(...)[name_arg] is a detached zero.
    
    
    If the control_model is None, then
    * args in controled_init will go along.
    """
    controled_names = ("kerK","beta","rho")# type: Tuple[str,...]
    def __init__(self,patch_shape:Tuple[int,int,int],n_steps:int,control_model:Module,controled_init:List[Tuple[str,Tensor]]):
        """
        patch_shape: CHW
        """
        # Free from variational model
        super().__init__()
        self.patch_shape = patch_shape
        self.n_steps = n_steps
        self.control_model = control_model 
        
        assert self.controled_names == tuple(i[0] for i in controled_init), f"Order and names should match:{self.controled_names} neq {tuple(i[0] for i in controled_init)}"
        
        self.controled_init = ParameterList([ParameterDict({k:v for k,v in controled_init}) for i in range(n_steps)])
        
        # Related to variational model
        ### Maintained through layers
        # u (B,C,H,W)
        # v (B,C,H,W)
        # p (B,C*kO,H,W)
        # lamh (B,C*kO,H,W)
        # repeat B later in function `step`
        C,H,W = self.patch_shape
        kerK = controled_init[self.controled_names.index("kerK")][1]
        kO= kerK.shape[0]//C
        assert (kerK.shape[-1]-1)%2 == 0,kerK.shape
        kR = (kerK.shape[-1]-1)//2
        self.state_init = BufferDict({
            "u": torch.zeros((1,C,H,W)),
            "p": torch.zeros((1,C*kO,H,W)),
            "mu": torch.zeros((1,C*kO,H,W)),
        },persistent=False)
        
        _Constants = namedtuple(f"{type(self).__name__}::_Constants","kO kR C H W")
        self.constants = _Constants(kO,kR,C,H,W)
        
    def repeat_state_var(self,v:torch.Tensor,batch_size:int):
        """
        Copy the tensor.
        """
        assert len(v.shape) == 4
        
        if  v.shape[0]!= batch_size:  
            if v.shape[0] == 1:
                v=v.repeat(batch_size,1,1,1)
            else:
                raise Exception(f"Try to {v.shape}-->dim0:{batch_size}")
            
        return v
    
    
    def step(self,f,u,p,mu,*,kerK,gamma,rho):
        """
        * arg[0]:     input:f
        * arg[1-3]:   state u,p,mu
        * karg:       kerK,gamma,rho 
        
        * f:1,C,H,W: 
            input, (y)
        * u:B,C,H,W: 
            output (x)
        * p:B,C*kO,H,W
            auxilliary, p ~= Dx
        - lamh:B,C*kO,H,W
            multiplier
        * mu = lamh/rho
        * kerK:kO,kI=1,kR,kR
            Kernel, TV
            
        * rho:1,C*kO,1,1
            like lamh
            
        - beta:1,C*kO,1,1
        * gamma: beta/rho
         
        """
        B,C,H,W = f.shape
        kO = self.constants.kO
        kR = self.constants.kR
        
        assert H == self.constants.H
        assert W == self.constants.W
        
        # Subsolver (u)
        #rho D^T
        kerW = rho* kerK.flip((-1,-2))
        # u = inv(kerH)*_w1
        _w1 = f + CKtoC(conv2c_parallel(kerW,(p-mu)))
        
        ktk  = squeezek(conv2k(kerK,kerK))*rho # C*kO,1,...
        sktk = ktk.reshape(C,kO,1,ktk.shape[-2],ktk.shape[-1]).sum(dim=1) # C,kO,1,2*kR+1,2*kR+1
        I= torch.zeros_like(sktk)
        IR = (I.shape[-1]-1)//2
        I[...,IR,IR] =1.0
        kerH = I +sktk #  fC,1,2*dR+1,2*dR+1
        # ==============================
        u_new=  ifft2(deconv2fft(fft2(_w1),psf2otf(kerH,(H,W)))) # 
        
        # Subsolver (p)
        c_new = conv2c_parallel(kerK,CtoCK(u_new,K=kO,dim=1)) # B, C*kO, H, W
        p_new = softt(c_new + mu,gamma)
        
        
        # Subsolver (mu)
        mu_new = mu + c_new -p_new

        return u_new,p_new,mu_new
    
    def loop(self,x0,state_dict:Optional[Dict[str,Tensor]]=None):
        B,C,H,W = x0.shape
        
        # Input
        f = x0
        
        # State
        if state_dict == None:
            ci = self.state_init
            ci.to(x0.device) # Wierd. But ci may not be to_device properly.
            u,p,mu = x0,ci.p,ci.mu
            # repeat states: shape[0]:1-->B
            p = self.repeat_state_var(p,B)
            u = self.repeat_state_var(u,B)
            mu = self.repeat_state_var(mu,B)
        else:
            p = state_dict["p"].to(device=x0.device)
            if p.shape[0] != B:
                assert p.shape[0] ==1
                p = self.repeat_state_var(p,B)
            
            u = state_dict["u"].to(x0.device)
            if u.shape[0] != B:
                assert u.shape[0] ==1
                u = self.repeat_state_var(u,B)
                
            mu = state_dict["mu"].to(x0.device)
            if mu.shape[0] != B:
                assert mu.shape[0] ==1
                mu = self.repeat_state_var(mu,B)
                
        
        for i in range(self.n_steps):
            #TODO:TVNet.loop: Mixin control_model
            args = {k:self.controled_init[i][k]
                for k in self.controled_names
            }
            u,p,mu = self.step(f, u, p, mu, 
                                   kerK=args["kerK"],
                                   gamma=args["gamma"], 
                                   rho=args["rho"])
        return u
        
    
    def forward(self,x):
        return self.loop(x,None)
        

from dataclasses import dataclass
@dataclass
class Varia:
    """
    Parameter
    
    Required:
    kO:int      #     number of kernels; kerK.shape[0]
    kR:int      #     (2*kR+1)  size of kernel
    C:int       #     channel of $f$, the input of DecNet
    kerK:Tensor #     str;  See getKernel_dispatch.
    beta:Tensor #     1,
    rho:Tensor  #     1,
    
    """
    kO:int 
    kR:int
    C:int
    kerK:str   
    beta:float  
    rho:float
    _initialized:dict
    def to(self,device):
        for k,v in vars(self).items():
            if isinstance(v,torch.Tensor):
                setattr(self,k,v.to(device))
        return self
    def as_controled_init(self, keys:Optional[List[str]]=None):
        """None: all in _initialized"""
        o = []
        if keys in None:
            for k,v in self._initialized:
                o.append((k,v))
        else:
            _i = self._initialized
            for k in keys:
                if k in _i:
                    o.append((k,_i[k]))
                else:
                    raise Exception(f"{k} not in ._initialized: {_i.keys()}")
        return o
        
        
        
def init_varia(varia:Varia,device:torch.device):
    """
    Return None; Store to varia._initialized

    

    beta:Tensor #     1,  C, 1, 1
    rho:Tensor#       C*kO,1, 1, 1
    gamma:Tensor#     1,C*kO, 1, 1
    
    kerK:Tensor #     C*K,1,2*kR+1,2*kR+1
    wR=kR
    kerW:Tensor #     C*K,1,2*wR+1,2*wR+1
    hR=8*kR
    kerH:Tensor #     C,  1,2*hR+1,2*hR+1

    """
    kO = varia.kO
    C = varia.C
    kR = varia.kR
    
    kerK = torch.cat(getKernel_dispatch(varia.kerK,kO,kR) ,dim=0)# K,1,kR,kR
    kerK = KtoCK(kerK,C=C,dim=0).clone().detach()
    
    
    beta = torch.tensor(varia.beta).reshape(1,1,1,1).float()
    rho =  torch.ones((C*kO,1,1,1))*varia.rho # CK 1 1 1

    gamma= beta.unsqueeze(1).expand(1,C,kO,1,1).reshape(1,C*kO,1,1)/varia.rho
    
    assert tuple(kerK.shape) == (C*kO,1,2*kR+1,2*kR+1),f"kerK {tuple(kerK.shape)} neq {(C*kO,1,2*kR+1,2*kR+1)}"
    assert tuple(beta.shape)== (1,1,1,1),f"beta {tuple(beta.shape) } neq {(1,1,1,1)}"
    assert tuple(gamma.shape)==(1,kO*C,1,1),f"gamma {tuple(gamma.shape)} neq {(1,kO*C,1,1)}"
    
    varia._initialized=  {
        "beta":beta.to(device), 
        "gamma":gamma.to(device), 
        "rho":rho.to(device), 
        "kerK":kerK.to(device)}