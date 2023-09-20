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

from .tv_m import TVNet


class TVNet_ctrl(Module):
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
    controled_names = ("kerK","gamma","rho")# type: Tuple[str,...]
    def __init__(self,patch_shape:Tuple[int,int,int],n_iteration:int,control_model:Module,controled_init:List[Tuple[str,Tensor]]):
        """
        patch_shape: CHW
        """
        # Free from variational model
        super().__init__()
        self.patch_shape = patch_shape
        self.n_iteration = n_iteration
        self.control_model = control_model 
        
        # assert self.controled_names == tuple(i[0] for i in controled_init), f"Order and names should match:{self.controled_names} neq {tuple(i[0] for i in controled_init)}"
        cid = {k:v for k,v in controled_init}
        assert all([k in cid for k in self.controled_names]), (cid.keys(),self.controled_names)
        ccid = {k:cid[k] for k in self.controled_names}
        
        self.controled_init = ParameterList([ParameterDict({k:v.detach().clone() for k,v in ccid.items()}) for i in range(n_iteration)])
        
        # Related to variational model
        ### Maintained through layers
        # u (B,C,H,W)
        # v (B,C,H,W)
        # p (B,C*kO,H,W)
        # lamh (B,C*kO,H,W)
        # repeat B later in function `step`
        C,H,W = self.patch_shape
        kerK = ccid["kerK"]
        
        kO= kerK.shape[0]//C
        assert (kerK.shape[-1]-1)%2 == 0,kerK.shape
        kR = (kerK.shape[-1]-1)//2
        
        self.state_init = BufferDict({
            "u": torch.zeros((1,C,H,W)),
            "p": torch.zeros((1,C*kO,H,W)),
            "mu": torch.zeros((1,C*kO,H,W)),
        },persistent=False)
        
        _Constants = namedtuple(f"{type(self).__name__}_Constants","kO kR C H W")
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
    
    @staticmethod
    def _step(f,u,p,mu,*,kerK,gamma,rho):
        B,C,H,W = f.shape
        kO= kerK.shape[0]//C
        assert (kerK.shape[-1]-1)%2 == 0,kerK.shape
        kR = (kerK.shape[-1]-1)//2
        
        # Subsolver (u)
        #rho D^T
        kerW = rho* kerK.flip((-1,-2))
        # u = inv(kerH)*_w1
        _w1 = f + CKtoC(conv2c_parallel(kerW,(p-mu)),C=C,dim = 1) # B, C, H, W
        
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
        
        assert H == self.constants.H
        assert W == self.constants.W
        
        return self._step(f,u,p,mu,kerK=kerK, gamma=gamma,rho=rho)
    def loop_init(self,x0:Tensor,state_d:Optional[Dict[str,Tensor]]):
        B,C,H,W = x0.shape
        # State
        if state_d == None:
            ci = self.state_init
            ci.to(x0.device) # Wierd. But ci may not be to_device properly.
            u,p,mu = x0,ci.p,ci.mu
            # repeat states: shape[0]:1-->B
            p = self.repeat_state_var(p,B)
            u = self.repeat_state_var(u,B)
            mu = self.repeat_state_var(mu,B)
        else:
            p = state_d["p"].to(device=x0.device)
            if p.shape[0] != B:
                assert p.shape[0] ==1
                p = self.repeat_state_var(p,B)
            
            u = state_d["u"].to(x0.device)
            if u.shape[0] != B:
                assert u.shape[0] ==1
                u = self.repeat_state_var(u,B)
                
            mu = state_d["mu"].to(x0.device)
            if mu.shape[0] != B:
                assert mu.shape[0] ==1
                mu = self.repeat_state_var(mu,B)
        return u,p,mu
    
    def loop(self,x0,state_d:Optional[Dict[str,Tensor]]=None):
        
        # Input
        f = x0
        u,p,mu = self.loop_init(x0,state_d)
        for i in range(self.n_iteration):
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
        
