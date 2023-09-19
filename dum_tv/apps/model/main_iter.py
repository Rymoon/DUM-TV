from pathlib import Path

p = Path("/mnt/d/onedrive/OneDrive_cityu/OneDrive - City University of Hong Kong - Student/workspace/Dataset/DenoiseToy/data/test/clean/29026.jpg")
assert p.exists(),p




from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

print("PSNR(dB): the higher, the cleaner\n 30dB+ acceptable, 40dB+ good")

def load(p,show=True):
    img = Image.open(p)
    imga = np.array(img)
    if show:
        plt.imshow(img)
        plt.show()
    return imga

clean = load(p,show=False)/255
sigma = 0.1
noisy = clean + np.random.randn(*clean.shape)*sigma
psnr = peak_signal_noise_ratio(clean, noisy)


import torch
import pytorch_lightning as pl
device = torch.device("cuda:0")



from dum_tv.apps.model.tv_m import Varia, init_varia


"""
kO:int      #     number of kernels; kerK.shape[0]
kR:int      #     (2*kR+1)  size of kernel
C:int       #     channel of $f$, the input of DecNet
kerK:Tensor #     str;  See getKernel_dispatch.
beta:Tensor #     1,
rho:Tensor  #     1,"""
varia = Varia(2,1,3,"DxDy",1,1)
init_varia(varia,device)
varia_init = varia._initialized # dict



from dum_tv.apps.model.tv_m import TVNet

def loop(f, # input
         u,p,mu, # state
         *,kerK,gamma,rho,# param
         N:int): #number of loop
    assert f.shape ==u.shape
    assert f.dtype ==u.dtype
    for i in range(N):
        u,p,mu = TVNet._step(f,u,p,mu,kerK=kerK,gamma=gamma,rho=rho)
    return u,p,mu

def loop_init(f,*,kO):
    B,C,H,W = f.shape
    device= f.device
    u= torch.zeros((B,C,H,W))
    p= torch.zeros((B,C*kO,H,W))
    mu= torch.zeros((B,C*kO,H,W))
    return u.to(device),p.to(device),mu.to(device)

f = torch.tensor(noisy).permute(2,0,1).unsqueeze(0).float().to(device)
kO = varia.kO
n_iteration = 65
state = {
    "kerK":varia_init["kerK"],
    "gamma":varia_init["gamma"],
    "rho":varia_init["rho"],
}
u,p,mu = loop(f,*loop_init(f,kO=kO),**state,N = n_iteration)

uN = u

restore = uN.detach().cpu()[0].permute(1,2,0).numpy()
psnr_rest = peak_signal_noise_ratio(clean,restore)
fig,axes = plt.subplots(1,2)
ax = axes[0]
ax.imshow(restore)
ax.set_title(f"restore\npsnr={psnr_rest:.4f}")
ax = axes[1]
ax.hist(restore.flatten())

plt.imshow()
