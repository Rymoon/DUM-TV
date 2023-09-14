
# Will referred by xxx_m
# Contents:
# * Base classes across this project.

from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from typing import Dict,List

# NOTICE: MyDataModule: Currently, valid set == test set
# model, for trainer of pl
class ModelBase(LightningModule):
    """
    """
    def __init__(self, lr:float =1e-3, lr_scheduler_frequency:int =1):
        super().__init__()
        

        self.lr= lr
        self.lr_scheduler_frequency=  lr_scheduler_frequency
        self.verbosed = {}

        self.last_train_device = None

        
        self.val_metrics_dl:Dict[str,List]|None = None
        self.val_metrics_dl_keys = []
    
    def to(self,*args,**kargs):
        returns =  super().to(*args,**kargs)
        self.last_train_device = self.device
        return returns


    def on_validation_epoch_start(self):
        self.val_metrics_dl=  {k:[] for k in self.val_metrics_dl_keys}
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        for k,l in self.val_metrics_dl.items():
            if len(l)>0:
                aver=sum(l).item()/len(l)
                print(f"- val_{k}_aver",aver)
                self.log(f"val_{k}_aver",aver)
                self.verbosed[f"val_{k}_aver"] = aver

        return super().on_validation_epoch_end()

    def on_train_epoch_end(self) -> None:
        """
        For Callback, it is on_train_epoch_end(self,trainer,pl_module). Different signature, What the fuck?? (pl v1.7.6)
        """
        return super().on_train_epoch_end()



# datamodule for trainer of pl
from torch.utils.data import DataLoader
class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset_train, dataset_test,*, batch_size:int ,num_workers = 8):
        super().__init__()
        self.save_hyperparameters()
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
    

    def train_dataloader(self,n_limit:int|None=None,batch_size:int|None=None):
        if n_limit is None:
            dataset = self.dataset_train
        else:
            assert n_limit>=1
            dataset = self.dataset_test.limit(n_limit)
        if batch_size is None:
            batch_size = self.hparams.batch_size
        # return DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=False,num_workers=self.hparams.num_workers)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=False) # No num_worker parameters
    
    def test_dataloader(self,n_limit:int|None=None, batch_size:int|None=None):
        if n_limit is None:
            dataset = self.dataset_test
        else:
            assert n_limit>=1
            dataset = self.dataset_test.limit(n_limit)
        if batch_size is None:
            batch_size = self.hparams.batch_size
        # return DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=False,num_workers=self.hparams.num_workers)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=False) # No num_workers paramter

    def val_dataloader(self,n_limit:int|None=None, batch_size:int|None=None):
        return self.test_dataloader(n_limit,batch_size)
    

# dataset for datamodule
from pathlib import Path
from typing import overload,Tuple,List,Dict,Optional
from ren_utils.data import ImageDataset

## dataset for denoising 

from PIL import Image
import numpy as np
from warnings import warn
from ren_utils.data import list_pictures
class DenoiseDataset(ImageDataset):
    """
    (noisy,clean)<--getitem
    """
    @overload
    def __init__(self):
        """Create an empty object"""
    @overload
    def __init__(self,noisy:List[str|Path],clean:List[str|Path],*,resize_size:Tuple[int,int]|None=None,add_noise_std=0.01):
        """
        samples: List(path-to-imgs,...)
        """
    def __init__(self,noisy:List[str|Path]=None,clean:List[str|Path]=None,*,resize_size:Tuple[int,int]|None=None,add_noise_std=0.1):
        """
        samples: List(path-to-imgs,...)
        """
        self.add_noise_std = add_noise_std
        if noisy is None:
            if clean is None:
                ImageDataset.__init__(self)
            else:
                _p = Path(clean[0]).parent
                p_noisy_root = Path(_p.parent,_p.stem+"_noisy")
                if p_noisy_root.exists() and any(p_noisy_root.iterdir()):# exist not empty
                        print(f"Already generated, use cache: {p_noisy_root}")
                        noisy = list_pictures(p_noisy_root)
                else:
                    p_noisy_root.mkdir(exist_ok=True,parents=True)
                    noisy = []
                    for p in clean:
                        imga = np.array(Image.open(p)) # [0,255]
                        noi = np.random.randn(*imga.shape)*self.add_noise_std
                        noi=noi*255
                        imgan = np.clip(np.round(imga+noi),0,255)
                        imgan = imgan.astype(imga.dtype)
                        p = Path(p_noisy_root,Path(p).stem+Path(p).suffix)
                        Image.fromarray(imgan).save(p)
                        noisy.append(p.as_posix())
                    
                ImageDataset.__init__(self,[noisy, clean], resize_size=resize_size)
        else:
            if clean is not None:
                ImageDataset.__init__(self,[noisy, clean], resize_size=resize_size)
            else: # ONLY noisy pictures
                raise Exception("Only noisy ones")