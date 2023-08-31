from pytorch_lightning import LightningModule
from typing import Dict,List
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

from pathlib import Path
from typing import overload,Tuple,List,Dict,Optional
from ren_utils.data import ImageDataset
class DenoiseDataset(ImageDataset):
    """
    However, currently, this works;
    [# TODO] Replace with kornia in mha7;
    """
    @overload
    def __init__(self):
        """Create an empty object"""
    @overload
    def __init__(self,samples:List[str|Path],labels:List[str|Path],masks:List[str|Path],*,resize_size:Tuple[int,int]|None=None):
        """
        samples: List(path-to-imgs,...)
        """
    def __init__(self,samples:List[str|Path]=None,labels:List[str|Path]=None,masks:List[str|Path]=None,*,resize_size:Tuple[int,int]|None=None):
        """
        samples: List(path-to-imgs,...)
        """
        if samples is not None:
            super().__init__([samples,labels,masks],resize_size=resize_size) # Not resize.
        else:
            super().__init__()
        