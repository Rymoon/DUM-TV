# Will import:
# * xxx_h
# * xxx_m
# Things for a bunch of specific experiments
from typing import Tuple
import dum_tv as pkg
from .tv_h import MyDataModule
class DenoiseDataModule(MyDataModule):
    pass

from .tv_h import DenoiseDataset
def get_dataset_DenoiseToy():
    """{"dataset_train","dataset_test"}"""
    from pathlib import Path
    import dum_tv as pkg
    from ren_utils.data import list_pictures
    root_pkg = Path(pkg.__file__).parent
    # Data
    proot = Path(root_pkg,"../Datasets/DenoiseToy/data/") # Do the sof-link
    dataset_train = DenoiseDataset(
            clean = list_pictures(Path(proot, "train/clean")))
    dataset_test = DenoiseDataset(
        clean = list_pictures(Path(proot, "test/clean")))

    return {"dataset_train":dataset_train,"dataset_test":dataset_test}



from .tv_h import ModelBase
from .tv_m import TVNet
import torch
class Model(ModelBase):
    """
    """
    def __init__(self,tvnet:TVNet, lr:float,  lr_scheduler_frequency):
        super().__init__(lr,lr_scheduler_frequency)
  
        self.tvnet = tvnet
        self.val_metrics_dl_keys = []

    def forward(self,x:torch.Tensor):
        f = x
        self.last_train_device = f.device
        u = self.tvnet(f)
        return u
    
    def training_step(self,pack,pack_idx):
        patches = self.pack_as_patches(pack)
        f,lab,*ext = patches
        self.last_train_device = f.device
        u = self.forward(f)

        loss = 0

        lo_ulab = torch.nn.functional.mse_loss(u,lab)
        loss+=lo_ulab
        self.log("train_loss_ulab",lo_ulab)
        
        self.log("train_loss",loss)
        return loss

    def validation_step(self,pack, pack_idx):
        patches = self.pack_as_patches(pack)
        f,lab,*ext = patches
        self.last_train_device = f.device
        u = self.forward(f)

        loss = 0

        lo_ulab = torch.nn.functional.mse_loss(u,lab)
        loss+=lo_ulab
        self.log("val_loss_ulab",lo_ulab)
        
        self.log("val_loss",loss)
        return loss
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam([{"params":self.decnet.parameters(),"lr":self.lr},{"params":self.segnet.parameters(),"lr":self.lr*5}])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",factor=0.8,patience=10,verbose=True)
        return {"optimizer":opt,"lr_scheduler":{
            "interval":"step",
            "frequency":250,
            "scheduler":sch,
            "monitor":"val_loss",
        }}
        
        
from pathlib import Path
root_pkg = Path(pkg.__file__).parent
configs_fname = ".".join(Path(__file__).relative_to(root_pkg).parts)
configs_fname  = Path(configs_fname).stem+".yaml"
configs_path = Path(root_pkg,"Scripts",configs_fname)


# PLOT compile_xxx
import pytorch_lightning as pl
root_Results = Path(root_pkg,"Results")
assert (root_Results).exists(),f"Results folder not exists. Create of softlink it: {root_Results}"
def compile_iteration_tv(
    patch_shape:Tuple[int,int,int],
    n_iteration:int,# depth of DU
    varia_d:dict,
    #
    dm:MyDataModule,
    gpuid:int,
    cfn:str,
):
    """
    Iterative solver;
    """
    
    from .tv_m import Varia,init_varia
    varia = Varia(varia_d)
    init_varia(varia,device = torch.device("cpu"))
    tvnet = TVNet(patch_shape,n_iteration,None,varia.as_controled_init())
   
    from datetime import timedelta
    root_dir = Path(root_Results,cfn).as_posix()
    trainer = pl.Trainer(default_root_dir=root_dir,gpus=[gpuid])
    log_dir=  trainer.log_dir
    model = Model(tvnet,0,1)
    
    def runner(trainer:pl.Trainer, model:Model,dm):
        log_dir = trainer.log_dir
        return trainer.predict(model,datamodule=dm,return_predictions=True)

    return trainer, model, runner




def compile_training_tv(
    patch_shape:Tuple[int,int,int],
    n_iteration:int,# depth of DU
    varia_d:dict,
    #
    dm:MyDataModule,
    gpuid:int,
    cfn:str,
):
    """
    Deep unfolding; Training
    """
    
    from .tv_m import Varia,init_varia
    varia = Varia(varia_d)
    init_varia(varia,device = torch.device("cpu"))
    tvnet = TVNet(patch_shape,n_iteration,None,varia.as_controled_init())
   
    from datetime import timedelta
    root_dir = Path(root_Results,cfn).as_posix()
    trainer = pl.Trainer(default_root_dir=root_dir,gpus=[gpuid])
    log_dir=  trainer.log_dir
    model = Model(tvnet,0,1)
    
    # TODO
    
    def runner(trainer:pl.Trainer, model:Model,dm):
        log_dir = trainer.log_dir
        return trainer.fit(model,datamodule=dm,return_predictions=True)

    return trainer, model, runner