# Will import:
# * xxx_h
# * xxx_m
# Things for a bunch of specific experiments
from pathlib import Path
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
            #noisy = list_pictures(Path(proot, "train/clean_noisy")),
            clean = list_pictures(Path(proot, "train/clean")))
    dataset_test = DenoiseDataset(
        #noisy = list_pictures(Path(proot, "test/clean_noisy")),
        clean = list_pictures(Path(proot, "test/clean"))
        )

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
        
        noisy,clean = pack
        f = noisy
        lab = clean
        self.last_train_device = f.device
        u = self.forward(f)

        loss = 0

        lo_ulab = torch.nn.functional.mse_loss(u,lab)
        loss+=lo_ulab
        self.log("train_loss_ulab",lo_ulab)
        
        self.log("train_loss",loss)
        return loss
    
    def validation_step(self,pack, pack_idx):
        
        noisy,clean = pack
        f = noisy
        lab = clean
        self.last_train_device = f.device
        u = self.forward(f)

        loss = 0

        lo_ulab = torch.nn.functional.mse_loss(u,lab)
        loss+=lo_ulab
        self.log("val_loss_ulab",lo_ulab)
        
        self.log("val_loss",loss)
        return loss
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam([{"params":self.tvnet.parameters(),"lr":self.lr}])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",factor=0.8,patience=10,verbose=True)
        return {"optimizer":opt,"lr_scheduler":{
            "interval":"step",
            "frequency":self.lr_scheduler_frequency,
            "scheduler":sch,
            "monitor":"val_loss",
        }}
        

# PLOT config        
from dum_tv.utils import get_configs_path
configs_path = get_configs_path(pkg,__file__)

# PLOT compile_xxx
import pytorch_lightning as pl
from dum_tv.utils import get_root_Results
import dum_tv as pkg
root_Results = get_root_Results(pkg)
assert (root_Results).exists(),f"Results folder not exists. Create of softlink it: {root_Results}"


# compiler
def compile_iteration_tv(
    # tvnet
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
    varia = Varia(**varia_d)
    init_varia(varia,device = torch.device("cpu"))
    tvnet = TVNet(patch_shape,n_iteration,None,varia.as_controled_init())
   
    from datetime import timedelta
    from ren_utils.pl import ElapsedTime,ModelSummarySave
    root_dir = Path(root_Results,cfn).as_posix()
    trainer = pl.Trainer(default_root_dir=root_dir,gpus=[gpuid],
                         callbacks=[ElapsedTime(),ModelSummarySave()])
    log_dir=  trainer.log_dir
    model = Model(tvnet,0,1)
    
    def runner(trainer:pl.Trainer, model:Model,dm):
        log_dir = trainer.log_dir
        pred = trainer.predict(model,dataloaders = dm.test_dataloader(),return_predictions=True)
        return pred

    return trainer, model, runner




def compile_training_tv(
    # tvnet
    patch_shape:Tuple[int,int,int],
    n_iteration:int,# depth of DU
    varia_d:dict,
    # model 
    lr:float,
    lr_scheduler_frequency:int, # per steps
    # optim
    max_epochs:int,
    # 
    dm:MyDataModule,
    gpuid:int,
    cfn:str,
):
    """
    Deep unfolding; Training
    """
    device = torch.device(f"cuda:{gpuid}")
    
    from .tv_m import Varia,init_varia
    varia = Varia( **varia_d)
    init_varia(varia,device = device)
    tvnet = TVNet(patch_shape,n_iteration,None,varia.as_controled_init())
   
    from datetime import timedelta
    from ren_utils.pl import ElapsedTime,ModelSummarySave
    from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,ModelCheckpoint)
    import dum_tv
    root_Results = get_root_Results(dum_tv)
    root_dir = Path(root_Results,cfn).as_posix()
    trainer = pl.Trainer(default_root_dir=root_dir,gpus=[gpuid],
                         max_epochs=max_epochs,
                         callbacks = [
                            ElapsedTime(),ModelSummarySave(),LearningRateMonitor(),
                            ModelCheckpoint(filename="last_epoch-{epoch}",save_last=True,train_time_interval= timedelta(minutes=15)),
                            ModelCheckpoint(filename='best_train_loss_epoch-{epoch}', monitor='train_loss', save_top_k=1, mode='min'),
                            ])
    log_dir=  trainer.log_dir
    model = Model(tvnet,lr,lr_scheduler_frequency)
    
    # TODO
    
    def runner(trainer:pl.Trainer, model:Model,dm):
        log_dir = trainer.log_dir
        return trainer.fit(model,datamodule=dm)

    return trainer, model, runner

if __name__ == "__main__":
    #
    from pathlib import Path

    #
    import dum_tv.apps.model.main as main
    __file__ = main.__file__

    import dum_tv as pkg
    cfn = Path(__file__).stem
    root_pkg = Path(pkg.__file__).parent


    dm_map = {
        "denoise": main.DenoiseDataModule(batch_size=4,**main.get_dataset_DenoiseToy())
    }


    # ====
    from ren_utils.pl import run_by_title

    gpuid = 0
    def run(title,gpuid,dm_name,dm):
        return run_by_title(title,gpuid,f"{cfn}__{dm_name}",dm,config_parser_dict=vars(main),p_configs=main.configs_path)

    dm_name = "denoise"
    dm = dm_map[dm_name]

    group = ["Iteration"]

    if "Iteration" in group:
        prediction = run("iteration",gpuid,dm_name,dm)