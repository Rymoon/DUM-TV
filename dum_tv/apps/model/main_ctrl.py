from .main import Model
from pathlib import Path
from typing import Tuple
import dum_tv as pkg
from .tv_h import MyDataModule
import torch
class Model_ctrl(Model):
    """
    """
    pass

# PLOT config        
from dum_tv.utils import get_configs_path
configs_path = get_configs_path(pkg,__file__)


# PLOT compile_xxx
import pytorch_lightning as pl
from dum_tv.utils import get_root_Results
import dum_tv as pkg
root_Results = get_root_Results(pkg)
assert (root_Results).exists(),f"Results folder not exists. Create of softlink it: {root_Results}"

from .tv_ctrl_m import TVNet_ctrl,CtrlNet_alex
def compile_training_tv_ctrl(
    # TVNet_ctrl.__init__(...)
    patch_shape:Tuple[int,int,int],
    n_iteration:int,# depth of DU
    varia_d:dict, # -> controled_init
    # Model_ctrl.__init__(tvnet,...)
    lr:float,
    lr_scheduler_frequency:int, # per steps
    # Trainer(...)
    max_epochs:int,
    # 
    dm:MyDataModule,
    gpuid:int,
    cfn:str,
):
    device = torch.device(f"cuda:{gpuid}")
    
    from .tv_m import Varia,init_varia
    varia = Varia( **varia_d)
    init_varia(varia,device = device)
    ctrlnet = CtrlNet_alex(varia.as_controled_init())
    tvnet = TVNet_ctrl(patch_shape,n_iteration,ctrlnet)
    
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