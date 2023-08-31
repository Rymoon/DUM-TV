from .tv_m import MyDataModule
class DenoiseDataModule(MyDataModule):
    pass

def get_dataset_Denoise():
    from pathlib import Path
    import dum_tv as pkg
    from ren_utils.data import list_pictures
    root_pkg = Path(pkg.__file__).parent
    # Data
    proot = Path(root_pkg,"../Datasets/Denoise/data/") # Do the sof-link
    dataset_train = DenoiseDataset(
            samples = list_pictures(Path(proot, "train/images")),
            labels = list_pictures(Path(proot, "train/labels")),
            masks = list_pictures(Path(proot, "train/masks")))
    dataset_test = DenoiseDataset(
        samples = list_pictures(Path(proot, "test/images")),
        labels = list_pictures(Path(proot, "test/labels")),
        masks = list_pictures(Path(proot, "test/masks")))

    return {"train":dataset_train,"test":dataset_test}



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
        
