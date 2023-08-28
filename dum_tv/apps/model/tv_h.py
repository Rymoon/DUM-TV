
class ModelBase(LightningModule):
    """
    """
    def __init__(self,patch_size:int, n_crop:int, lr:float =1e-3, lr_scheduler_frequency:int =1):
        super().__init__()
        
        self.patch_size = patch_size
        self.n_crop = n_crop

        self.train_sam_crops = [DataAugmentation(apply_flip=True,crop_shape=(patch_size, patch_size)) for i in range(n_crop)]
        self.train_lab_crops = self.get_sync_crops()

        self.margin = 4
        self.stride = patch_size//3

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
    
    def get_sync_crops(self):
        """
        After: self.patch_size,n_crop,train_sam_crop
        """
        return [DataAugmentation(apply_flip=True,crop_shape=(self.patch_size, self.patch_size),sync_tranl_mesh=self.train_sam_crops[i].tranl_mesh) for i in range(self.n_crop)]

    def pack_as_patches(self,pack):
        """Usage:
        ````python
        patches = self.pack_as_patches(pack)
        f,lab,mas = patches
        self.last_train_device = f.device
        ````"""
        sam, lab, *ext = pack
        
        sam_p = [crop(sam) for crop in self.train_sam_crops]
        lab_p = [crop(lab) for crop in self.train_lab_crops]
        
        sam_p = torch.cat(sam_p,dim=0)
        lab_p = torch.cat(lab_p,dim=0)
        return (sam_p,lab_p,ext)

    def measure_binseg(self,s,lab,prefix:str):
        """
        * s: BCHW
        * lab:BCHW
        """
        import torchmetrics

        # Reshape y_true and y_pred to be 2-dimensional
        C = s.shape[1]
        s_1d = s.reshape(-1) # 2-class
        lab_1d = lab.reshape(-1)# 2-class

        # Compute the AUC score for each channel separately
        # Aware the difference in torchmetrics : AUC, ROC, AUROC
        auc = torchmetrics.AUROC(num_classes=1).to(s.device)
        auc_score = auc(s_1d,torch.round(lab_1d).to(torch.int))

        # Reshape the result back to the original shape
        # Since lab.channel === 1
        # auc_score = auc_score.view(lab.shape[0],lab.shape[1])

        self.log(f"{prefix}_auc",auc_score)
        
        
        from torchmetrics import Dice

        # Compute the Dice score for each channel separately
        dice = Dice().to(s.device)
        dice_score = dice(s, torch.round(lab).to(torch.int)) #  each channel separately by default
        # Print the result
        self.log(f"{prefix}_dice", dice_score)

        return (auc_score,dice_score)

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
