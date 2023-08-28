



from .tv_m import MyDataModule
class DriveDataModule(MyDataModule):
    pass




from .tv_h import ModelBase
from .tv_m import TVNet
class Model(ModelBase):
    """For training;
    
    [# TODO] Port from mha7, the kornia-batch-aug module
    
    Crop patches from each
    """
    def __init__(self,tvnet, umasked_thr , patch_size:int, n_crop:int, lr:float,  lr_scheduler_frequency):
        super().__init__(patch_size,n_crop,lr,lr_scheduler_frequency)
  
        
        assert patch_size == decnet.patch_size,"patch_size == decnet.patch_size"
        self.decnet = decnet
        self.umasked_thr = umasked_thr

        self.segnet = segnet

        self.val_metrics_dl_keys = ["auc","dice"]

    def forward(self,x:torch.Tensor):
        f = x
        self.last_train_device = f.device

        u,v = self.decnet(f)
        sraw = self.segnet(v)
        s = torch.sigmoid(sraw)
        return s,u,v
    
    def training_step(self,pack,pack_idx):
        patches = self.pack_as_patches(pack)
        f,lab,*ext = patches
        self.last_train_device = f.device
        s,u,v = self.forward(f)

        loss = 0

        lo_uf = torch.nn.functional.mse_loss(f,u)
        loss+=lo_uf
        self.log("train_loss_uf",lo_uf)

        lo_s = torch.nn.functional.mse_loss(s,lab)
        loss+=lo_s
        self.log("train_loss_seg",lo_s)
        
        self.log("train_loss",loss)
        return loss
    
    def _decomp_fu(self,batch):
        sam= batch
        patchls, meta = unfold([sam],self.patch_size-2*self.margin,stride=self.stride)
        batch, Ps = patchl2batch(patchls[0],self.patch_size-2*self.margin) 
        batch_pad = pad_reflect(batch,self.margin)

        (ub_pad,vb_pad) = self.decnet.decomposite(batch_pad)
        (ub,vb) = crop_center(ub_pad,self.margin),crop_center(vb_pad,self.margin)

        return (ub,vb),(meta,Ps)
    
    def decomposite_fully(self,batch):
        (ub,vb),(meta,Ps) = self._decomp_fu(batch)

        feature_patchls = [batch2patchl(b,Ps) for b in (ub,vb)]
        features = fold(feature_patchls,*meta)
        u,v = features
        return u,v
    
    def predict_fully(self,batch):
        """
        s,u,v"""
        (ub,vb),(meta,Ps) = self._decomp_fu(batch)
        srawb = self.segnet(vb)
        
        feature_patchls = [batch2patchl(b,Ps) for b in (ub,vb)]
        features = fold(feature_patchls,*meta) # C=3
        u,v = features

        
        sraw_patchls = [batch2patchl(srawb,Ps)]
        sraw = fold(sraw_patchls,1,*meta[1:])[0] # C=1
        s = torch.sigmoid(sraw) # Before 23-05-23, not have this line; fully is sraw, though train on s, and aver_score on s;
        return s,u,v

    def measure(self,s,lab,prefix:str):
        return self.measure_binseg(s,lab,prefix)

    def validation_step(self,pack, pack_idx):
        patches = self.pack_as_patches(pack)
        f,lab,*ext = patches
        u,v = self.decnet.decomposite(f)
        sraw = self.segnet(v)
        s = torch.sigmoid(sraw)

        loss =0
        lo_uf = torch.nn.functional.mse_loss(f,u)
        loss +=lo_uf
        self.log("val_loss_uf",lo_uf)
        lo_s = torch.nn.functional.mse_loss(s,lab)
        loss +=lo_s
        self.log("val_loss_seg",lo_s)
        
        
        auc,dice = self.measure(s,lab,"val")
        self.val_metrics_dl["auc"].append(auc)
        self.val_metrics_dl["dice"].append(dice)
        self.verbosed["segmentation"] = s.detach()
        self.verbosed["label"] = lab.detach()
        self.verbosed["features"] = (u.detach(),v.detach())

        # u_mask

        s_umasked = (1-(u.mean(dim=1,keepdim=True)<=self.umasked_thr).float()) *s
        self.measure(s_umasked,lab,"val_umasked")
        self.verbosed["segmentation_umasked"] = s_umasked.detach()

        self.log("val_loss",loss)

        ### fully
        sam_fu, lab_fu, *ext= pack
        s_fu,u_fu,v_fu =self.predict_fully(sam_fu)

        self.measure(s_fu,lab_fu,"val_fully")
        self.verbosed["segmentation_fully"]= s_fu.detach()
        self.verbosed["label_fully"] = lab_fu.detach()
        self.verbosed["features_fully"] = (u_fu.detach(),v_fu.detach())

        s_fu_umasked = (1-(u_fu.mean(dim=1,keepdim=True)<=self.umasked_thr).float()) *s_fu
        self.measure(s_fu_umasked,lab_fu,"val_fu_umasked")
        self.verbosed["segmentation_fully_umasked"] = s_fu_umasked.detach()

        return loss
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam([{"params":self.decnet.parameters(),"lr":self.lr},{"params":self.segnet.parameters(),"lr":self.lr*5}])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",factor=0.5,patience=10,verbose=True)
        return {"optimizer":opt,"lr_scheduler":{
            "interval":"step",
            "frequency":250,
            "scheduler":sch,
            "monitor":"val_loss",
        }}
        
        # 23-05-27
        # ignore self.lr_scheduler_frequency
        # opt = torch.optim.Adam([{"params":self.decnet.parameters(),"lr":self.lr},{"params":self.segnet.parameters(),"lr":self.lr*3}])
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10000,T_mult=1,verbose=False)
        # return {"optimizer":opt,"lr_scheduler":{
        #     "interval":"step",
        #     "frequency":1,
        #     "scheduler":sch,
        # }}
