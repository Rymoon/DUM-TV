# NOTICE: MyDataModule: Currently, valid set == test set
import pytorch_lightning as pl



# DataModule
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
    
    
    
# Model
from torch.nn import Module
from .tv_opr import *
class TVNet(Module):
    pass