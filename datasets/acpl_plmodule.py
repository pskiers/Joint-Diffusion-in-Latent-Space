from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
#.pytorch.utilities.combined_loader import CombinedLoader

class ACPLDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader, unlabeled_loader=None, anchor_loader=None):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.unlabeled_loader = unlabeled_loader
        self.anchor_loader = anchor_loader

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader# CombinedLoader([self.val_loader, self.anchor_loader, self.unlabeled_loader], "sequential")

    def update_train_loader(self, new_loader):
        self.train_loader = new_loader
    
    # def predict_dataloader(self):
    #     return self.anchor_loader
    
    # def update_val_loader(self, new_loader_anchor, new_loader_unlabeled):
    #     self.anchor_loader = new_loader_anchor
    #     self.unlabeled_loader = new_loader_unlabeled