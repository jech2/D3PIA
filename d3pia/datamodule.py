from torch.utils.data import DataLoader
import pytorch_lightning as pl

from d3pia.dataset import POP909

class POP909_DataModule(pl.LightningDataModule):
    def __init__(self,
                data_dir: str,
                train_seq_len: int,
                valid_seq_len: int,
                batch_size: int,
                num_workers: int,
                pr_res: int, 
                transpose: bool,
                bridge_in_arrangement: bool,
                no_chord_in_lead: bool,
                ):
        super().__init__()
        self.data_dir = data_dir
        self.train_seq_len = train_seq_len
        self.valid_seq_len = valid_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pr_res = pr_res
        self.transpose = transpose
        
        self.bridge_in_arrangement = bridge_in_arrangement
        self.no_chord_in_lead = no_chord_in_lead
    
    def setup(self, stage=None):
        self.train = POP909(path=self.data_dir, groups=['train'], sequence_length=self.train_seq_len,
                                random_sample=True, pr_res=self.pr_res, transpose=self.transpose, bridge_in_arrangement=self.bridge_in_arrangement, no_chord_in_lead=self.no_chord_in_lead)
        self.val = POP909(path=self.data_dir, groups=['valid'], sequence_length=self.valid_seq_len,
                                random_sample=False, pr_res=self.pr_res, transpose=False, bridge_in_arrangement=self.bridge_in_arrangement, no_chord_in_lead=self.no_chord_in_lead)
        self.test= POP909(path=self.data_dir, groups=['valid'], sequence_length=None,
                                random_sample=False, pr_res=self.pr_res, transpose=False, bridge_in_arrangement=self.bridge_in_arrangement, no_chord_in_lead=self.no_chord_in_lead)
    
    def train_dataloader(self):
        return DataLoader(self.train, sampler=None,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=True,
                          persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, sampler=None,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_sampler=None,
                        #   batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False)