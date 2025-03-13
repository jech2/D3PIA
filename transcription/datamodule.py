import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import pytorch_lightning as pl

from collections import defaultdict

from transcription.dataset import MAESTRO_V3, Pop1k7, POP909

class PadCollate:
    def __init__(self, hop_size):
        self.hop_size = hop_size
    
    def __call__(self, data):
        max_len = data[0]['audio'].shape[0] // self.hop_size
        
        for datum in data:
            step_len = datum['audio'].shape[0] // self.hop_size
            datum['step_len'] = step_len
            pad_len = max_len - step_len
            pad_len_sample = pad_len * self.hop_size
            datum['audio'] = F.pad(datum['audio'], (0, pad_len_sample))

        batch = defaultdict(list)
        for key in data[0].keys():
            if key == 'audio':
                batch[key] = torch.stack([datum[key] for datum in data], 0)
            else :
                batch[key] = [datum[key] for datum in data]
        return batch

class MAESTRO_V3_DataModule(pl.LightningDataModule):
    def __init__(self,
                data_dir: str,
                train_seq_len: int,
                valid_seq_len: int,
                batch_size: int,
                hop_size: int,
                num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.train_seq_len = train_seq_len
        self.valid_seq_len = valid_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hop_size = hop_size
    
    def setup(self, stage=None):
        self.train = MAESTRO_V3(path=self.data_dir, groups=['train'], sequence_length=self.train_seq_len,
                                random_sample=True, transform=False)
        self.val = MAESTRO_V3(path=self.data_dir, groups=['validation'], sequence_length=self.valid_seq_len,
                                random_sample=False, transform=False)
        self.test= MAESTRO_V3(path=self.data_dir, groups=['test'], sequence_length=None,
                                random_sample=False, transform=False)
    
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
                          pin_memory=False,
                          collate_fn=PadCollate(hop_size=self.hop_size))
        
        
class Pop1k7_DataModule(pl.LightningDataModule):
    def __init__(self,
                data_dir: str,
                train_seq_len: int,
                valid_seq_len: int,
                batch_size: int,
                hop_size: int,
                num_workers: int,
                pr_res: int, 
                transpose: bool,
                ):
        super().__init__()
        self.data_dir = data_dir
        self.train_seq_len = train_seq_len
        self.valid_seq_len = valid_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hop_size = hop_size
        self.pr_res = pr_res
        self.transpose = transpose
    
    def setup(self, stage=None):
        self.train = Pop1k7(path=self.data_dir, groups=['train'], sequence_length=self.train_seq_len,
                                random_sample=True, transform=False, pr_res=self.pr_res, transpose=self.transpose)
        self.val = Pop1k7(path=self.data_dir, groups=['validation'], sequence_length=self.valid_seq_len,
                                random_sample=False, transform=False, pr_res=self.pr_res, transpose=False)
        self.test= Pop1k7(path=self.data_dir, groups=['test'], sequence_length=None,
                                random_sample=False, transform=False, pr_res=self.pr_res, transpose=False)
    
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
                          pin_memory=False,
                        #   collate_fn=PadCollate(hop_size=self.hop_size)
                          )
    
class POP909_DataModule(Pop1k7_DataModule):
    def __init__(self,
                data_dir: str,
                train_seq_len: int,
                valid_seq_len: int,
                batch_size: int,
                hop_size: int,
                num_workers: int,
                pr_res: int, 
                transpose: bool,
                bridge_in_arrangement: bool,
                no_chord_in_lead: bool,
                ):
        super().__init__(data_dir, train_seq_len, valid_seq_len, batch_size, hop_size, num_workers, pr_res, transpose)
        self.bridge_in_arrangement = bridge_in_arrangement
        self.no_chord_in_lead = no_chord_in_lead
    
    def setup(self, stage=None):
        self.train = POP909(path=self.data_dir, groups=['train'], sequence_length=self.train_seq_len,
                                random_sample=True, transform=False, pr_res=self.pr_res, transpose=self.transpose, bridge_in_arrangement=self.bridge_in_arrangement, no_chord_in_lead=self.no_chord_in_lead)
        self.val = POP909(path=self.data_dir, groups=['valid'], sequence_length=self.valid_seq_len,
                                random_sample=False, transform=False, pr_res=self.pr_res, transpose=False, bridge_in_arrangement=self.bridge_in_arrangement, no_chord_in_lead=self.no_chord_in_lead)
        self.test= POP909(path=self.data_dir, groups=['valid'], sequence_length=None,
                                random_sample=False, transform=False, pr_res=self.pr_res, transpose=False, bridge_in_arrangement=self.bridge_in_arrangement, no_chord_in_lead=self.no_chord_in_lead)
    