from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from termcolor import colored

import numpy as np

# from adabelief_pytorch import AdaBelief

from transcription.diffusion.trainer import DiscreteDiffusion
from transcription.diffusion.ema import EMA
from transcription.constants import HOP
from transcription.evaluate import evaluate

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, ReduceLROnPlateau

torch.autograd.set_detect_anomaly(True)

def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)
   

class D3RM(DiscreteDiffusion):
    def __init__(self,
                encoder: nn.Module,
                decoder: nn.Module,
                encoder_parameters: str,
                pretrained_encoder_path: str,
                freeze_encoder: bool,
                test_save_path: str,
                # use_ema: bool,
                # ema_decay: float,
                # ema_update_interval: int,
                optimizer: OptimizerCallable = torch.optim.AdamW,
                scheduler: LRSchedulerCallable = ReduceLROnPlateau,
                 *args, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder, *args, **kwargs)
        self.save_hyperparameters() # for wandb logging
        # self.encoder = encoder
        # self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_save_path = test_save_path
        # self.use_ema = use_ema
        if encoder_parameters == "pretrained":
            ckpt = torch.load(pretrained_encoder_path)
            self.encoder.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(colored('Load pretrained encoder', 'green', attrs=['bold']))
        else: print(colored('Loading random initialized encoder', 'green', attrs=['bold']))
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(colored('Freeze encoder', 'blue', attrs=['bold']))
        else: print(colored('Train encoder', 'blue', attrs=['bold']))
        # if self.use_ema: # TODO: check if EMA are also well-saved and loaded in ckpt
        #     print(colored(f'Use EMA, load in {self.device}', 'blue', attrs=['bold']))
        #     self.ema_encoder = EMA(self.encoder, decay=ema_decay, update_interval=ema_update_interval, device=self.device)
        #     self.ema_decoder= EMA(self.decoder, decay=ema_decay, update_interval=ema_update_interval, device=self.device)
        self.step = 0
        self.validation_step_outputs = defaultdict(list)

    def training_step(self, batch, batch_idx):
        # audio = batch['audio'].to(self.device) # B x L
        # label = batch['label'][:,1:,:].to(self.device) # B x T x 88
        # _ = batch['velocity'].to(device)
        
        arrangement = batch['arrangement'].to(self.device).to(torch.int64) # B x T x 88
        leadsheet = batch['leadsheet'].to(self.device).to(torch.float32) # B x T x 88

        # forward
        arrangement = arrangement.reshape(arrangement.shape[0], -1)

        disc_diffusion_loss = self(arrangement, leadsheet, return_loss=True)
        self.log('train/diffusion_loss', disc_diffusion_loss['loss'].mean(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        # if self.use_ema:
        #     self.ema_encoder.update(iteration=self.step)
        #     self.ema_decoder.update(iteration=self.step)
        return disc_diffusion_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        # audio = batch['audio'].to(self.device)
        # label = batch['label'][:,1:,:].to(self.device)

        arrangement = batch['arrangement'].to(self.device).to(torch.int64) # B x T x 88
        leadsheet = batch['leadsheet'].to(self.device).to(torch.float32) # B x T x 88

        shape = arrangement.shape
        # Forward step (for loss calculation)    
        arrangement = arrangement.reshape(arrangement.shape[0], -1)
        disc_diffusion_loss = self(arrangement, leadsheet, return_loss=True)
        frame_out, _ = self.sample_func(leadsheet) # frame out: B x T x 88 x C
        accuracy = (frame_out == arrangement).float()
        validation_metric = defaultdict(list)
        for n in range(leadsheet.shape[0]):
            sample = frame_out.reshape(*shape)[n] 
            metrics = evaluate(sample, arrangement.reshape(*shape)[n], band_eval=False)
            for k, v in metrics.items(): validation_metric[k].append(v)
        # for k, v in validation_metric.items():
        #     validation_metric[k] = torch.tensor(np.mean(np.concatenate(v)), device=self.device)
        validation_metric['val/accuracy_loss'] = accuracy.mean(dim=-1)
        validation_metric['val/diffusion_loss'] = disc_diffusion_loss['loss'].unsqueeze(0)
        # self.log_dict(validation_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in validation_metric.items():
            self.validation_step_outputs[k].extend(v)
    
    def on_validation_epoch_end(self):
        validation_metric_mean = defaultdict(list)
        for k, v in self.validation_step_outputs.items():
            if 'loss' in k:
                validation_metric_mean[k] = torch.mean(torch.stack(v))
            else:
                validation_metric_mean[k] = torch.tensor(np.mean(np.concatenate(v)), device=self.device)
        self.log_dict(validation_metric_mean, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        leadsheet = batch['leadsheet'].to(self.device)   
        B, n_step, _ = leadsheet.shape
        # audio = batch['audio'].to(self.device)
        # B, audio_len = audio.shape[0], audio.shape[1]
        test_metric = defaultdict(list)

        frame_outs = [] 
        # n_step = (audio_len - 1) // HOP + 1
        # shape = (audio.shape[0], n_step, 88)
        shape = leadsheet.shape
        seg_len = 313
        overlap = 50

        n_seg = (n_step - overlap) // (seg_len - overlap) + 1

        frame_outs = torch.zeros(shape, dtype=torch.int)

        for seg in tqdm(range(n_seg)):
        # for seg in tqdm(range(2)):
            start = seg * (seg_len - overlap)
            end = start + seg_len
            start = seg * (seg_len - overlap)
            end = start + seg_len
            if end > n_step:
                pad_len = end - n_step - 1
                leadsheet_pad = F.pad(leadsheet[:, int(start):], (0, 0, pad_len, 1))
                leadsheet_pad = leadsheet_pad.to(self.device)
            else:
                leadsheet_pad = leadsheet[:, int(start):int(end)].to(self.device)

            print(leadsheet_pad.shape)
            # if end > n_step:
            #     pad_len = n_step*HOP - audio_len
            #     start = n_step - seg_len
            #     end = n_step
            #     audio_pad = F.pad(audio[:, int(start*HOP):], (0, pad_len))
            #     audio_pad = audio_pad.to(self.device)
            # else:
            #     audio_pad = audio[:, int(start*HOP):int(end*HOP)].to(self.device)

            frame_out, _ = self.sample_func(leadsheet_pad)
            # frame_out, _ = self.sample_func(audio_pad)
            sample = frame_out.reshape(frame_out.shape[0], -1, 88).detach()
            if end > n_step:
                frame_outs[:, start:] = sample[:, :n_step-start]
            else:
                frame_outs[:, start:end] = sample[:, :seg_len]
            # if seg == 0:
            #     frame_outs[:, :end-overlap//2] = sample[:, :-overlap//2]
            # elif seg == n_seg - 1:
            #     frame_outs[:, start+overlap//2:] = sample[:, overlap//2:n_step-start]
            # else:
            #     frame_outs[:, start+overlap//2:end-overlap//2] = sample[:, overlap//2:-overlap//2]
        
        # frame out: B x T x 88 x C
        # vel_outs = []
        # obtain metrics on full-length audio for batches
        # for n in range(batch['audio'].shape[0]):
        #     step_len = batch['step_len'][n]
        #     frame = frame_outs[n][:step_len].detach().cpu()
        #     metrics = evaluate(frame, batch['label'][n][1:].detach().cpu()[:-1], band_eval=False)
        #     for k, v in metrics.items():
        #         test_metric[k].append(v)
        #     print(f'\n note f1: {metrics["metric/note/f1"][0]:.4f}, note_with_offsets_f1: {metrics["metric_note_with_offsets_f1"][0]:.4f}', batch['path'])
        #     print(f'\n note precision: {metrics["metric/note/precision"][0]:.4f}, note_with_offsets_precision : {metrics["metric/note-with-offsets/precision"][0]:.4f}', batch['path'])
        #     print(f'\n note recall: {metrics["metric/note/recall"][0]:.4f}, note_with_offsets_recall : {metrics["metric/note-with-offsets/recall"][0]:.4f}', batch['path'])
        
        # self.log_dict(test_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True,sync_dist=True)

        Path(self.test_save_path).mkdir(parents=True, exist_ok=True)
        for idx in range(len(frame_outs)):
            out_fp = Path(self.test_save_path) / f'{idx}.npz'
            if out_fp.exists():
                print('already inference done, skip this index', {idx})
            
            pred = frame_outs[idx].detach().cpu().numpy()
            print(pred.shape)
            print(batch['arrangement'][0].shape)
            output = {
                'pred': pred,
                'arrangement': batch['arrangement'][n].detach().cpu().numpy(),
                'leadsheet': batch['leadsheet'][n].detach().cpu().numpy(),
            }
            np.savez(out_fp, **output)
            
    def sample_func(self, audio, visualize_denoising=False):
        tic = time.time()
        # if self.use_ema:
        #     self.ema_encoder.modify_to_inference()
        #     self.ema_decoder.modify_to_inference()
        #     suffix = '_ema'
        # else:
        #     suffix = ''
        with torch.no_grad():
            samples, labels = self.sample(audio,
                                        filter_ratio=0,
                                        visualize_denoising=visualize_denoising
                                        ) 

            # if self.use_ema:
            #     self.ema_encoder.modify_to_train()
            #     self.ema_decoder.modify_to_train()
        return samples['label_token'], labels

    def configure_optimizers(self):
        optimizer = self.optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()))
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler.monitor,
                    "interval": "step",
                    "frequency": 1,
                    }}
        
    def predict_start(self, log_x_t, cond_audio, t, sampling=False):
        # p(x0|xt)
        from transcription.diffusion.trainer import log_onehot_to_index
        x_t = log_onehot_to_index(log_x_t)
        
        from transcription.decoder import LSTM_NATTEN
        if sampling==False:
            if isinstance(self.encoder, LSTM_NATTEN):
                feature = self.encoder(cond_audio, label_emb=self.decoder.label_emb)
            else:
                feature = self.encoder(cond_audio)
                
            out = self.decoder(x_t, feature, t)
        if sampling==True:
            if t[0].item() == self.num_timesteps-1:
                if isinstance(self.encoder, LSTM_NATTEN):
                    feature = self.encoder(cond_audio, label_emb=self.decoder.label_emb)
                else:
                    feature = self.encoder(cond_audio)
                out = self.decoder(x_t, feature, t)
                self.saved_encoder_features = feature
            else:
                assert self.saved_encoder_features is not None
                out = self.decoder(x_t, self.saved_encoder_features, t)
            if t[0].item() == 0: self.saved_encoder_features = None

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float() # softmax then log
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.label_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred
