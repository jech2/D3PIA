from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from termcolor import colored

import numpy as np
import matplotlib.pyplot as plt

# from adabelief_pytorch import AdaBelief

from transcription.diffusion.trainer import DiscreteDiffusion
from transcription.diffusion.ema import EMA
from transcription.constants import HOP
from transcription.evaluate import evaluate

from transcription.diffusion.trainer import log_onehot_to_index
from transcription.decoder import LSTM_NATTEN

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, ReduceLROnPlateau

torch.autograd.set_detect_anomaly(True)

def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)
   

class D3RM(DiscreteDiffusion):
    def __init__(self,
                encoder: nn.Module,
                decoder: nn.Module,
                use_style_enc: bool,
                style_enc_ckpt: str,
                ref_arr_style_path: str | None,
                encoder_parameters: str,
                pretrained_encoder_path: str,
                freeze_encoder: bool,
                test_save_path: str,
                inpainting_ratio: float,
                optimizer: OptimizerCallable = torch.optim.AdamW,
                scheduler: LRSchedulerCallable = ReduceLROnPlateau,
                 *args, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder, *args, **kwargs)
        self.save_hyperparameters() # for wandb logging
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_save_path = test_save_path
        self.inpainting_ratio = inpainting_ratio
        self.ref_arr_style_path = ref_arr_style_path
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

        # freeze texture encoder
        if use_style_enc:
            assert style_enc_ckpt is not None and Path(style_enc_ckpt).exists()
            assert self.encoder.use_style_enc == False and self.decoder.use_style_enc == True
            
            from transcription.style_encoder import load_pretrained_style_enc
            self.style_enc = load_pretrained_style_enc(
                    style_enc_ckpt,
                    256, # txt_emb_size
                    1024, # txt_hidden_dim
                    256, # txt_z_dim
                    10, # txt_num_channel
                )
            
            print(colored('Finished loading style_enc', 'blue', attrs=['bold']))
        else:
            assert self.encoder.use_style_enc == False and self.decoder.use_style_enc == False
            self.style_enc = None
            print(colored('No style encoder', 'blue', attrs=['bold']))
            
        if self.style_enc is not None:
            for param in self.style_enc.parameters():
                param.requires_grad = False
                
            # number of param
            total_params = sum(p.numel() for p in self.style_enc.parameters())
            print(total_params)

        self.step = 0
        self.validation_step_outputs = defaultdict(list)

    def _encode_style(self, prmat):
        z_list = []
        if self.style_enc is not None:
            for prmat_seg in prmat.split(32, 1):  # (#B, 32, 128) * 4
                z_seg = self.style_enc(prmat_seg).mean
                z_list.append(z_seg)
            z = torch.cat(z_list, dim=-1)
            return z
        else:
            # print(f"unencoded txt: {prmat.shape}")
            return None

    def training_step(self, batch, batch_idx):
        arrangement = batch['arrangement'].to(self.device).to(torch.int64) # B x T x 88
        leadsheet = batch['leadsheet'].to(self.device).to(torch.float32) # B x T x 88
        chord = batch['chord'].to(self.device).to(torch.float32) # B x T x 88

        prmat = batch['prmat'].to(self.device).to(torch.float32)
        
        style_emb = self._encode_style(prmat)

        # forward
        arrangement = arrangement.reshape(arrangement.shape[0], -1)

        disc_diffusion_loss = self(arrangement, leadsheet, style_emb, return_loss=True, cfg_features=chord)
        self.log('train/diffusion_loss', disc_diffusion_loss['loss'].mean(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        return disc_diffusion_loss['loss']
    
    def validation_step(self, batch, batch_idx):

        arrangement = batch['arrangement'].to(self.device).to(torch.int64) # B x T x 88
        leadsheet = batch['leadsheet'].to(self.device).to(torch.float32) # B x T x 88
        chord = batch['chord'].to(self.device).to(torch.float32) # B x T x 88

        prmat = batch['prmat'].to(self.device).to(torch.float32)
        style_emb = self._encode_style(prmat)

        shape = arrangement.shape
        # Forward step (for loss calculation)    
        arrangement = arrangement.reshape(arrangement.shape[0], -1)
        disc_diffusion_loss = self(arrangement, leadsheet, style_emb, return_loss=True, cfg_features=chord)
        frame_out, _ = self.sample_func(leadsheet, prev_piano=None, style_emb=style_emb) # frame out: B x T x 88 x C
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
        fname = Path(batch['fname'][0]).stem
        # out_fp = Path(self.test_save_path) / f'{batch_idx}_{fname}.npz'
        out_fp = Path(self.test_save_path) / f'{fname}.npz'
        if out_fp.exists():
            print('already inference done, skip this index', {batch_idx})
            return
                    
        leadsheet = batch['leadsheet'].to(self.device)   

        if self.ref_arr_style_path != None:
            print('using ref_arr_style_path', self.ref_arr_style_path)
            prmat = get_prmat_from_midi(self.ref_arr_style_path)
            prmat = torch.tensor(prmat).to(self.device).to(torch.float32)
        else:
            prmat = batch['prmat'].to(self.device).to(torch.float32)

        B, total_frame, _ = leadsheet.shape
        test_metric = defaultdict(list)

        shape = leadsheet.shape
        seg_len = 128 
        hop_size = seg_len * (1-self.inpainting_ratio)

        # n_seg = int((total_frame - seg_len) // hop_size + 1)
        n_seg = int(total_frame // hop_size) + 1

        frame_outs = torch.zeros(shape, dtype=torch.int)

        for seg in tqdm(range(n_seg)):
            start = int(seg * hop_size)
            end = start + seg_len
            
            leadsheet_pad = leadsheet[:, int(start):int(end)].to(self.device)
            if self.ref_arr_style_path != None:
                prmat_pad = prmat[:seg_len].to(self.device)
            else:
                prmat_pad = prmat[:, int(start):int(end)].to(self.device)
                
            if seg_len > leadsheet_pad.shape[1]:
                n_pad = seg_len - leadsheet_pad.shape[1]
                leadsheet_pad = F.pad(leadsheet_pad, (0, 0, 0, n_pad, 0, 0), mode='constant', value=0)
                prmat_pad = F.pad(prmat_pad, (0, 0, 0, n_pad, 0, 0), mode='constant', value=0)
            style_emb = self._encode_style(prmat_pad)

            # print(start, end, seg_len, seg, hop_size, leadsheet_pad.shape)
            # print(leadsheet_pad.shape)
            if self.inpainting_ratio == 0:
                frame_out, _ = self.sample_func(leadsheet_pad, prev_piano=None, style_emb=style_emb)
            else:
                if seg == 0:
                    frame_out, _ = self.sample_func(leadsheet_pad, prev_piano=None, style_emb=style_emb)
                    prev_piano = frame_out.reshape(frame_out.shape[0], -1, 88)
                else:   
                    frame_out, labels = self.sample_func(leadsheet_pad, prev_piano, style_emb=style_emb, visualize_denoising=True)
                    # Path(self.test_save_path).mkdir(parents=True, exist_ok=True)
                    # for idx in range(len(frame_out)):
                    #     print(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}.png')
                    #     plt.figure(figsize=(8, 4))
                    #     plt.imshow(prev_piano[idx].detach().cpu().numpy().T, origin='lower', aspect='auto')
                    #     plt.title(f'Piano roll {idx}')
                    #     plt.savefig(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}_prev_piano.png')
                    #     plt.close()
                    
                    #     plt.figure(figsize=(8, 4))
                    #     plt.imshow(labels[idx].T, origin='lower', aspect='auto')
                    #     plt.title(f'Piano roll {idx}')
                    #     plt.savefig(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}_labels.png')
                    #     plt.close() 
                    
                    prev_piano = frame_out.reshape(frame_out.shape[0], -1, 88)
                    # save labels as visualized piano roll using matplotlib
                    # print(len(frame_out), frame_out.shape)
                    
                    # for idx in range(len(frame_out)):
                    #     print(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}.png')
                    #     plt.figure(figsize=(8, 4))
                    #     plt.imshow(prev_piano[idx].detach().cpu().numpy().T, origin='lower', aspect='auto')
                    #     plt.title(f'Piano roll {idx}')
                    #     plt.savefig(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}.png')
                    #     plt.close()

                    sample = frame_out.reshape(frame_out.shape[0], -1, 88).detach()
                    frame_outs[:, int(start+seg_len*(1-self.inpainting_ratio)):end] = sample[:, int(seg_len*(1-self.inpainting_ratio)):seg_len]

            
            # for idx in range(len(frame_out)):
            #     # leadsheet
            #     plt.figure(figsize=(8,4))
            #     plt.imshow(leadsheet_pad[idx].detach().cpu().numpy().T, origin='lower', aspect='auto')
            #     plt.title(f'Piano roll {idx}')
            #     plt.savefig(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}_leadsheet.png')
            #     plt.close()
                    
            sample = frame_out.reshape(frame_out.shape[0], -1, 88).detach()
            if end > total_frame:
                frame_outs[:, start:] = sample[:, :seg_len-(end-total_frame)]
            else:
                frame_outs[:, start:end] = sample[:, :seg_len]

            # for idx in range(len(frame_out)):
            #     # frame out
            #     plt.figure(figsize=(8,4))
            #     plt.imshow(sample[idx].detach().cpu().numpy().T, origin='lower', aspect='auto')
            #     plt.title(f'Piano roll {idx}')
            #     plt.savefig(Path(self.test_save_path) / f'piano_roll_{batch_idx}_{seg}_{idx}.png')
            #     plt.close()


        Path(self.test_save_path).mkdir(parents=True, exist_ok=True)
        for idx in range(len(frame_outs)):
            
            pred = frame_outs[idx].detach().cpu().numpy()
            print(pred.shape)
            print('arr shape', batch['arrangement'][0].shape)
            output = {
                'pred': pred,
                'arrangement': batch['arrangement'][idx].detach().cpu().numpy(),
                'leadsheet': batch['leadsheet'][idx].detach().cpu().numpy(),
            }
            np.savez(out_fp, **output)
            
    def sample_func(self, leadsheet, prev_piano, style_emb=None, visualize_denoising=False):
        tic = time.time()
        if self.inpainting_ratio == 0 or prev_piano is None:
            samples, labels = self.sample(leadsheet, style_emb,
                                        visualize_denoising=visualize_denoising)
        else:
            with torch.no_grad():
                samples, labels = self.sample_inpainting(leadsheet, style_emb, prev_piano, 
                                            inpainting_ratio=self.inpainting_ratio,
                                            visualize_denoising=visualize_denoising,
                                            shape=leadsheet.shape)
        
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
        
    def predict_start(self, log_x_t, cond_audio, style_emb, t, sampling=False, cond_chord_for_cfg=None):
        # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        
        if sampling==False:
            if isinstance(self.encoder, LSTM_NATTEN):
                feature = self.encoder(cond_audio, label_emb=self.decoder.label_emb)
                feature_for_cfg = self.encoder(cond_chord_for_cfg, label_emb=self.decoder.label_emb) if cond_chord_for_cfg is not None else None
            else:
                feature = self.encoder(cond_audio)
                feature_for_cfg = self.encoder(cond_chord_for_cfg) if cond_chord_for_cfg is not None else None
                
            out = self.decoder(x_t, feature, t, style_emb)
        if sampling==True:
            if t[0].item() == self.num_timesteps-1:
                if isinstance(self.encoder, LSTM_NATTEN):
                    feature = self.encoder(cond_audio, label_emb=self.decoder.label_emb)
                    feature_for_cfg = self.encoder(cond_chord_for_cfg, label_emb=self.decoder.label_emb) if cond_chord_for_cfg is not None else None
                else:
                    feature = self.encoder(cond_audio)
                    feature_for_cfg = self.encoder(cond_chord_for_cfg) if cond_chord_for_cfg is not None else None
        
                out = self.decoder(x_t, feature, t, style_emb, cfg_feature=feature_for_cfg)
                self.saved_encoder_features = feature
                self.saved_encoder_features_for_cfg = feature_for_cfg
            else:
                assert self.saved_encoder_features is not None
                out = self.decoder(x_t, self.saved_encoder_features, t, style_emb, cfg_feature=self.saved_encoder_features_for_cfg)
            
            if hasattr(self.decoder, 'cond_scale'):
                cond_scale = self.decoder.cond_scale
                null_out = self.decoder(x_t, self.saved_encoder_features, t, style_emb, cond_drop_prob=1.)
                out = out * cond_scale + null_out * (1-cond_scale)
                    
            # if t[0].item() == 0: self.saved_encoder_features = None

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


def get_prmat_from_midi(path):
    from midisym.parser.midi import MidiParser
    from midisym.converter.matrix import make_grid_quantized_notes, make_grid_quantized_note_prmat

    midi_parser = MidiParser(path, use_symusic=True)
    sym_obj = midi_parser.sym_music_container
    # piano_rolls, piano_roll_xs, note_infos = get_absolute_time_mat(sym_obj, pr_res=self.pr_res, chord_style=self.chord_style)

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI -- grid from ticks",
    )
    prmat = make_grid_quantized_note_prmat(sym_obj, grid, value='duration', do_slicing=False)
    return prmat