# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import importlib

import numpy as np
import pytorch_lightning as pl
from typing import List
from termcolor import colored

from inspect import isfunction
eps = 1e-8

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    # a = log(a') -> return log(1 - a')
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    # a = log(a'), b = log(b') -> return log(a' + b')
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # (1, 1024) -> (1, 1024, 2888)
    permute_order = (0, -1) + tuple(range(1, len(x.size()))) # 0, -1, 1
    x_onehot = x_onehot.permute(permute_order) # (1, 1024, 2888) -> (1, 2888, 1024)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # accumulate (0.9999 ~ 0)
    att = np.concatenate(([1], att)) 
    at = att[1:]/att[:-1] # alpha (no change prob)
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # accumulate (0 ~ 0.9999)
    ctt = np.concatenate(([0], ctt)) # this is \bar{gamma}
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct # This is gamma (random mask prob)
    bt = (1-at-ct)/N # This is beta (random resample prob)
    att = np.concatenate((att[1:], [1])) # last 1 is for mask
    ctt = np.concatenate((ctt[1:], [0])) # last 0 is for mask
    btt = (1-att-ctt)/N # This is \bar{beta}
    return at, bt, ct, att, btt, ctt

class DiscreteDiffusion(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        label_seq_len: int,
        diffusion_step: int,
        gamma_bar_T: float,
        auxiliary_loss_weight: float,
        adaptive_auxiliary_loss: bool,
        mask_weight: List[float],
        onset_suppress_sample,
        onset_weight_kl,
        no_mask: bool,
        sample_from_fully_masked: bool,
        reverse_sampling: bool,
        repaint: int,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_seq_len = self.shape = label_seq_len
        self.num_classes = self.decoder.num_state
        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        self.repaint = repaint
        if onset_suppress_sample:
            self.onset_suppress = onset_suppress_sample
        else : self.onset_suppress = False
        if onset_weight_kl:
            self.onset_weight_kl = onset_weight_kl
        else : self.onset_weight_kl = False
        
        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1, ctt_T=gamma_bar_T) # (t,)
        atr, btr, ctr, attr, bttr, cttr = alpha_schedule(self.num_timesteps, N=self.num_classes-1, ctt_T=0.99999) # (t,)
        print(colored(f"alpha schedule: {att[1]:.4f} -> {att[-2]:.4f}, {btt[1]:.4f} -> {btt[-2]:.4f}, {ctt[1]:.4f} -> {ctt[-2]:.4f}", "green"))
        print(colored(f"reverse schedule: {attr[1]:.4f} -> {attr[-2]:.4f}, {bttr[1]:.4f} -> {bttr[-2]:.4f}, {cttr[1]:.4f} -> {cttr[-2]:.4f}", "green"))
        
        self.reverse_sampling = reverse_sampling
        print(f"Reverse Sampling : {self.reverse_sampling}")
        self.fully_masked = sample_from_fully_masked

        at = torch.tensor(at.astype('float64')) # (t,)
        bt = torch.tensor(bt.astype('float64')) # (t,)
        ct = torch.tensor(ct.astype('float64')) # (t,)
        log_at = torch.log(at) # (t,)
        log_bt = torch.log(bt) # (t,)
        log_ct = torch.log(ct) # (t,)
        att = torch.tensor(att.astype('float64')) # (t+1,)
        btt = torch.tensor(btt.astype('float64')) # (t+1,)
        ctt = torch.tensor(ctt.astype('float64')) # (t+1,)
        log_cumprod_at = torch.log(att) # (t+1,)
        log_cumprod_bt = torch.log(btt) # (t+1,)
        log_cumprod_ct = torch.log(ctt) # (t+1,)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        if self.reverse_sampling:
            atr = torch.tensor(atr.astype('float64')) # (t,)
            btr = torch.tensor(btr.astype('float64')) # (t,)
            ctr = torch.tensor(ctr.astype('float64'))
            log_atr = torch.log(atr) # (t,)
            log_btr = torch.log(btr) # (t,)
            log_ctr = torch.log(ctr) # (t,)
            attr = torch.tensor(attr.astype('float64'))
            bttr = torch.tensor(bttr.astype('float64'))
            cttr = torch.tensor(cttr.astype('float64'))
            log_cumprod_atr = torch.log(attr) # (t+1,)
            log_cumprod_btr = torch.log(bttr)
            log_cumprod_ctr = torch.log(cttr)
            
            log_1_min_ctr = log_1_min_a(log_ctr)
            log_1_min_cumprod_ctr = log_1_min_a(log_cumprod_ctr)

            assert log_add_exp(log_ctr, log_1_min_ctr).abs().sum().item() < 1.e-5
            assert log_add_exp(log_cumprod_ctr, log_1_min_cumprod_ctr).abs().sum().item() < 1.e-5

            self.register_buffer('log_atr', log_atr.float())
            self.register_buffer('log_btr', log_btr.float())
            self.register_buffer('log_ctr', log_ctr.float())
            self.register_buffer('log_cumprod_atr', log_cumprod_atr.float())
            self.register_buffer('log_cumprod_btr', log_cumprod_btr.float())
            self.register_buffer('log_cumprod_ctr', log_cumprod_ctr.float())
            self.register_buffer('log_1_min_ctr', log_1_min_ctr.float())
            self.register_buffer('log_1_min_cumprod_ctr', log_1_min_cumprod_ctr.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None
        self.saved_encoder_features = None # for sampling


    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        if self.onset_weight_kl:
            kl = (log_prob1.exp() * (log_prob1 - log_prob2))
            kl[:, [2, 4], :] *= (1 + self.onset_weight_kl)  # Apply onset_weight_kl factor to indices 2 and 4
            kl = kl.sum(dim=1)
        else:
            kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl


    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred_one_timestep_reverse(self, log_x_t, t):         # q(xt|xt_1)
        log_atr = extract(self.log_atr, t, log_x_t.shape)             # at
        log_btr = extract(self.log_btr, t, log_x_t.shape)             # bt
        log_ctr = extract(self.log_ctr, t, log_x_t.shape)             # ct
        log_1_min_ctr = extract(self.log_1_min_ctr, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_atr, log_btr),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ctr, log_ctr)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        '''
        log_x_start shape : (B, 2888(=one-hot token), 1024(=label_len))
        '''
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)        
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)        
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)        
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape) 
        

        log_probs = torch.cat( 
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct) 
            ],
            dim=1
        ) # B x state x F*T

        return log_probs
    
    def q_pred_reverse(self, log_x_start, t):           # q(xt|x0)
        '''
        log_x_start shape : (B, 2888(=one-hot token), 1024(=label_len))
        '''
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_atr = extract(self.log_cumprod_atr, t, log_x_start.shape)         # log \bar{\alpha},  B x 1 x 1
        log_cumprod_btr = extract(self.log_cumprod_btr, t, log_x_start.shape)         # log \bar{\beta},  B x 1 x 1
        log_cumprod_ctr = extract(self.log_cumprod_ctr, t, log_x_start.shape)         # log \bar{\gamma}
        log_1_min_cumprod_ctr = extract(self.log_1_min_cumprod_ctr, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat( 
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_atr, log_cumprod_btr), 
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ctr, log_cumprod_ctr) 
            ],
            dim=1
        ) # B x state x F*T

        return log_probs

    def predict_start(self, log_x_t, cond_audio, style_emb, t, sampling=False, cond_chord_for_cfg=None):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if sampling==False:
            feature = self.encoder(cond_audio)
            if cond_chord_for_cfg is not None:
                feature_for_cfg = self.encoder(cond_chord_for_cfg)
            else:
                feature_for_cfg = None
            out = self.decoder(x_t, feature, feature_for_cfg, t, style_emb)
        if sampling==True:
            if t[0].item() == self.num_timesteps-1:
                feature = self.encoder(cond_audio)
                if cond_chord_for_cfg is not None:
                    feature_for_cfg = self.encoder(cond_chord_for_cfg)
                else:
                    feature_for_cfg = None
                out = self.decoder(x_t, feature, t, style_emb)
                self.saved_encoder_features = feature
                self.saved_encoder_features_for_cfg = feature_for_cfg
            else:
                assert self.saved_encoder_features is not None
                out = self.decoder(x_t, self.saved_encoder_features, t, style_emb, cfg_feature=self.saved_encoder_features_for_cfg)
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

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt-1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) # [B, 1, L]
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.label_seq_len) # B x 1 x 27544 vector filled with -69.08

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)

        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:] # B x 5 x L 
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1) # B x 5 x 1
        
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector # replace the position having masked tokens's probability to only select mask token afterwards

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1) # B x 6 x 27544
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True) # B x 1 x 27544
        q = q - q_log_sum_exp # normalize q
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_posterior_reverse(self, log_x_start, log_x_t, t):            # p_theta(xt-1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) # [B, 1, L]
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.label_seq_len) # B x 1 x 27544 vector filled with -69.08

        log_qt = self.q_pred_reverse(log_x_t, t)

        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:] # B x 5 x L 
        log_cumprod_ctr = extract(self.log_cumprod_ctr, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ctr.expand(-1, self.num_classes-1, -1) # B x 5 x 1
        
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector # replace the position having masked tokens's probability to only select mask token afterwards

        log_qt_one_timestep = self.q_pred_one_timestep_reverse(log_x_t, t)  
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ctr = extract(self.log_ctr, t, log_x_start.shape)
        ct_vector = log_ctr.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1) # B x 6 x 27544
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True) # B x 1 x 27544
        q = q - q_log_sum_exp # normalize q
        log_EV_xtmin_given_xt_given_xstart = self.q_pred_reverse(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)


    def p_pred(self, log_x, cond_audio, t, style_emb=None, target=None, mask=None, repaint=1, cfg_features=None):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        # p_pred for sampling
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, cond_audio, style_emb, t, sampling=True, cond_chord_for_cfg=cfg_features)
            if target is not None and mask is not None:
                log_x_recon = log_x_recon * (1 - mask) + target * mask
            if self.reverse_sampling:
                log_model_pred = self.q_posterior_reverse(log_x_start=log_x_recon, log_x_t=log_x, t=t)
            elif not self.reverse_sampling:
                log_model_pred = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        else:
            raise ValueError
        return log_model_pred
    
    def p_pred_inpainting(self, log_x, cond_audio, t, style_emb=None, prev_log_x=None, mask=None, repaint:int = 10):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        # p_pred for sampling
        for r in range(repaint):
            if self.parametrization == 'x0':
                log_x_recon = self.predict_start(log_x, cond_audio, t, style_emb, sampling=True)
                log_x_recon = log_x_recon * (1 - mask) + prev_log_x * mask
                if self.reverse_sampling:
                    log_model_pred = self.q_posterior_reverse(log_x_start=log_x_recon, log_x_t=log_x, t=t)
                elif not self.reverse_sampling:
                    log_model_pred = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
            else:
                raise ValueError
            if r < repaint - 1:
                log_model_pred = self.q_pred_one_timestep(log_model_pred, t)
        return log_model_pred
    

    @torch.no_grad()
    def p_sample(self, log_x, cond_audio, t, style_emb=None, cond_drop_prob=None, cfg_features=None):               # sample q(xt-1) for next step from xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, cond_audio, t, style_emb, cfg_features=cfg_features) # onset, reonset -> 2, 4 (0~4)
        if self.onset_suppress: # suppress onset, offsets when sampling
            model_log_prob[:, 1, :] = model_log_prob[:, 1, :] * (1 + self.onset_suppress)
            model_log_prob[:, 2, :] = model_log_prob[:, 2, :] * (1 + self.onset_suppress)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_inpainting(self, log_x, cond_audio, t, prev_log_x, inpainting_ratio, shape, style_emb=None, cond_drop_prob=None):               # sample q(xt-1) for next step from xt, actually is p(xt-1|xt)
        prev_log_x_reshape = prev_log_x.reshape((shape[0], -1, shape[1], shape[2])) # B x class x T x 88
        source = prev_log_x_reshape[:, :, int(shape[1]*inpainting_ratio):, :] 
        empty = torch.zeros(prev_log_x_reshape.shape, dtype=prev_log_x_reshape.dtype, device=prev_log_x_reshape.device)[:, :, :int(shape[1]*(1-inpainting_ratio)), :]
        mask = torch.cat([torch.ones_like(source), torch.zeros_like(empty)], dim=-2) # B x class x T x 88
        prev_log_x = torch.cat([source, empty], dim=-2)
        mask = mask.reshape((mask.shape[0], mask.shape[1], -1)) # B x class x T*88
        prev_log_x = prev_log_x.reshape((mask.shape[0], mask.shape[1], -1)) # B x class x T*88
        model_log_prob = self.p_pred_inpainting(log_x, cond_audio, t, style_emb, prev_log_x, mask, self.repaint) # onset, reonset -> 2, 4 (0~4)
        if self.onset_suppress: # suppress onset, offsets when sampling
            model_log_prob[:, 0, :] = model_log_prob[:, 0, :] * (1 + self.onset_suppress)
            # model_log_prob[:, 2, :] = model_log_prob[:, 2, :] * (1 + self.onset_suppress)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits) # gaussian distribution of mean 0 and variance 1
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_audio, style_emb=None, is_train=True, cfg_features=None):                       # get the KL loss
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')

        # DALLE encodes image to 2887+1(mask) tokens (=self.num_classes)
        log_x_start = index_to_log_onehot(x_start, self.num_classes) 
        # x0 -> xt
        log_xt = self.q_sample(log_x_start=log_x_start, t=t) # B x 6 x L , discretely-noised log one-hot vector
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_audio, style_emb=style_emb, t=t, cond_chord_for_cfg=cfg_features)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon) # start prediction
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob) # xt-1 prediction
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes-1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss


    def forward(
            self, 
            label,
            features, 
            style_emb=None,
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            is_train=True,
            cfg_features=None,
            **kwargs):
        """
        input shape : B x T*88 x H+1 
        """
        # now we get cond_emb and sample_image
        ######## Diffusion Training ########
        if is_train == True:
            log_model_prob, loss = self._train_loss(label, features, style_emb, cfg_features=cfg_features)
            loss = loss.sum()/(label.size()[0] * label.size()[1]) # TODO: check if input.size()[0] and [1] is right

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss 
        return out


    def sample(
            self,
            audio,
            style_emb=None,
            return_logits = False,
            visualize_denoising = False):

        batch_size = audio.shape[0]
        device = self.log_at.device
        start_step = 0

        cond_audio = audio # no condition for now
        labels = [] # for plotting labels getting denoised

        if start_step == 0:
            # use full mask sample
            if self.fully_masked:
                zero_logits = torch.zeros((batch_size, self.num_classes-1, self.shape),device=device)
                one_logits = torch.ones((batch_size, 1, self.shape),device=device)
                mask_logits = torch.cat((zero_logits, one_logits), dim=1) # B, class, token_len
                log_z = torch.log(mask_logits)
            elif not self.fully_masked:
                transition_prob_logits = torch.ones((batch_size, self.num_classes-1, self.shape),device=device) * torch.exp(self.log_cumprod_bt[-2])
                mask_prob_logits = torch.ones((batch_size, 1, self.shape),device=device) * (1 - (torch.exp(self.log_cumprod_bt[-2]) * (self.num_classes-1)))
                final_logits = torch.cat((transition_prob_logits, mask_prob_logits), dim=1)
                log_z = torch.log(final_logits)

            start_step = self.num_timesteps
            labels.append(log_z.argmax(1).cpu().numpy())
            with torch.no_grad():
                for diffusion_index in range(start_step-1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long) # make batch tensor filled with value 't'
                    log_z = self.p_sample(log_z, cond_audio, t, style_emb, cfg_features=cfg_features)     # log_z is log_onehot
                    if visualize_denoising:
                        labels.append(log_z.argmax(1).cpu().numpy())

        else: 
            raise ValueError("Not implemented yet")
        

        label_token = log_onehot_to_index(log_z)
        
        output = {'label_token': label_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output, labels

    def sample_inpainting(
            self,
            leadsheet: Tensor, # leadsheet
            style_emb: Tensor,
            prev_piano,
            inpainting_ratio: float,
            shape,
            return_logits = False,
            visualize_denoising = False):
        batch_size = leadsheet.shape[0]
        device = self.log_at.device
        cond_leadsheet = leadsheet # no condition for now
        prev_log_z = index_to_log_onehot(prev_piano.to(torch.int64), self.num_classes)
        labels = [] # for plotting labels getting denoised
        start_step = 0

        if start_step == 0:
            # use full mask sample
            zero_logits = torch.zeros((batch_size, self.num_classes-1, self.shape),device=device)
            one_logits = torch.ones((batch_size, 1, self.shape),device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1) # B, class, token_len
            log_z = torch.log(mask_logits)

            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in range(start_step-1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long) # make batch tensor filled with value 't'
                    log_z = self.p_sample_inpainting(log_z, cond_leadsheet, t, prev_log_z, inpainting_ratio, shape, style_emb)     # log_z is log_onehot
                    if visualize_denoising:
                        labels.append(log_z.argmax(1).cpu().numpy())
        else: 
            raise ValueError("Not implemented yet")
        

        label_token = log_onehot_to_index(log_z)
        
        output = {'label_token': label_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output, labels
