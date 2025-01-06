import torch as th
from torch import nn

from transcription.diffusion.natten_diffusion import NeighborhoodAttention2D_diffusion, NeighborhoodAttention1D_diffusion, NeighborhoodCrossAttention2D_diffusion, NeighborhoodAttention2D_diffusion_encoder
from typing import List


class TransModel(nn.Module):
    def __init__(self,
                label_embed_dim: int,
                lstm_dim: int,
                n_layers: int,
                window: List[int],
                dilation: List[int],
                condition_method: str,
                diffusion_step: int,
                timestep_type: str,
                natten_direction: str,
                spatial_size: List[int],
                num_state: int = 5,
                classifier_free_guidance: bool = False,
                features_embed_dim: int = 128,
                num_class: int = 4,
                 ):
        super().__init__()
        # self.hidden_per_pitch = config.hidden_per_pitch
        self.label_embed_dim = label_embed_dim
        self.features_embed_dim = features_embed_dim 
        self.n_unit = lstm_dim
        self.n_layers = n_layers
        self.window = window
        self.cross_condition = condition_method
        self.diffusion_step = diffusion_step
        self.timestep_type = timestep_type
        self.natten_direction = natten_direction
        self.spatial_size = spatial_size
        self.dilation = dilation
        self.num_state = num_state
        self.classifier_free_guidance = classifier_free_guidance
        self.num_class = num_class

        # self.trans_model = NATTEN(config.hidden_per_pitch)
        self.trans_model = LSTM_NATTEN((label_embed_dim+features_embed_dim), 
                                        timestep_type=self.timestep_type,
                                        diffusion_step=self.diffusion_step,
                                        natten_direction=self.natten_direction,
                                        spatial_size=self.spatial_size,
                                        dilation = self.dilation,
                                        window=self.window,
                                        n_unit=self.n_unit,
                                        n_layers=self.n_layers,
                                        cross_condition=self.cross_condition,
                                        )
        self.output = nn.Linear(self.n_unit, self.num_class) 
        self.label_emb = nn.Embedding(num_state + 1, # +1 is for mask
                                      label_embed_dim)
        # classifier_free_guidance
        if classifier_free_guidance:
            self.use_cfg = True
            # self.cond_scale = cond_scale
            # self.cond_drop_prob = cond_drop_prob
            self.null_features_emb = nn.Parameter(th.randn(1, 128)) # null embedding for cfg
        else:
            self.use_cfg = False
        

    def forward(self, label, feature, t, cond_drop_prob=None):
        # feature (=cond_emb) : B x T*88 x H
        # label (=x_t) : B x T*88 x 1
        # feature are concatanated with label as input to model
        batch = label.shape[0]
        # feature = feature.transpose(-2, -1) # B x T x 88 x H(=128)
        feature = feature.reshape(feature.shape[0], -1, feature.shape[-1]) # B x T*88 x H
        if self.use_cfg == True:
            if cond_drop_prob != 1: cond_drop_prob = self.cond_drop_prob
            keep_mask = th.zeros((batch,1,1), device=self.devices).float().uniform_(0, 1) < (1 - cond_drop_prob)
            null_cond_emb = self.null_feature_emb.repeat(label.shape[0], label.shape[1], 1) # B x T*88 x label_embed_dim
            feature = th.where(keep_mask, feature, null_cond_emb) 
        
        assert label.max() <= self.label_emb.num_embeddings - 1 and label.min() >= 0, f"Label out of range: {label.max()} {label.min()} {self.label_emb.num_embeddings}"
        label_emb = self.label_emb(label) # B x T*88 x label_embed_dim
        input_feature = th.cat((label_emb, feature), dim=-1) 
        if self.cross_condition == 'self':
            x = self.trans_model(input_feature, None, t)
        elif self.cross_condition == 'cross' or self.cross_condition == 'self_cross':
            x = self.trans_model(input_feature, feature, t)
        out = self.output(x)
        return out.reshape(x.shape[0], x.shape[1]*x.shape[2], -1).permute(0, 2, 1) # B x 5 x T*88


class NATTEN(nn.Module):
    def __init__(self, hidden_per_pitch, window=25, n_unit=24, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers

        self.linear = nn.Sequential(nn.Linear(hidden_per_pitch+5, n_unit),
                                    nn.ReLU())
        self.na = nn.Sequential(*([NeighborhoodAttention2D_diffusion(n_unit, 4, window)]* n_layers))

    def forward(self, x):
        # x: B x T x 88 x H+5
        cat = self.linear(x)
        na_out = self.na(cat) # B x T x 88 x N
        return na_out
        
class LSTM_NATTEN(nn.Module):
    def __init__(self,
                 hidden,
                 timestep_type,
                 diffusion_step,
                 natten_direction,
                 spatial_size,
                 dilation: List[int],
                 window=25,
                 n_unit=24,
                 n_head=4,
                 n_layers=2,
                 cross_condition=False,
                 type='decoder',
        ):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_unit = n_unit
        self.timestep_type = timestep_type
        self.diffusion_step = diffusion_step
        self.lstm = nn.LSTM(hidden, n_unit//2, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.natten_dir = natten_direction
        self.spatial_size = spatial_size
        self.cross_condition = cross_condition
        self.dilation = dilation
        self.module_type = type
        
        if self.natten_dir == '1d':
            self.na_1t = nn.ModuleList([NeighborhoodAttention1D_diffusion(n_unit, 1, window[0],
                                                                        spatial_size=self.spatial_size,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                    
                                    ] * n_layers)
            self.na_1f = nn.ModuleList([NeighborhoodAttention1D_diffusion(n_unit, 1, window[1],
                                                                        spatial_size=self.spatial_size,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type)
                                    ] * n_layers)
        elif self.natten_dir == '2d' and cross_condition == 'self':
            self.na = []
            for i in range(n_layers):
                self.na.append(NeighborhoodAttention2D_diffusion(n_unit, 4, window[i],
                                                                    diffusion_step=self.diffusion_step,
                                                                    dilation=self.dilation[i],
                                                                    timestep_type=self.timestep_type))
            self.na = nn.ModuleList(self.na)
        elif self.natten_dir == '2d' and cross_condition == 'cross':
            self.na = []
            for i in range(n_layers):
                self.na.append(NeighborhoodCrossAttention2D_diffusion(n_unit, 4, window[i],
                                                                    diffusion_step=self.diffusion_step,
                                                                    dilation=self.dilation[i],
                                                                    timestep_type=self.timestep_type))
            self.na = nn.ModuleList(self.na)
        elif self.natten_dir == '2d' and cross_condition == 'self_cross':
            self.na = nn.ModuleList([NeighborhoodAttention2D_diffusion(n_unit, 4, window,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                    NeighborhoodCrossAttention2D_diffusion(n_unit, 4, window,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                     ] * (n_layers // 2))


    def forward(self, x, cond=None, t=None, label_emb=None):
        """
        x shape : B x T*88 x n_unit
        cond shape : B x T*88 x feature_embed_dim(=128)
        """
        
        if self.module_type == 'encoder' and label_emb is not None:
            x = x.long()
            x = label_emb(x)
        
        B = x.shape[0]
        H = x.shape[-1]
        if self.module_type == 'encoder':
            T = x.shape[1]
        elif self.module_type == 'decoder':
            T = x.shape[1]//88
            x = x.reshape(B, T, 88, H) # B x T x 88 x H
        x = x.permute(0, 2, 1, 3).reshape(B*88, T, H) # B*88 x T x H
        x, c = self.lstm(x)
        # if use natten1d
        if self.natten_dir == '1d':
            for layer_1t, layer_1f in zip(self.na_1t, self.na_1f):
                # x_res = x
                x, t = layer_1t(x, t)
                # x = x + x_res
                x = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3).reshape(B*T, 88, self.n_unit)
                # x_res = x
                x, t = layer_1f(x, t)
                x = x.reshape(B, T, 88, self.n_unit).permute(0,2,1,3).reshape(B*88, T, self.n_unit)
                # x = x + x_res
            x = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3) # B x T x 88 x H
        
        # if use natten2d 
        # elif self.natten_dir == '2d' and self.cross_condition == 'cross':
        elif self.natten_dir == '2d':
            x = x.reshape(B, 88, T, -1).permute(0,2,1,3) # B x T x 88 x H
            for layers in self.na:
                x_res = x
                x, _, t = layers(x, cond, t)
                x = x + x_res

        return x