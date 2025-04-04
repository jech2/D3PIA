import torch
from d3pia.polydis.ptvae import TextureEncoder

from torch import nn
from torch.distributions import Normal


class ChordEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(ChordEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, x):
        x = self.gru(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        dist = Normal(mu, var)
        return dist


def load_pretrained_style_enc(fpath, emb_size, hidden_dim, z_dim, num_channel):
    txt_enc = TextureEncoder(emb_size, hidden_dim, z_dim, num_channel)
    checkpoint = torch.load(fpath)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    from collections import OrderedDict

    enc_chkpt = OrderedDict()
    for k, v in checkpoint.items():
        part = k.split(".")[0]
        name = ".".join(k.split(".")[1:])
        if part == "rhy_encoder":
            enc_chkpt[name] = v
    txt_enc.load_state_dict(enc_chkpt)
    return txt_enc

def load_pretrained_chd_enc(
    fpath, input_dim, z_input_dim, hidden_dim, z_dim, n_step
):
    chord_enc = ChordEncoder(input_dim, hidden_dim, z_dim)
    checkpoint = torch.load(fpath)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    from collections import OrderedDict

    enc_chkpt = OrderedDict()
    dec_chkpt = OrderedDict()
    for k, v in checkpoint.items():
        part = k.split(".")[0]
        name = ".".join(k.split(".")[1:])
        if part == "chord_enc":
            enc_chkpt[name] = v
    chord_enc.load_state_dict(enc_chkpt)
    return chord_enc
