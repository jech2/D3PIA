import torch
from transcription.polydis.ptvae import TextureEncoder

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