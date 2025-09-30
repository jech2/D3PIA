# D3PIA: A Discrete Denoising Diffusion Model for Piano Accompaniment Generation

This is the official repository for our paper: **“D3PIA: A Discrete Denoising Diffusion Model for Piano Accompaniment Generation”** (submitted to ICASSP 2026).


We propose a discrete diffusion-based piano accompaniment generation model, **D3PIA**, leveraging the locally aligned structure of musical accompaniments with the lead sheet in the piano roll representation. Our model incorporates Neighborhood Attention (NA) to both encode the lead sheet and condition it for predicting note states in the piano accompaniment, which enhances local contextual modeling by efficiently attending to nearby melody and chord conditions. 


<!-- <img src="./images/model_structure.png" height="500"/>  -->


## Links

- **Interactive Demo:** [Anonymized Demo Page](https://anonymous-researcher-mir.github.io/D3PIA/)
- [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset)



## Installation

Click **Download Repository**.

```shell
# install uv from https://docs.astral.sh/uv/getting-started/installation/
uv venv --python 3.10
uv sync
# install midisym (private midi library)
./install_midisym.sh
```

## Checkpoint Download
* Pretrained D3PIA model [link](https://drive.google.com/file/d/1dw_TbM6yn5EiO9_fqRjd9o3TO5kXxy0L/view?usp=drive_link)

Unzip the pre-trained D3PIA model and change the unzipped directory as ./checkpoints.


## Download POP909
Place the dataset folder inside ./data and split the dataset.
We utilized polyffusion's split and used the pre-processed MIDI of POP909 by [WholeSongGen](https://github.com/ZZWaang/whole-song-gen), which can be downloadable from [here](https://drive.google.com/file/d/1yc3NzUNpEMY8MnF4cNyQnCLjrXT2S0Jp/view?usp=sharing).

## Training the model
```shell
uv run python main_cli.py fit -c ./configs/D3PIA_default.yaml
```

## Inference
```shell
# first, you need to update ckpt_path of config file.
# you can choose checkpoint from ./logs/{exp_id} (last checkpoint) or ./checkpoints/{exp_id}.
uv run python main_cli.py test -c ./logs/{exp_id}/config.yaml

# convert midi file from npy
uv run python pr_mat_to_midi.py --wandb_id {exp_id}
```
By utilizing config files of logs/2025-02-21T00-01-12(with bridge) and logs/2025-03-04T19-05-17(without bridge), you can inference with pretrained checkpoints.
Note that we uploaded the D3PIA inference samples on whole POP909-valid (polyffusion split) and subjecitve evaluation samples in `./inference_samples` directory.

## License
This project is licensed under [The MIT License](https://opensource.org/licenses/MIT). 

<!-- ## Citations -->
