# D3RM : A Discrete Denoising Diffusion Refinement Model for Piano Transcription

This is the source code of D3RM paper accepted in ICASSP 2025.
Regarding the reproducement of the paper, please let me know your concerns and feel free to comment them in the `Issues` part.

<img src="https://github.com/hanshounsu/d3rm/blob/main/images/Absorbing%20state.png?raw=true" height="300"/> <img src="https://github.com/hanshounsu/d3rm/blob/main/images/Model%20architecture.png" height="300"/>


## Installation

```shell
git clone https://github.com/hanshounsu/d3rm.git
pip -r install requirements.txt
```
Current project is based on pytorch-lightning 2.5.0.

## Model Download
* Pretrained NAR-HC baseline model [[link](https://drive.google.com/file/d/1puA0CkXGioXs9OrS1w-AvwN71yi2cxae/view?usp=sharing)]
* Pretrained D3RM model [[link]()] 

Place the pretrained D3RM model in ./checkpoints/pretrained/

## Download MAESTRO
Download here [[link](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip)]

Place the dataset folder inside ./data

## Training the model
```shell
python3 main_cli.py fit -c ./configs/D3RM_cli.yaml
```

## Inference
```shell
python3 main_cli.py test -c ./logs/{exp_id}/config.yaml
# you need to update ckpt_path in ckpt file
```

<!-- ## Acknowledge -->

<!-- 1. We had a consistent design of [FunASR](https://github.com/alibaba/FunASR), including dataloader, model definition and so on. -->
<!-- 2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation. -->
<!-- 4. We borrowed the design of model architecture from [Enocdec](https://github.com/facebookresearch/encodec) and [Enocdec_Trainner](https://github.com/Mikxox/EnCodec_Trainer). -->

## License
This project is licensed under [The MIT License](https://opensource.org/licenses/MIT). 

## Citations

``` bibtex
@misc{hskim2023d3rm,
      title={D3RM : A Discrete Denoising Diffusion Refinement Model for Piano Transcription},
      author={Hounsu Kim, Taegyun Kwon, Juhan Nam},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.Sound}
}
```

