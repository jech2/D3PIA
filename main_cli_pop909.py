# main.py
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from transcription.module import D3RM
from transcription.datamodule import MAESTRO_V3_DataModule, Pop1k7_DataModule, POP909_DataModule

# simple demo classes for your convenience
import argparse, os, glob, datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import colored
import torch
import wandb

torch.backends.cudnn.benchmark = True

class D3RMCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:

        parser.add_argument( "-d", "--debug", type=bool, default=False, help="enable post-mortem debugging",)
        parser.add_argument("--wandb", type=bool, default=False, help="wandb online/offline",)

    def before_fit(self):
        if not self.config.fit.ckpt_path: # if not resuming from checkpoint
            self.now = id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        elif self.config.fit.ckpt_path: # resuming from checkpoint
            ckpt_date = os.path.basename(os.path.dirname(self.config.fit.ckpt_path))
            print(colored("Continue training from checkpoint: ", "red", attrs=['bold']), ckpt_date)
            self.now = id = ckpt_date
        # Logging
        # wandb.login(key="99aeb92834216fc5de43eb5235fbe169caf149c0")
        wandb_logger = WandbLogger(save_dir=f"./logs/{self.now}",
                                   name=self.now,
                                   project="PianoArrDiffusion",
                                   offline=(not self.config.fit.wandb),
                                   entity="habang",
                                   id=id)

        # Model checkpoint (automatically called after validation)
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/{self.now}',
            monitor='metric_note_with_offsets_f1',
            mode='max',
            save_top_k=7,
            save_last=True,
            verbose=True,
            # save_on_train_epoch_end=True,
            filename='{step:07}-{metric_note_with_offsets_f1:.4f}') # python recognized '/', '-' as '_'

        self.trainer.logger = wandb_logger
        self.trainer.callbacks.append(model_checkpoint_callback)
        self.config.fit.model.test_save_path = f"./results/{self.now}"
        print(colored("Test results will be saved in: ", "green", attrs=['bold']), self.config.fit.model.test_save_path)

        # if self.trainer.ckpt_path:
        #     checkpoint = torch.load(self.trainer.ckpt_path, map_location="cpu")
        #     model_keys = set(self.model.state_dict().keys())
        #     filtered_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_keys}
        #     self.model.load_state_dict(filtered_state_dict, strict=False)

        
    # def before_instantiate_classes(self) -> None:
    

def cli_main():
    # cli = D3RMCLI(D3RM, MAESTRO_V3_DataModule,
    #             #   save_config_kwargs={"overwrite": True}, #  save_config_callback=None # when using wandb, saving config leads to conflicts.
    #               )
    data_module = POP909_DataModule
    cli = D3RMCLI(D3RM, data_module,
                  save_config_kwargs={"overwrite": True},
                  )

if __name__ == "__main__":
    cli_main()
