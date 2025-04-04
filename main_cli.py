# main.py
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from d3pia.module import D3RM
from d3pia.datamodule import POP909_DataModule

# simple demo classes for your convenience
import os, datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import colored
import torch

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
        wandb_logger = WandbLogger(save_dir=f"./logs/{self.now}",
                                   name=self.now,
                                   project="D3PIA",
                                   offline=(not self.config.fit.wandb),
                                   id=id)

        model_checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/{self.now}',
            monitor='metric_note_with_offsets_f1',
            mode='max',
            save_top_k=7,
            save_last=True,
            verbose=True,
            filename='{step:07}-{metric_note_with_offsets_f1:.4f}') 

        self.trainer.logger = wandb_logger
        self.trainer.callbacks.append(model_checkpoint_callback)
        self.config.fit.model.test_save_path = f"./results/{self.now}"
        print(colored("Test results will be saved in: ", "green", attrs=['bold']), self.config.fit.model.test_save_path)    

def cli_main():
    cli = D3RMCLI(D3RM, POP909_DataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()