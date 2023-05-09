import os
from logging import log

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from config import configs
from learner import Learner
from scheduler import CustomScheduler
from dataset import get_translation_dataloaders
from callbacks import CheckpointSaver, MoveToDeviceCallback, TrackLoss, TrackExample, TrackBleu
from architectures.machine_translation_transformer import MachineTranslationTransformer

# Initialize configuration
import wandb
from config import configs

config_name='unofficial_single_gpu_config' # MODIFY THIS TO CHANGE CONFIGURATION
config = configs[config_name]
wandb.init(config=config,project="grammar-as-a-foreign-language")

# Configure Logging
from utils.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Seed the Random Number Generators
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


class TrainingApp:
    def __init__(self):

        log.info('----- Training Started -----')

        # Device handling
        if wandb.config.DEVICE=='gpu':
            if not torch.cuda.is_available():
                raise ValueError('GPU is not available.')
            self.device = 'cuda'
            log.info(f'Device name is {torch.cuda.get_device_name()}')
        else:
            log.info(f'Device name is CPU')
            self.device='cpu'
        self.init()

    def init(self):

        model = MachineTranslationTransformer(
            d_model=wandb.config.D_MODEL,
            n_blocks=wandb.config.N_BLOCKS,
            src_vocab_size=wandb.config.VOCAB_SIZE,
            trg_vocab_size=wandb.config.VOCAB_SIZE,
            n_heads=wandb.config.N_HEADS,
            d_ff=wandb.config.D_FF,
            dropout_proba=wandb.config.DROPOUT_PROBA
        )

        from torchinfo import summary
        batch_size = config['BATCH_SIZE']
        max_seq_len = config['MAX_SEQ_LEN']
        summary(model, [(batch_size, max_seq_len), (batch_size, max_seq_len)], dtypes=[torch.long, torch.long])

        loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

        optimizer = optim.Adam(model.parameters(), betas=wandb.config.BETAS, eps=wandb.config.EPS)
        scheduler=CustomScheduler(optimizer, wandb.config.D_MODEL, wandb.config.N_WARMUP_STEPS)
        
        # # The above scheduler's efficiency is highly influenced by dataset and batch size,
        # # alternatively you can use the below configuration, which also works much better for overfit configs.
        # optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=wandb.config.BETAS, eps=wandb.config.EPS)
        # scheduler=optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, epochs=wandb.config.EPOCHS, steps_per_epoch=len(train_dl), pct_start=0.3)
        
        cbs = [
            CheckpointSaver(epoch_cnt=wandb.config.MODEL_SAVE_EPOCH_CNT,),
            MoveToDeviceCallback(),
            TrackLoss(),
            TrackBleu(),
            TrackExample(),
            ]
        
        wandb.watch(model, log_freq=1000)

        train_dl, val_dl = get_translation_dataloaders(
            dataset_size=wandb.config.DATASET_SIZE,
            vocab_size=wandb.config.VOCAB_SIZE,
            tokenizer_save_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME,'tokenizer.json'),
            tokenizer_type=wandb.config.TOKENIZER_TYPE,
            batch_size=wandb.config.BATCH_SIZE,
            report_summary=True,
            max_seq_len=wandb.config.MAX_SEQ_LEN,
            test_proportion=wandb.config.TEST_PROPORTION,
        )

        learner = Learner(model,
                          train_dl,
                          val_dl,
                          loss_func,
                          cbs,
                          optimizer,
                          scheduler,
                          self.device)
        self.model, self.learner, self.dl, self.cbs = model, learner, [train_dl, val_dl], cbs
        return self.model, self.learner, self.dl, self.cbs

    def main(self):
        # self.init()
        self.learner.fit(wandb.config.EPOCHS)


if __name__ == "__main__":
    TrainingApp().main()
