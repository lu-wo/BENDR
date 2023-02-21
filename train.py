import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from model import create_bendr
from config import params
import time
import os, sys
from datasets_single_load import MultiTaskDataModule
import logging


def main():
    # Create model directory and Logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f'reports/logs/{run_id}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # change std output to file in log_dir 
    sys.stderr = open(f"{log_dir}/stderr.log", "w")
    sys.stdout = open(f"{log_dir}/stdout.log", "w")    
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")

    data_module = MultiTaskDataModule(
            root_dir=params['root_dir'], 
            window_len=params['window_len'],
            batch_size=params['batch_size']
            )
    logging.info("Created data module.")
    model = create_bendr(params)
    logging.info("Created model.")

    # Run the model
    tb_logger = TensorBoardLogger("./reports/logs/",
                                  name=f"{run_id}"
                                  )
    tb_logger.log_hyperparams(params)
    # wandb_logger = WandbLogger(project=f"bendr-pretraining",
    #                             name=f"{run_id}",
    #                             save_dir="./reports/logs/"
    #                     )
    logging.info("Created logger.")

    trainer = pl.Trainer(
        accelerator='gpu', # gpu or cpu 
        devices=params['gpus'],  # -1: use all available gpus, for cpu e.g. 4
        strategy='ddp', # ddp

        enable_progress_bar=True,  # disable progress bar
        # precision=16, # 16 bit float precision for training
        logger = [tb_logger],
        log_every_n_steps=10, # every n-th batch is logged

        max_epochs=params['epochs'],
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15),  # early stopping
            ModelCheckpoint(
                log_dir, 
                monitor='val_loss', 
                save_top_k=1),
            ],
        auto_lr_find=True  # automatically find learning rate
    )
    
    logging.info("Start training.")
    trainer.fit(model, data_module)  # train the model
    logging.info("Finished training.")

    logging.info("Start testing.")
    trainer.test(model, data_module)  # test the model
    logging.info("Finished testing.")

    logging.info("\n--- Finished logging.")

    
if __name__ == "__main__":
    #load sweep files
    # with open('sweep.yaml') as f:
    #     sweep_params = yaml.load(f, Loader=yaml.FullLoader)
    # print(sweep_params)
    # sweep_id = wandb.sweep(sweep_params)
    # wandb.agent(sweep_id, function=main)
    main()
