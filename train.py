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
from datasets import MultiTaskDataModule
import logging


def main():
    #wandb.init()
    #wandb.init(config=default_params)
    #params = wandb.config

    # Create model directory and Logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f'reports/logs/{run_id}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # change std output to file in log_dir 
    sys.stdout = open(f"{log_dir}/stdout.log", "w")
    
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset
    # Log hyperparameters and config file
    logging.info(params)

    data_module = MultiTaskDataModule(
            root_dir=params['root_dir'], 
            window_len=params['window_len'],
            buffer_size=20, 
            batch_size=params['batch_size']
        ) #used for sweepiong
    # data_module.setup(stage="fit")

    logging.info("Created data module.")

    # train_loader = data_module.train_dataloader()
    # val_loader = data_module.val_dataloader()
    # test_loader = data_module.test_dataloader()

    # for batch in train_loader:
    #     print(batch.shape)
    #     print(f"requires grad {batch.requires_grad}")

    # for batch in val_loader:
    #     print(batch.shape)
    #     print(f"requires grad {batch.requires_grad}")

    # for batch in test_loader:
    #     print(batch.shape)
    #     print(f"requires grad {batch.requires_grad}")

    
    # Create model based on config.py and hyperparameters.py settings
    # changed to include model factory
    #model = get_model(params[config['model']], config['model'])
    model = create_bendr(params)
    logging.info("Created model.")
    # print model summary
    # summary(model, (config['input_height'], config['input_width']))

    # Log hyperparameters and config file
    #log_params(log_dir)

    # Run the model
    tb_logger = TensorBoardLogger("./reports/logs/",
                                  name=f"{run_id}"
                                  )
    #tb_logger.log_hyperparams(params[config['model']])  # log hyperparameters
    tb_logger.log_hyperparams(params)
    # wandb_logger = WandbLogger(project=f"Bendr",
    #                            entity="Pretrain",
    #                            # save_dir=f"reports/logs/{run_id}_{config['model']}_" \
    #                            #          f"{config['dataset']}_{params[config['model']]['word_length']}",
    #                            save_dir=f"reports/logs/{run_id}_{config['model']}_" \
    #                                     f"{config['dataset']}_{params['word_length']}_k_{params['k']}_M_{params['M']}",
    #                            # id=f"{run_id}_{config['model']}_" \
    #                            #    f"{config['dataset']}_{params[config['model']]['word_length']}"
    #                            id=f"{run_id}_{config['model']}_" \
    #                               f"{config['dataset']}_{params['word_length']}_k_{params['k']}_M_{params['M']}"
    #                            )
    #wandb_logger.experiment.config["Model"] = config['model']
    #wandb_logger.experiment.config.update(params[config['model']])
    #wandb_logger.experiment.config.update(params)
    logging.info("Created logger.")

    trainer = pl.Trainer(
        accelerator="gpu",  # cpu or gpu
        devices=-1,  # -1: use all available gpus, for cpu e.g. 4
        enable_progress_bar=True,  # disable progress bar
        # show progress bar every 500 iterations
        # precision=16, # 16 bit float precision for training
        #logger=[tb_logger, wandb_logger],  # log to tensorboard and wandb
        logger = [tb_logger],

        #max_epochs=params[config['model']]['epochs'],  # max number of epochs
        max_epochs=params['epochs'],
        callbacks=[
        #EarlyStopping(monitor="val_loss", patience=5),  # early stopping
        # ModelSummary(max_depth=1),  # model summary
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

    logging.info("Finished logging.")

    
if __name__ == "__main__":
    #load sweep files
    # with open('sweep.yaml') as f:
    #     sweep_params = yaml.load(f, Loader=yaml.FullLoader)
    # print(sweep_params)
    # sweep_id = wandb.sweep(sweep_params)
    # wandb.agent(sweep_id, function=main)

    main()

