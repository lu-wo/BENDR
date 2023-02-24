import torch 
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os, sys 
import yaml
import numpy as np 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import logging 
import time

from finetuning_config import params
from model import create_bendr


# Create model directory and Logger
run_id = time.strftime("%Y%m%d-%H%M%S")
log_dir = f'reports/logs/{run_id}_finetuning'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# change std output to file in log_dir 
sys.stderr = open(f"{log_dir}/stderr.log", "w")
sys.stdout = open(f"{log_dir}/stdout.log", "w")    
# Create logging file
logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
logging.info("Started logging.")

params_path = os.path.join(params['bendr_dir'], 'version_0', 'hparams.yaml')
# load yaml file as dict 
with open(params_path, 'r') as f:
    params = yaml.unsafe_load(f)

# Load and split data
np.random.seed(42)
np_file = np.load(params['data_dir'])
X = np_file['EEG']
y = np_file['labels']

# split data based on first column id 
ids = np.unique(y[:, 0])
np.random.shuffle(ids)
train_ids = ids[:int(len(ids)*0.8)]
val_ids = ids[int(len(ids)*0.8):int(len(ids)*0.9)]
test_ids = ids[int(len(ids)*0.9):]

# split data
X_train = X[np.isin(y[:, 0], train_ids)]
y_train = y[np.isin(y[:, 0], train_ids)]
X_val = X[np.isin(y[:, 0], val_ids)]
y_val = y[np.isin(y[:, 0], val_ids)]
X_test = X[np.isin(y[:, 0], test_ids)]
y_test = y[np.isin(y[:, 0], test_ids)]

# create dataloaders
batch_size = 64
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# load model
model = create_bendr(params)
# make not trainable 
for param in model.parameters():
    param.requires_grad = False

# get file that ends with .ckpt 
ckpt_path = [f for f in os.listdir(params['bendr_dir']) if f.endswith('.ckpt')][0]
ckpt_path = os.path.join(params['bendr_dir'], ckpt_path)
# load weights
model.load_state_dict(torch.load(ckpt_path)['state_dict'])

# build finetuning MLP
def build_mlp(hidden_layers, output_logits, hidden_size, dropout):
    layers = []
    for i in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(hidden_size, output_logits))
    return torch.nn.Sequential(*layers)

# create lightning model
class FinetuningModel(pl.LightningModule):
    def __init__(self, model, hidden_layers, output_logits, hidden_size, dropout, lr, loss):
        super().__init__()
        self.model = model
        self.mlp = build_mlp(hidden_layers, output_logits, hidden_size, dropout)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss() if loss == 'cross_entropy' else nn.MSELoss()
    
    def forward(self, x):
        x = self.model.encoder(x) # pass through BENDRCNN
        x = self.mlp(x)
        return x
    
    def loss(self, y_hat, y):
        return self.loss(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

tb_logger = TensorBoardLogger("./reports/logs/",
                                  name=f"{run_id}_finetuning"
                                  )
tb_logger.log_hyperparams(params)
wandb_logger = WandbLogger(entity="deepseg",
                            project=f"bendr-finetuning",
                            name=f"{run_id}",
                            save_dir="./reports/logs/"
                    )

# create finetuning model
model = FinetuningModel(model, params['hidden_layers'], params['output_logits'], params['hidden_size'], params['dropout'], params['learning_rate'], params['loss'])
# create trainer
trainer = Trainer(
    gpus=1, 
    max_epochs=params['epochs'], 
    progress_bar_refresh_rate=20,
    logger=[tb_logger, wandb_logger],
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=True),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, verbose=True)
    ]
    )
# train
trainer.fit(model, train_loader, val_loader)

# test
trainer.test(model, test_dataloaders=test_loader)
