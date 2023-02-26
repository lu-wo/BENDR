import torch 
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import MeanSquaredError
import os, sys 
import yaml
import numpy as np 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import logging 
from dn3.utils import min_max_normalize_np
import time
from functools import partial

from finetuning_config import params as params
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

# Load and split data
np.random.seed(42)
np_file = np.load(params['data_dir'])
X = np_file['EEG'][:,:,:128]
global_min = np.min(X)
global_max = np.max(X)
y = np_file['labels']
# logging.info(f"y {y[:, 1]}")
logging.info("Loaded data.")
logging.info(f"X shape: {X.shape}")
logging.info(f"y shape: {y.shape}")

# normalize and add channel per sample in X 
X_norm = np.zeros((X.shape[0], X.shape[1], X.shape[2] + 1))
for i in range(len(X)):
    x_min = np.min(X[i])
    x_max = np.max(X[i])
    X[i] = min_max_normalize_np(X[i], x_min, x_max)
    val = (x_max - x_min) / (global_max - global_min)
    # create additional channel 
    const = np.ones((X[i].shape[0], 1)) * val
    if i == 0:
        logging.info(f"const shape: {const.shape}")
        logging.info(f"X[i] shape: {X[i].shape}")
    X_norm[i] = np.concatenate((X[i], const), axis=1)
X = X_norm
logging.info("Normalized data and added channel.")
logging.info(f"X shape: {X.shape}")

# split data based on first column id 
ids = np.unique(y[:, 0])
np.random.shuffle(ids)
train_ids = ids[:int(len(ids)*0.8)]
val_ids = ids[int(len(ids)*0.8):int(len(ids)*0.9)]
test_ids = ids[int(len(ids)*0.9):]

# split data
X_train = np.transpose(X[np.isin(y[:, 0], train_ids)], (0, 2, 1))
y_train = y[np.isin(y[:, 0], train_ids)][:, 1:]
X_val = np.transpose(X[np.isin(y[:, 0], val_ids)], (0, 2, 1))
y_val = y[np.isin(y[:, 0], val_ids)][:, 1:]
X_test = np.transpose(X[np.isin(y[:, 0], test_ids)], (0, 2, 1))
y_test = y[np.isin(y[:, 0], test_ids)][:, 1:]
# to each sample 

# create dataloaders
batch_size = params['batch_size']
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

# we need to normalize and add the additional channel 
# def collate_fn(global_min, global_max, batch):
#     X, y = zip(*batch)
#     X = torch.stack([torch.from_numpy(x) for x, y in batch])
#     x_min = torch.min(X)
#     x_max = torch.max(X)
#     X = min_max_normalize(X, x_min, x_max)
#     x_min = torch.min(X)
#     x_max = torch.max(X)
#     val = (x_max - x_min) / (global_max - global_min)
#     # create additional channel 
#     const = torch.ones(len(X), 1) * val
#     X = torch.cat((X, const), dim=1)
#     y = torch.stack([torch.from_numpy(y) for x, y in batch])
#     return X, y
# collate = partial(collate_fn, x_min, x_max)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
logging.info("Created dataloaders.")

# load model
bendr_params_path = os.path.join(params['bendr_dir'], 'version_0', 'hparams.yaml')
# load yaml file as dict 
with open(bendr_params_path, 'r') as f:
    bendr_params = yaml.unsafe_load(f)
model = create_bendr(bendr_params)
# make not trainable 
for param in model.parameters():
    param.requires_grad = False

# get file that ends with .ckpt 
ckpt_path = [f for f in os.listdir(params['bendr_dir']) if f.endswith('.ckpt')][0]
ckpt_path = os.path.join(params['bendr_dir'], ckpt_path)
# load weights
model.load_state_dict(torch.load(ckpt_path)['state_dict'])
logging.info("Loaded BENDR model.")

# build finetuning MLP
def build_mlp(hidden_layers, input_size, output_logits, hidden_size, dropout):
    layers = []
    for i in range(hidden_layers):
        layers.append(torch.nn.Linear(input_size if i == 0 else hidden_size, 2*hidden_size if i == 0 else hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(hidden_size, output_logits))
    return torch.nn.Sequential(*layers)

# create lightning model
class FinetuningModel(pl.LightningModule):
    def __init__(self, model, hidden_layers, input_size, output_logits, hidden_size, dropout, lr, loss):
        super().__init__()
        self.model = model
        self.mlp = build_mlp(hidden_layers, 512*input_size, output_logits, hidden_size, dropout)
        self.lr = lr
        self.loss_fn = nn.BCELoss() if loss == 'cross_entropy' else nn.MSELoss()
        self.loss_name = loss
        self.train_metric = BinaryAccuracy() if loss == 'cross_entropy' else MeanSquaredError()
        self.val_metric = BinaryAccuracy() if loss == 'cross_entropy' else MeanSquaredError()
    
    def forward(self, x):
        x = self.model.encoder(x) # pass through BENDRCNN
        x = x.view(x.size(0), -1) # flatten
        x = self.mlp(x)
        return x
    
    def loss(self, y_hat, y):
        if self.loss_name == 'cross_entropy':
            y_hat = torch.sigmoid(y_hat)
        # logging.info(f"y_hat and y concat: {torch.cat((y_hat, y), dim=1)}")
        return self.loss_fn(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        # root mse as metric if regression
        self.log('train_metric', self.train_metric(y_hat, y) if self.loss_name == 'cross_entropy' else torch.sqrt(self.train_metric(y_hat, y)))
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_metric_epoch', self.train_metric.compute() if self.loss_name == 'cross_entropy' else torch.sqrt(self.train_metric.compute()))
        self.train_metric.reset()    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if batch_idx % 40 == 0:
            logging.info(f'validation batch {batch_idx} loss: {loss} metric: {self.val_metric(y_hat, y)}')
        self.log('val_loss', loss)
        self.log('val_metric', self.val_metric(y_hat, y) if self.loss_name == 'cross_entropy' else torch.sqrt(self.val_metric(y_hat, y)))
        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else torch.sqrt(self.val_metric.compute()))
        self.val_metric.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_metric', self.val_metric(y_hat, y) if self.loss_name == 'cross_entropy' else torch.sqrt(self.val_metric(y_hat, y)))
        return loss
    
    def test_epoch_end(self, outputs):
        self.log('test_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else torch.sqrt(self.val_metric.compute()))
        self.val_metric.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

tb_logger = TensorBoardLogger("./reports/logs/",
                                  name=f"{run_id}_finetuning"
                                  )
tb_logger.log_hyperparams(params)
wandb_logger = WandbLogger(entity="deepseg",
                            project=f"bendr-finetuning-{params['task']}",
                            name=f"{run_id}",
                            save_dir="./reports/logs/"
                    )

x_rand = torch.rand(1, 129, 500)
x_output = model.encoder(x_rand)
input_size = x_output.shape[2]

# create finetuning model
model = FinetuningModel(model, params['hidden_layers'], input_size, params['output_logits'], params['hidden_size'], params['dropout'], params['learning_rate'], params['loss'])
# create trainer
trainer = Trainer(
    gpus=1, 
    max_epochs=params['epochs'], 
    logger=[tb_logger, wandb_logger],
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=True),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, verbose=True)
    ]
    )

# train
x_rand = torch.rand(batch_size, 129, 500)
x_output = model(x_rand)
logging.info(f"Model output shape: {x_output.shape}")

logging.info("Started training.")
trainer.fit(model, train_loader, val_loader)
logging.info("Finished training.")

# test
logging.info("Started testing.")
trainer.test(model, dataloaders=test_loader)
logging.info("Finished testing.")
