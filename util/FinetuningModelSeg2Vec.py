import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import torch.nn.functional as F
import pandas as pd 
import numpy as np

# build finetuning MLP
def build_mlp(hidden_layers, input_size, output_logits, hidden_size, dropout):
    layers = []
    for i in range(hidden_layers):
        layers.append(torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(hidden_size, output_logits))
    return torch.nn.Sequential(*layers)


# create lightning model
class FinetuningModel(pl.LightningModule):
    def __init__(self, model, hidden_layers, output_logits, hidden_dim, hidden_size, dropout,
                 lr, loss,
                 encoding="backbone", use_mask=False):
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.use_mask = use_mask

        if encoding == "backbone":
            self.encoding_size = self.model.backbone.num_channels
        elif encoding == "transformer":
            self.encoding_size = self.model.transformer.d_model
        elif encoding == "transformer_projection":
            self.encoding_size = self.model.backbone.num_channels
        elif encoding == "both":
            self.encoding_size = self.model.backbone.num_channels + self.model.transformer.d_model
        elif encoding == "bendr_backbone":
            #@Lukas change this to the correct call and size
            self.encoding_size = self.model.backbone.num_channels
        elif encoding == "bendr_transformer":
            self.encoding_size = self.model.transformer.d_model
        elif encoding == "inception_time":
            raise NotImplementedError
        else:
            logging.error("Encoding not supported")
            raise NotImplementedError

        self.cnn = nn.Sequential(
            nn.Conv1d(self.encoding_size,
                      self.encoding_size, 3, padding=1),
            nn.BatchNorm1d(self.encoding_size),
            nn.Dropout1d(0.1),
            nn.Conv1d(self.encoding_size,
                      self.hidden_dim, 3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout1d(0.1),
            nn.Conv1d(self.hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout1d(0.1),
        )
        self.output_logits = output_logits

        self.mlp = build_mlp(hidden_layers, self.hidden_dim * model.num_time_steps, output_logits*model.num_time_steps, hidden_size, dropout)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss() if loss == 'cross_entropy' else nn.MSELoss()
        self.loss_name = loss
        if loss == "cross_entropy":

            self.train_metric = {"accuracy":Accuracy(task="multiclass", num_classes=14, average="none"),
                                 "f1": MulticlassF1Score(num_classes=14, average="none"),
                                 "precision": MulticlassPrecision(num_classes=14, average="none"),
                                 "recall": MulticlassRecall(num_classes=14, average="none"),
                                 "confusion_matrix": MulticlassConfusionMatrix(num_classes=14)
                                 } 
        else:
            self.train_metric = {"mse": MeanSquaredError(squared=False)}  # RMSE
        
        if loss == "cross_entropy":

            self.val_metric = {"accuracy":Accuracy(task="multiclass", num_classes=14, average="none"),
                                 "f1": MulticlassF1Score(num_classes=14, average="none"),
                                 "precision": MulticlassPrecision(num_classes=14,average="none"),
                                 "recall": MulticlassRecall(num_classes=14, average="none"),
                                 "confusion_matrix": MulticlassConfusionMatrix(num_classes=14)
                                } 
        else:
            self.val_metric = {"mse": MeanSquaredError(squared=False)}  # RMSE
        
        # self.val_metric = Accuracy(task="multiclass", num_classes=26) if loss == 'cross_entropy' else MeanSquaredError(squared=False)
        logging.info(
            f"Created finetuning model for task with loss {loss} and metric {self.train_metric} and {self.val_metric}")
        self.model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for _, d in self.train_metric.items():
            d.to(self.model_device)
        for _, d in self.val_metric.items():
            d.to(self.model_device)
         
        self.encoding = encoding
        self.save_hyperparameters()

    def on_validation_start(self):
        logging.info("Manually setting DETRtime to train_mode, for eval()")
        self.model.eval()

    def forward(self, x, regions):
        # logging.info(f"input_size {x.size()}")
        # logging.info(f"region_size {regions.size()}")
        
        if self.encoding == "transformer_projection":
            # print(x)
            _, _, _, memory, _ = self.model(x, regions)
            # logging.info(f"memory {memory.size()}")
        
            x = memory

        elif self.encoding == "backbone":
            _, z, _, _, _ = self.model(x, regions)
            x = z
            # logging.info(f"z {z.size()}")
        elif self.encoding == "transformer":
            raise NotImplementedError
        elif self.encoding == "both":
            raise NotImplementedError

        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.mlp(x)  # pass through MLP
        return x

    def loss(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        if self.encoding == "bendr":
            raise NotImplementedError
        else:

            x, _, _, regions, _, mask, y = batch
            x = x.permute(0, 2, 1)

            y_hat = self.forward(x, regions)
            bs = x.size(0)
            y_hat = y_hat.reshape(bs, self.output_logits, -1) #bs, C, seq
            # logging.info(f"y_hat {y_hat.size()}")
            seq_len = y_hat.size(2)
            # logging.info(f"Original y {y.size()}")
            y = y.permute(0, 2, 1) #bs, c, seq_len

            #Downsampling Targets
            if self.loss_name == "cross_entropy":
                y = y.float()
                y = F.interpolate(y, size=seq_len)#here we only downsample the labels
                y = y.long()
                
                y = torch.squeeze(y, dim=1)
            else:
                y = F.adaptive_avg_pool1d(y, seq_len) #30000 --> seq_len
            # logging.info(f"Downsampled y {y.size()}")

            #Removing rest class
            mask = mask.permute(0, 2, 1)
            mask = mask.float()
            mask = F.interpolate(mask, size=seq_len)#here we only downsample the labels
            mask = mask.bool()
            mask = mask.squeeze(dim=1)
            y_hat = y_hat.permute(0, 2, 1)

            y_hat = y_hat[~mask]
            y = y[~mask]
            loss = self.loss(y_hat, y)
            # logging.info(f"Shapes of y_hat and y: {y_hat.shape}, {y.shape}")

            self.log('train_loss', loss)
            # root mse as metric if regression
            for metric, f in self.train_metric.items():
                if metric != "confusion_matrix":
                    self.log(f'train_{metric}', f(y_hat, y).mean())
                else:
                    f(y_hat, y)
            return loss

    def on_train_epoch_end(self):
        for metric, f in self.train_metric.items():
                t = f.compute() #tensor
                
                if metric == "confusion_matrix":
                    confmat = t
                    num_classes = 14

                    df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in range(14)], columns = [i for i in range(14)])
        
                    print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum() ))
                    # df_cm.to_csv('raw_nums.csv') # you can use this to validate the number of samples is correct
                
                    #normalise the confusion matrix 
                    norm =  np.sum(df_cm, axis=1)
                    normalized_cm = (df_cm.T/norm).T # 

                    # normalized_cm.to_csv('norm_cdf.csv') #saved locally so that I could validate outside of wandb
                    
                    #log to wandb
                    # f, ax = plt.subplots(figsize = (15,10)) 
                    print(self.loggers)
                    wandb_logger = self.loggers[1]
                    wandb_logger.log_text(key="confusion_matrix", dataframe=normalized_cm)
                elif t.dim() == 0:
                    self.log(f'train_epoch_{metric}_mean', t.mean())
                    pass
                else:
                    self.log(f'train_epoch_{metric}_mean', t.mean())
                    for i, a in enumerate(t):
                        self.log(f'train_epoch_{metric}_class_{i}', a)

                f.reset()

        # self.log('train_metric_epoch',
        #          self.train_metric.compute() if self.loss_name == 'cross_entropy' else self.train_metric.compute())
        # logging.info('train_metric_epoch', self.train_metric.compute() if self.loss_name == 'cross_entropy' else self.train_metric.compute())

        # self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        if self.encoding == "bendr":
            raise NotImplementedError
        else:

            x, _, _, regions, _, mask, y = batch
            x = x.permute(0, 2, 1)

            y_hat = self.forward(x, regions)
            bs = x.size(0)
            y_hat = y_hat.reshape(bs, self.output_logits, -1) #bs, C, seq
            logging.info(f"y_hat {y_hat.size()}")
            seq_len = y_hat.size(2)
            logging.info(f"Original y {y.size()}")
            y = y.permute(0, 2, 1) #bs, c, seq_len
            if self.loss_name == "cross_entropy":
                y = y.float()
                y = F.interpolate(y, size=seq_len)#here we only downsample the labels
                y = y.long()
                y = torch.squeeze(y, dim=1)
            else:
                y = F.adaptive_avg_pool1d(y, seq_len)

            mask = mask.permute(0, 2, 1)
            mask = mask.float()
            mask = F.interpolate(mask, size=seq_len)#here we only downsample the labels
            mask = mask.bool()
            mask = mask.squeeze(dim=1)
            
            print(mask.size())
            y_hat = y_hat.permute(0, 2, 1)
            print(f"Unsampled {y_hat.size()}")
            y_hat = y_hat[~mask]
            print(f"Masked {y_hat.size()}")
            print(f"Unsampled {y.size()}")
            y = y[~mask]
            print(f"Unsampled {y.size()}")
            logging.info(f"Downsampled y {y.size()}")
        

            loss = self.loss(y_hat, y)
            # logging.info(f"Shapes of y_hat and y: {y_hat.shape}, {y.shape}")

            self.log('val_loss', loss)
            # root mse as metric if regression
            for metric, f in self.val_metric.items():
                if metric != "confusion_matrix":
                    self.log(f'val_{metric}', f(y_hat, y).mean())
                else:
                    f(y_hat, y)
            return loss

    def on_validation_epoch_end(self):
        for metric, f in self.val_metric.items():
                t = f.compute() #tensor
                
                if metric == "confusion_matrix":
                    confmat = t
                    num_classes = 14

                    df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in range(14)], columns = [i for i in range(14)])
        
                    print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum() ))
                    # df_cm.to_csv('raw_nums.csv') # you can use this to validate the number of samples is correct
                
                    #normalise the confusion matrix 
                    norm =  np.sum(df_cm, axis=1)
                    normalized_cm = (df_cm.T/norm).T # 

                    # normalized_cm.to_csv('norm_cdf.csv') #saved locally so that I could validate outside of wandb
                    
                    #log to wandb
                    # f, ax = plt.subplots(figsize = (15,10)) 
                    print(self.loggers)
                    wandb_logger = self.loggers[1]
                    wandb_logger.log_text(key="confusion_matrix", dataframe=normalized_cm)
                elif t.dim() == 0:
                        self.log(f'val_epoch_{metric}_mean', t.mean())
                        pass
                else:
                    self.log(f'val_epoch_{metric}_mean', t.mean())
                    for i, a in enumerate(t):
                        
                        self.log(f'val_epoch_{metric}_class_{i}', a)
                f.reset()
        # logging.info('val_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else self.val_metric.compute())

        # self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        if self.encoding == "bendr":
            raise NotImplementedError
        else:

            x, _, _, regions, _, mask, y = batch
            x = x.permute(0, 2, 1)

            y_hat = self.forward(x, regions)
            bs = x.size(0)
            y_hat = y_hat.reshape(bs, self.output_logits, -1) #bs, C, seq
            logging.info(f"y_hat {y_hat.size()}")
            seq_len = y_hat.size(2)
            logging.info(f"Original y {y.size()}")
            y = y.permute(0, 2, 1) #bs, c, seq_len
            if self.loss_name == "cross_entropy":
                y = y.float()
                y = F.interpolate(y, size=seq_len)#here we only downsample the labels
                y = y.long()
                y = torch.squeeze(y, dim=1)
            else:
                y = F.adaptive_avg_pool1d(y, seq_len)
            logging.info(f"Downsampled y {y.size()}")

            mask = mask.permute(0, 2, 1)
            mask = mask.float()
            mask = F.interpolate(mask, size=seq_len)#here we only downsample the labels
            mask = mask.bool()
            mask = mask.squeeze(dim=1)
            y_hat = y_hat.permute(0, 2, 1)

            y_hat = y_hat[~mask]
            y = y[~mask]
            loss = self.loss(y_hat, y)
            # logging.info(f"Shapes of y_hat and y: {y_hat.shape}, {y.shape}")

            self.log('test_loss', loss)
            # root mse as metric if regression
            for metric, f in self.val_metric.items():
                if metric != "confusion_matrix":
                    self.log(f'train_{metric}', f(y_hat, y).mean())
                else:
                    f(y_hat, y)
            return loss

    def on_test_epoch_end(self):
        for metric, f in self.val_metric.items():
                t = f.compute() #tensor
                
                if metric == "confusion_matrix":
                    confmat = t
                    num_classes = 14

                    df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in range(14)], columns = [i for i in range(14)])
        
                    print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum() ))
                    # df_cm.to_csv('raw_nums.csv') # you can use this to validate the number of samples is correct
                
                    #normalise the confusion matrix 
                    norm =  np.sum(df_cm, axis=1)
                    normalized_cm = (df_cm.T/norm).T # 

                    # normalized_cm.to_csv('norm_cdf.csv') #saved locally so that I could validate outside of wandb
                    
                    #log to wandb
                    # f, ax = plt.subplots(figsize = (15,10)) 
                    wandb_logger = self.loggers[1]
                    wandb_logger.log_text(key="confusion_matrix", dataframe=normalized_cm)
                elif t.dim() == 0:
                    self.log(f'test_epoch_{metric}_mean', t.mean())
                    pass
                else:
                    self.log(f'test_epoch_{metric}_mean', t.mean())
                    for i, a in enumerate(t):
                        
                        self.log(f'test_epoch_{metric}_class_{i}', a)
                f.reset()
        # logging.info('test_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else self.val_metric.compute())

        # self.val_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
