import torch
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import MeanSquaredError
import logging
import torch.nn as nn
from torchmetrics import Metric 
import numpy as np
import pytorch_lightning as pl

def min_max_normalize_np(x:np.array, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax - xmin == 0:
            x = np.zeros_like(x)
            return x
    elif len(x.shape) == 3:
        xmin = np.min(np.min(x, axis=1, keepdims=True), axis=-1, keepdims=True)
        xmax = np.max(np.max(x, axis=1, keepdims=True), axis=-1, keepdims=True)
        constant_trials = (xmax - xmin) == 0
        if np.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6
    x = (x - xmin) / (xmax - xmin)
    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x
# build finetuning MLP
def build_mlp(hidden_layers, input_size, output_logits, hidden_size, dropout):
    layers = []
    for i in range(hidden_layers):
        layers.append(torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(hidden_size, output_logits))
    return torch.nn.Sequential(*layers)


class AngleMetric(Metric): 
    def __init__(self):
        super().__init__()

        # these states will be reset by the built-in reset method
        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape
        # print(f"preds shape {preds.shape}, target shape {target.shape}")
        errors = self._compute_angle_loss(preds, target)
        # print(f"update: metric error {errors}")
        self.errors += errors 
        self.total += 1 # reduced error already 

    def compute(self):
        logging.info(f"metric error {self.errors}, total {self.total} = {self.errors / self.total}")
        return self.errors.float() / self.total
    
    def _compute_angle_loss(self, preds, target): 
        """
        Compute the angle loss between two vectors
        Note that we already reduce here 
        We take the absolute value for the metric
        """
        return torch.mean(torch.abs(torch.atan2(torch.sin(preds - target), torch.cos(preds - target))))


class AngleLoss(nn.Module):
    def __init__(self):
        """
        Custom loss function for models that predict the angle on the fix-sacc-fix dataset
        Angles -pi and pi should lead to 0 loss, since this is actually the same angle on the unit circle
        Angles pi/2 and -pi/2 should lead to a large loss, since this is a difference by pi on the unit circle
        Therefore we compute the absolute error of the "shorter" direction on the unit circle
        """
        super(AngleLoss, self).__init__()

    def forward(self, pred, target, ):
        """
        Inputs:
        pred: predicted angle in rad
        pred: target angle in rad 
        Output: reduced angle diff MSE 
        """
        # atan2(sin(x-y), cos(x-y)) gives us the angle difference 
        # we square it to get the MSE angle loss 
        return torch.mean(torch.square(torch.atan2(torch.sin(pred - target), torch.cos(pred - target))))

# create lightning model
class FinetuningModel(pl.LightningModule):
    def __init__(self, model, hidden_layers, input_size, output_logits, hidden_dim, hidden_size, dropout, lr, loss, encoding="backbone"):
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(self.model.backbone.num_channels+self.model.transformer.d_model, 
                      self.model.backbone.num_channels+self.model.transformer.d_model, 3, padding=1),
            nn.BatchNorm1d(self.model.backbone.num_channels+self.model.transformer.d_model),
            nn.Dropout1d(0.1),
            nn.Conv1d(self.model.backbone.num_channels+self.model.transformer.d_model, self.model.backbone.num_channels, 3, padding=1),
            nn.BatchNorm1d(self.model.backbone.num_channels),
            nn.Dropout1d(0.1),
            nn.Conv1d(self.model.backbone.num_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout1d(0.1),
        )
        self.mlp = build_mlp(hidden_layers, self.hidden_dim*input_size, output_logits, hidden_size, dropout)
        self.lr = lr
        self.loss_fn = nn.BCELoss() if loss == 'cross_entropy' else nn.MSELoss()
        self.loss_fn = AngleLoss() if loss == 'angle_loss' else self.loss_fn
        self.loss_name = loss
        self.train_metric = BinaryAccuracy() if loss == 'cross_entropy' else MeanSquaredError(squared=False) #RMSE
        self.train_metrics = AngleMetric() if loss == 'angle_loss' else self.train_metric
        self.val_metric = BinaryAccuracy() if loss == 'cross_entropy' else MeanSquaredError(squared=False)
        self.val_metrics = AngleMetric() if loss == 'angle_loss' else self.val_metric
        logging.info(f"Created finetuning model for task with loss {loss} and metric {self.train_metrics} and {self.val_metrics}")
        self.model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.encoding = encoding
        self.save_hyperparameters()

    def on_validation_start(self):
        logging.info("Manually setting DETRtime to train_mode, for eval()")
        self.model.eval()

    def forward(self, x):
        if self.encoding == "model":
        # print(x)
            out, _, _, _, memory, _ = self.model(x)
            timestamps = memory.size(2)
            discrete = self.model.generate_hidden_sequence_predictions(out, timestamps)
            discrete = discrete.permute(0, 2, 1)
            x = torch.cat([memory, discrete], dim=1)

        # print(outputs)
        elif self.encoding == "backbone":
            out, _, z, _, _, _ = self.model(x)
            timestamps = z.size(2)
            discrete = self.model.generate_hidden_sequence_predictions(out, timestamps)
            discrete = discrete.permute(0, 2, 1)
            x = torch.cat([z, discrete], dim=1)

        x = self.cnn(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.mlp(x)  # pass through MLP
        return x
    
    def loss(self, y_hat, y):
        # if self.loss_name == 'cross_entropy':
            # y_hat = torch.sigmoid(y_hat)
        # logging.info(f"y_hat and y concat: {torch.cat((y_hat, y), dim=1)}")
        # logging.info(f"y_hat and y concatentared shape: {torch.cat((y_hat, y), dim=1)}")
        loss = self.loss_fn(y_hat, y)
        # logging.info(f"Loss: {loss}")
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self(x)
        print(y_hat.size())
        print(y.size())
        loss = self.loss(y_hat, y)
        
        # logging.info(f"Shapes of y_hat and y: {y_hat.shape}, {y.shape}")

        self.log('train_loss', loss)
        # root mse as metric if regression
        self.log('train_metric', self.train_metric(y_hat, y)) 
        return loss

    def on_train_epoch_end(self):
        self.log('train_metric_epoch', self.train_metric.compute() if self.loss_name == 'cross_entropy' else self.train_metric.compute())
        #logging.info('train_metric_epoch', self.train_metric.compute() if self.loss_name == 'cross_entropy' else self.train_metric.compute())
        
        self.train_metric.reset()    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self(x)
        print(y_hat.size())
        print(y.size())
        loss = self.loss(y_hat, y)
        # if batch_idx % 40 == 0:
        #     logging.info(f'validation batch {batch_idx} loss: {loss} metric: {self.val_metric.compute()}')

        # logging.info(f"Shapes of y_hat and y: {y_hat.shape}, {y.shape}")

        self.log('val_loss', loss)
        self.log('val_metric', self.val_metric(y_hat, y) if self.loss_name == 'cross_entropy' else self.val_metric(y_hat, y))
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else self.val_metric.compute())
        #logging.info('val_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else self.val_metric.compute())
        
        self.val_metric.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_metric', self.val_metric(y_hat, y) if self.loss_name == 'cross_entropy' else self.val_metric(y_hat, y))
        return loss
    
    def on_test_epoch_end(self):
        self.log('test_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else self.val_metric.compute())
        #logging.info('test_metric_epoch', self.val_metric.compute() if self.loss_name == 'cross_entropy' else self.val_metric.compute())
        
        self.val_metric.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
