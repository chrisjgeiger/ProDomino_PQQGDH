
from functools import wraps
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC,BinaryPrecision

import lightning.pytorch as pl

from .loss_fn import LOSS

MODELS = {}
def register_model(model):
    @wraps(model)
    def _wrap_model(model):
        MODELS[model.__name__.lower()] = model
        return model

    return _wrap_model(model)

class BaseModel(pl.LightningModule):
    def __init__(self,loss):
        super().__init__()
        self.model = None
        self.loss_fn = LOSS[loss]
        self.optimizer = torch.optim.Adam
        self.learning_rate = 1e-3
        self.input_channels = 256
        self.auroc = BinaryAUROC()
        self.precision = BinaryPrecision()
        self.validation_step_outputs = []
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        padded_target, padded_insert, padded_target_mask, padded_insert_mask, long_label, sample = batch
        x = padded_target.swapaxes(1,2).float()
        pred = self.model(x)

        loss = self.loss_fn(pred,long_label,padded_target_mask)
        self.log('loss/train',loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        padded_target, padded_insert, padded_target_mask, padded_insert_mask, long_label, sample = batch


        x = padded_target.swapaxes(1,2).float()
        pred = self.model(x)

        loss = self.loss_fn(pred, long_label, padded_target_mask)

        self.log('loss/valid',loss.item())
        auroc_score = self.auroc(pred.squeeze()[padded_target_mask[:,:,0]], long_label.squeeze()[padded_target_mask[:,:,0]]).cpu()
        pr_score = self.precision(pred.squeeze()[padded_target_mask[:,:,0]], long_label.squeeze()[padded_target_mask[:,:,0]]).cpu()
        self.validation_step_outputs.append(torch.Tensor((auroc_score,pr_score)))

        return loss

    def on_validation_epoch_end(self) -> None:
        auroc,pr = torch.stack(self.validation_step_outputs).mean(0)
        self.log('metrics/precision',pr)
        self.log('metrics/auroc', auroc)



    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

@register_model
class Mini_3B_MLP(BaseModel):
    def __init__(self,input_channels,optimizer,loss,warmup=False,warmup_rate=10_000,*args,**kwargs):
        super().__init__(loss)
        self.model = nn.Sequential(nn.Linear(input_channels,1280),nn.ReLU(),nn.Linear(1280,1))
        self.input_channels = input_channels
        self.learning_rate = float(optimizer.learning_rate)
        self.warmup = warmup
        self.warmup_rate = warmup_rate

    def forward(self, x):
        pred = self.model(x)
        return pred
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        padded_target, padded_insert, padded_target_mask, padded_insert_mask, long_label, sample = batch
        x = padded_target.float()
        pred = self.model(x)
        loss = self.loss_fn(pred,long_label,padded_target_mask)
        self.log('loss/train',loss.item())
        self.log('learning_rate',self.learning_rate)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        padded_target, padded_insert, padded_target_mask, padded_insert_mask, long_label, sample = batch
        x = padded_target.float()
        pred = self.model(x)

        loss = self.loss_fn(pred, long_label, padded_target_mask)
        self.log('loss/valid',loss.item())
        auroc_score = self.auroc(pred.squeeze()[padded_target_mask[:,:,0]], long_label.squeeze()[padded_target_mask[:,:,0]]).cpu()
        pr_score = self.precision(pred.squeeze()[padded_target_mask[:,:,0]], long_label.squeeze()[padded_target_mask[:,:,0]]).cpu()
        self.validation_step_outputs.append(torch.Tensor((auroc_score,pr_score)))

        return loss

