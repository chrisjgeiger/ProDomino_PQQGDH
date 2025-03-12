from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOSS = {}
def register_loss(model):
    @wraps(model)
    def _wrap_model(model):
        LOSS[model.__name__.lower()] = model
        return model

    return _wrap_model(model)

@register_loss
def BCE(pred,target,*args,**kwargs):
    loss = nn.BCEWithLogitsLoss()(pred.squeeze(),target.squeeze())

    return loss

def masked_BCE_3(pred,target,target_mask,*args,**kwargs):
    loss = nn.BCEWithLogitsLoss(reduction='none')(pred.squeeze(),target.squeeze())
    return (loss*target_mask[:,:,0]).mean()

@register_loss
def masked_BCE_onepos(pred,target,target_mask,*args,**kwargs):
    loss = nn.BCEWithLogitsLoss(reduction='none')(pred.squeeze(),target.squeeze())
    mask = target.squeeze().clone()
    pos = target_mask[:, :, 0].sum(1).cpu().apply_(lambda x: torch.randint(x,(1,1)))
    mask[np.arange(len(pos)), pos] = 1
    return (loss*mask).mean()