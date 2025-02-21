import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffuser.utils as utils

class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        # weighted_loss = (loss * self.weights).mean()
        if self.action_dim > 0:
            a0_loss = (
                loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
            ).mean()
            info = {"a0_loss": a0_loss}
        else:
            info = {}
        return loss * self.weights, info
        # return weighted_loss, {"a0_loss": a0_loss}


class WeightedStateLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return loss * self.weights, {"a0_loss": weighted_loss}
        # return weighted_loss, {"a0_loss": weighted_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(), utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": utils.to_torch(corr, device=pred.device),
        }

        return loss, info


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedStateL2(WeightedStateLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "state_l2": WeightedStateL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}


def apply_conditioning(x, conditions):
    cond_masks = conditions["masks"].to(bool)
    x[cond_masks] = conditions["x"][cond_masks].clone()

    if "player_idxs" in conditions.keys():
        if x.shape[-1] < 4:  # pure position information w.o. player info
            x = torch.cat([conditions["player_idxs"], x], dim=-1)
            x = torch.cat([x, conditions["player_hoop_sides"]], dim=-1)
        else:
            x[:, :, :, 0] = conditions["player_idxs"]
            x[:, :, :, -1] = conditions["player_hoop_sides"]

    return x
