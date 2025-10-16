import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import morphology
from typing import Callable, Optional


class DistLoss(nn.Module):
    def __init__(self, apply_nonlin: Optional[Callable] = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1.0, clip_tp: Optional[float] = None):
        """
        Distance-based Dice Loss
        """
        super(DistLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp

    def forward(self, x, y):
        shp_x = x.shape

        # Compute distance map for weighting
        with torch.no_grad():
            loss_mask = self.dist_map(y)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = self.get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

    def dist_map(self, tensor):
        """
        Compute distance transform for each item in the batch
        """
        batch_size = tensor.shape[0]
        num_classes = tensor.shape[1] if len(tensor.shape) > 1 else 1
        dist_maps = torch.ones_like(tensor)

        for b in range(batch_size):
            for c in range(num_classes):
                if len(tensor.shape) == 4:  # 2D data
                    array = tensor[b, c].detach().cpu().numpy() if num_classes > 1 else tensor[
                        b, 0].detach().cpu().numpy()
                else:  # 3D data
                    array = tensor[b, c].detach().cpu().numpy() if num_classes > 1 else tensor[
                        b, 0].detach().cpu().numpy()

                if array.sum() == 0:  # Skip if no pixels in this class
                    continue

                skeleton = morphology.skeletonize(array > 0)
                distance_map = distance_transform_edt(~skeleton)
                distance_to_centerline = distance_map * (array > 0)

                if distance_to_centerline.max() != 0:
                    distance_to_centerline /= distance_to_centerline.max()

                dist_map = 1 - (np.power(distance_to_centerline, 0.6))

                if num_classes > 1:
                    dist_maps[b, c] = torch.tensor(dist_map, dtype=torch.float32, device=tensor.device)
                else:
                    dist_maps[b, 0] = torch.tensor(dist_map, dtype=torch.float32, device=tensor.device)

        return dist_maps

    def get_tp_fp_fn_tn(self, net_output, gt, axes=None, mask=None, square=False):
        """
        Calculate true positive, false positive, false negative, true negative
        """
        if axes is None:
            axes = tuple(range(2, len(net_output.size())))

        shp_x = net_output.shape
        shp_y = gt.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if net_output.shape == gt.shape:
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x, device=net_output.device)
                y_onehot.scatter_(1, gt, 1)

        tp = net_output * y_onehot
        fp = net_output * (1 - y_onehot)
        fn = (1 - net_output) * y_onehot
        tn = (1 - net_output) * (1 - y_onehot)

        if mask is not None:
            # Ensure mask has the same shape as net_output
            if mask.shape != net_output.shape:
                # If mask has fewer channels, expand it
                if mask.shape[1] < net_output.shape[1]:
                    expanded_mask = torch.ones_like(net_output)
                    for c in range(mask.shape[1]):
                        expanded_mask[:, c:c + 1] = mask[:, c:c + 1]
                    mask = expanded_mask
                # If mask has more channels, truncate it
                elif mask.shape[1] > net_output.shape[1]:
                    mask = mask[:, :net_output.shape[1]]

            tp *= mask
            fp *= mask
            fn *= mask
            tn *= mask

        if square:
            tp = tp ** 2
            fp = fp ** 2
            fn = fn ** 2
            tn = tn ** 2

        if len(axes) > 0:
            tp = tp.sum(dim=axes, keepdim=False)
            fp = fp.sum(dim=axes, keepdim=False)
            fn = fn.sum(dim=axes, keepdim=False)
            tn = tn.sum(dim=axes, keepdim=False)

        return tp, fp, fn, tn


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Optional[Callable] = None, batch_tversky: bool = False,
                 do_bg: bool = True, smooth: float = 1.0, alpha: float = 0.2, beta: float = 0.8):
        """
        Memory-efficient Tversky Loss
        """
        super(TverskyLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_tversky = batch_tversky
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Make everything shape (b, c)
        axes = list(range(2, len(x.shape)))

        with torch.no_grad():
            if len(x.shape) != len(y.shape):
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        tp = (x * y_onehot).sum(axes)
        fp = (x * (~y_onehot)).sum(axes)
        fn = ((1 - x) * y_onehot).sum(axes)

        if self.batch_tversky:
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        tversky = (tp + self.smooth) / (
            torch.clip(tp + self.alpha * fp + self.beta * fn + self.smooth, 1e-8))

        tversky = tversky.mean()
        return -tversky


class TSandDSLoss(nn.Module):
    def __init__(self, dist_kwargs=None, tversky_kwargs=None,
                 weight_dist: float = 0.3, weight_tversky: float = 0.7):
        """
        Combined Tversky and Distance-based Dice Loss
        """
        super(TSandDSLoss, self).__init__()

        # Set default parameters if not provided
        if dist_kwargs is None:
            dist_kwargs = {}
        if tversky_kwargs is None:
            tversky_kwargs = {}

        self.weight_tversky = weight_tversky
        self.weight_dist = weight_dist

        self.ds = DistLoss(apply_nonlin=torch.sigmoid, **dist_kwargs)
        self.tversky = TverskyLoss(apply_nonlin=torch.sigmoid, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ds_loss = self.ds(net_output, target)
        tversky_loss = self.tversky(net_output, target)

        result = self.weight_dist * ds_loss + self.weight_tversky * tversky_loss
        return result


# Example usage
if __name__ == '__main__':
    # Create sample data
    pred = torch.rand((2, 2, 32, 32))
    ref = torch.randint(0, 2, (2, 1, 32, 32)).float()

    # Initialize loss function
    loss_fn = TSandDSLoss(
        dist_kwargs={'batch_dice': True, 'do_bg': False},
        tversky_kwargs={'batch_tversky': True, 'do_bg': False},
        weight_dist=0.3,
        weight_tversky=0.7
    )

    # Calculate loss
    loss = loss_fn(pred, ref)
    print(f"Combined loss: {loss.item()}")