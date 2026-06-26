"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from einops import rearrange, repeat

import numpy as np
from scipy import integrate
import torch as th
import torch
import sys
import torch.distributed as dist

from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import wandb
from signspark.sampling_rflow import (
    from_flattened_numpy,
    to_flattened_numpy,
)

sys.path.append(".")

import torch.nn.functional as F

from .utils.nn import mean_flat

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def _euler_step(model, x_t_masked, t, t_next, model_kwargs):
    vf_pred = model(t, x_t_masked, **model_kwargs)  # Model handles 3D→4D→3D internally
    x_1_est_next = x_t_masked + (rearrange(t_next, "b->b 1 1") - rearrange(t, "b->b 1 1")) * vf_pred
    return x_1_est_next

def _heun_step(model, x_t_masked, t, t_next, model_kwargs):
    t_exp = rearrange(t, "b->b 1 1")
    t_next_exp = rearrange(t_next, "b->b 1 1")
    vf_pred_t = model(t, x_t_masked, **model_kwargs)

    x_1_est_next = x_t_masked + (t_next_exp - t_exp) * vf_pred_t
    vf_pred_t_next = model(t_next, x_1_est_next, **model_kwargs)

    vf_pred = 0.5 * (vf_pred_t + vf_pred_t_next)
    z_next = x_t_masked + (t_next_exp - t_exp) * vf_pred
    return z_next

@torch.no_grad()
def sflow_sampler(
    model,
    sde,
    x_0=None,
    model_kwargs=None,
    x_embed=None,
    ode_package="torchdiffeq",
    ode_stepnum=None,
    **kwargs,
):
    assert ode_package in ["torchdiffeq", "heun"]
    device, shape = x_0.device, x_0.shape
    rtol = atol = sde.ode_tol
    eps = sde.eps

    def mask_and_clip_per_step(x_t, keyframe_mask, x_1,):
        assert keyframe_mask is not None
        x_t = x_1 * keyframe_mask + x_t * (~keyframe_mask)
        return x_t


    if ode_package == "torchdiffeq":
        progress_bar = tqdm(desc="sampling torchdiffeq")

        def ode_func(t, x_t):
            # print("torchdiffeq ode time", t)
            t_tensor = torch.ones(len(x_0), device=x_t.device) * t
            #########
            x_t = mask_and_clip_per_step(x_t, model_kwargs["obs_mask"], x_1=x_embed)
            #########
            vf_t = model(t_tensor, x_t, **model_kwargs) ## Model kwargs passed through closure

            progress_bar.update(int(t * 1000))
            return vf_t
        # def ode_func(t, x_t):
        #     # print("torchdiffeq ode time", t)
        #     t_tensor = torch.ones(len(x_0), device=x_t.device) * t
        #     #########
        #     x_t = mask_and_clip_per_step(x_t, model_kwargs["obs_mask"], x_1=x_embed)
        #     #########
        #     x_1_est = model(t_tensor, x_t, **model_kwargs) ## Model kwargs passed through closure

        #     t_expanded = rearrange(t_tensor, "b->b 1 1")
        #     vf_t = (x_1_est - x_t) / (1 - t_expanded).clamp_min(5e-2)  # [B,J,T]

        #     progress_bar.update(int(t * 1000))
        #     return vf_t

        result = odeint(
            ode_func,
            x_0,
            torch.tensor([0, sde.T], device=x_0.device, dtype=x_0.dtype),
            method="euler",
            rtol=rtol,
            atol=atol,
            adjoint_params=(),
            options=dict(step_size=sde.T / ode_stepnum),
        )[-1]
        progress_bar.close()
        return result, ode_stepnum
    
    elif ode_package == "heun":
        x_t = x_0
        timesteps = torch.linspace(0, sde.T, ode_stepnum + 1, device=x_0.device)
        for i in tqdm(range(ode_stepnum-1)):
            t_tensor = torch.ones(len(x_0), device=x_t.device) * timesteps[i]
            t_next_tensor = torch.ones(len(x_0), device=x_t.device) * timesteps[i+1]
            #########
            x_t = mask_and_clip_per_step(x_t, model_kwargs["obs_mask"], x_1=x_embed)
            #########
            x_t = _heun_step(model, x_t, t_tensor, t_next_tensor, model_kwargs)

        # Final step euler
        x_t = mask_and_clip_per_step(x_t, model_kwargs["obs_mask"], x_1=x_embed)
        x_t = _euler_step(model, 
                          x_t, 
                          torch.ones(len(x_0), device=x_t.device) * timesteps[-2], 
                          torch.ones(len(x_0), device=x_t.device) * timesteps[-1], 
                          model_kwargs)
        
        return x_t, ode_stepnum
    else:
        raise NotImplementedError(f"ode_package {ode_package} not implemented")


def _interp_xt_and_mask(x_1, t, x_0, mask=None, keyframe_mask=None):
    assert x_0.shape == x_1.shape
    t_expand = t[:, None, None]

    x_t = t_expand * x_1 + (1.0 - t_expand) * x_0

    if keyframe_mask == None:
        return x_t
    else:
        x_t = x_1 * keyframe_mask + x_t * (~keyframe_mask)
        return x_t
    # else:
    #     # mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_1.shape)
    #     # return th.where(mask == 0, x_1, x_t)
    #     return x_t * mask


class RFlow:
    def __init__(
        self,
        lambda_flow=1,
        lambda_recon=1,
        lambda_velocity=0.,
        lambda_acceleration=0.,
        *kwargs,
    ):
        self.l2_loss = lambda a, b: (
            a - b
        )**2
        self.lambda_velocity = lambda_velocity
        self.lambda_flow = lambda_flow
        self.lambda_recon = lambda_recon
        self.lambda_f2facc = lambda_acceleration
        print(f"### RFlow with lbd_flow {self.lambda_flow}, lbd_recon {self.lambda_recon}, lbd_velocity {self.lambda_velocity}")

    def _get_x_start(self, x_start_mean, std):
        """
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        """
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return x_start_mean + std * noise
    
    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_poses(model, *args, **kwargs)

    def decode(
        self,
        model,
        noise,
        keyframe_mask,
        **kwargs,
    ):
        class _SDE:
            def __init__(
                self,
            ):
                self.ode_tol = 1e-5
                self.T = 1
                self.eps = 1e-3

        def func(t, x, **kwargs):
            return model(x, t, **kwargs)
        
        masked_noise = kwargs['x_embed'] * keyframe_mask + noise * (~keyframe_mask)
        result, _nfe = sflow_sampler(model=func, x_0=masked_noise, sde=_SDE(), **kwargs)
        return result, _nfe
    
    def training_losses_poses(self, model, x_1, t, model_kwargs=None, x_0=None):
        mask = model_kwargs['y']['mask']  # [B,1,T]

        if x_0 is None:
            x_0 = th.randn_like(x_1)
            keyframed_x0 = x_1 * model_kwargs['obs_mask'] + x_0 * (~model_kwargs['obs_mask'])
        
        x_t_masked = _interp_xt_and_mask(
            x_1, t, x_0=x_0, mask=mask, keyframe_mask=model_kwargs['obs_mask']
        )

        terms = {}
        # t_expanded = rearrange(t, "b->b 1 1")
        # vf_gt = (x_1 - x_t_masked) / (1 - t_expanded).clamp_min(5e-2) # [B,J,T]
        # x_1_est = model(x_t_masked, t, **model_kwargs)  # Model handles 3D→4D→3D internally
        # vf_pred = (x_1_est - x_t_masked) / (1 - t_expanded).clamp_min(5e-2)  # [B,J,T]

        vf_gt = x_1 - keyframed_x0  # [B,J,T]
        vf_pred = model(x_t_masked, t, **model_kwargs)  # Model handles 3D→4D→3D internally
        x_1_est = x_t_masked + (1 - rearrange(t, "b->b 1 1")) * vf_pred

        # Unsqueezing for weighted L2
        vf_gt_unsqueezed = vf_gt.unsqueeze(2)  # [B,J,1,T]
        vf_pred_unsqueezed = vf_pred.unsqueeze(2)  # [B,J,1,T]
        x_1_unsqueezed = x_1.unsqueeze(2)  # [B,J,1,T]
        x_1_est_unsqueezed = x_1_est.unsqueeze(2)  # [B,J,1,T]
        mask_unsqueezed = mask.unsqueeze(1)  # [B,1,1,T]
        weights = torch.ones(*x_1_unsqueezed.shape[:-1],
                            1,
                            device=x_1_unsqueezed.device,
                            dtype=x_1_unsqueezed.dtype)  # [B,J,1,1]
        time_weights = torch.ones(*x_1_unsqueezed.shape,
                                  device=x_1_unsqueezed.device,
                                  dtype=x_1_unsqueezed.dtype)  # [B,J,1,T]
        
        # Vector Field MSE
        terms["flow_mse"] = self.masked_l2_weighted(
            vf_gt_unsqueezed, vf_pred_unsqueezed, mask_unsqueezed, weights, time_weights
        )

        # Weighted Reconstruction Loss
        terms["recon_mse"] = self.masked_l2_weighted(
            x_1_unsqueezed, x_1_est_unsqueezed, mask_unsqueezed, weights, time_weights
        )

        # Velocity Loss
        target_velocity = (x_1_unsqueezed[..., 1:] - x_1_unsqueezed[..., :-1])
        pred_velocity = (x_1_est_unsqueezed[..., 1:] - x_1_est_unsqueezed[..., :-1])
        if self.lambda_velocity > 0.:
            terms["velocity_l2"] = self.masked_l2(
                target_velocity, pred_velocity, mask_unsqueezed[:, :, :, 1:],
            )
        
        if self.lambda_f2facc > 0.:
            # Frame-to-frame acceleration loss
            target_acceleration = target_velocity[..., 1:] - target_velocity[..., :-1]
            pred_acceleration = pred_velocity[..., 1:] - pred_velocity[..., :-1]
            terms["f2facc_l2"] = self.masked_l2(
                target_acceleration, pred_acceleration, mask_unsqueezed[:, :, :, 2:],
            )
            

        terms["loss"] = terms["flow_mse"] * self.lambda_flow +\
                        terms.get("velocity_l2", 0.) * self.lambda_velocity +\
                        terms.get("recon_mse", 0.) * self.lambda_recon +\
                        terms.get("f2facc_l2", 0.) * self.lambda_f2facc
        # terms["loss"] = terms.get("recon_mse", 0.) * self.lambda_recon +\
        #                 terms.get("velocity_l2", 0.) * self.lambda_velocity    

        terms["wandb_dict"] = self.log_min_max(x_1, x_1_est)
        return terms


    def log_min_max(self, x_1, x_1_est):
        x_1_np = x_1.detach().cpu().numpy()
        x_1_est_np = x_1_est.detach().cpu().numpy()

        _wandb_dict = dict(
            x_1_min=x_1_np.min().item(),
            x_1_max=x_1_np.max().item(),
            x_1_mean=x_1_np.mean().item(),
            x_1_std=x_1_np.std().item(),
            x_1_est_min=x_1_est_np.min().item(),
            x_1_est_max=x_1_est_np.max().item(),
            x_1_est_mean=x_1_est_np.mean().item(),
            x_1_est_std=x_1_est_np.std().item(),
        )
        return _wandb_dict
    
    def masked_l2_weighted(self, a, b, mask, weights, time_weights, over_keyframes=False):
        """
        Args:
            a: bs, J, Jdim, seqlen
            b: bs, J, Jdim, seqlen
            mask: bs, 1, 1, seqlen
            weights: bs, J, Jdim, 1
            time_weights: bs, J, Jdim, seqlen
        """
        assert ((mask.shape == (a.shape[0], 1, 1, a.shape[3]) and not over_keyframes) or (mask.shape == a.shape and over_keyframes))
        assert weights.shape == (a.shape[0], a.shape[1], a.shape[2],
                                 1), f'{weights.shape}'
        assert time_weights.shape == (a.shape[0], a.shape[1], a.shape[2],
                                 a.shape[3]), f'{time_weights.shape}'

        loss = self.l2_loss(a, b)
        # average over the features, dim [1,2]
        weights = weights / weights.sum(dim=[1, 2], keepdims=True)
        loss = loss * weights

        # time_weights = time_weights / time_weights.sum(dim=[3], keepdims=True)
        loss = loss * time_weights

        # [bs, J, Jdim, seqlen]
        loss = loss * mask.float(
        )  # gives \sigma_euclidean over unmasked elements
        # n_entries = a.shape[1] * a.shape[2]
        # [bs]
        loss = sum_flat(loss)
        # average over the length
        # [bs]
        non_zero_elements = sum_flat(mask)
        loss = loss / non_zero_elements
        return loss
    
    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(
            loss *
            mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

