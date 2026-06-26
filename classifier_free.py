import torch.nn as nn
from copy import deepcopy


class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, text_scale=1.0):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.keyframe_conditioned = self.model.keyframe_conditioned
        self.text_scale = text_scale

    def forward(self, x, timesteps, y=None, obs_x0=None, obs_mask=None, **kwargs):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        # Classifier-free guidance: blend the conditional and unconditional
        # velocity fields by the per-sample text scale.
        out = self.model(x, timesteps, y, obs_x0, obs_mask, **kwargs)
        out_uncond = self.model(x, timesteps, y_uncond, obs_x0, obs_mask, **kwargs)

        return out_uncond + (self.text_scale.view(-1, 1, 1) * (out - out_uncond))
