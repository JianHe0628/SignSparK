"""Keyframe conditioning mask, shared by training and sampling."""

import numpy as np
import torch


def get_keyframes_mask(data, keyframe_list=None, joint_prob=0):
    """Bool mask (B, n_joints, n_frames), True at observed keyframes.

    data: (B, n_joints, n_frames); keyframe_list: per-sample keyframe indices;
    joint_prob in (0.1, 1) observes a random joint subset per keyframe, else all.
    """
    batch_size, n_joints, n_frames = data.shape
    assert keyframe_list is not None, "keyframe_list must be provided"
    obs_feature_mask = torch.zeros(
        (batch_size, n_joints, n_frames), dtype=bool, device=data.device
    )

    for i in range(batch_size):
        keyframes = keyframe_list[i]
        for keyframe in keyframes:
            if keyframe < n_frames:
                if 0.1 < joint_prob < 1:
                    num_joints_selected = np.random.randint(int(joint_prob * n_joints), n_joints)
                    selected_joints = np.random.choice(n_joints, num_joints_selected, replace=False)
                    obs_feature_mask[i, selected_joints, keyframe] = True
                else:
                    obs_feature_mask[i, :, keyframe] = True
    return obs_feature_mask
