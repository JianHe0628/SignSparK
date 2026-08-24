import copy
import functools
import os, json

import torch as th
import torch
import hydra

from transformers import set_seed
from signspark.pose_datasets_lmdb import load_lmdb_dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np

import time
from signspark.utils import dist_util, logger
from signspark.keyframes import get_keyframes_mask
from classifier_free import ClassifierFreeSampleModel
from functools import partial
from basic_utils import (
    args_to_dict,
    create_model_and_flow,
)


def update_configs(config):
    config.checkpoint_path = config.checkpoint_path + config.note
    assert config.eval.candidate_num > 0
    return config


@torch.no_grad()
@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg):
    # cfg = cfg.data  # hydra
    cfg = update_configs(cfg)
    _current_seed = cfg.seed
    set_seed(_current_seed)

    output_path = os.path.join(
        Path(cfg.eval.model_path).parent,
        f"debug{int(cfg.eval.is_debug)}_eulerstepsize{cfg.eval.ode_stepnum}_can{cfg.eval.candidate_num}_anchor{int(cfg.eval.clip_denoised)}_samples",
    )

    dist_util.setup_dist()
    logger.configure()

    logger.log("### Creating model and flow...")
    logger.log(cfg)
    logger.log("load model from", cfg.eval.model_path)
    print(f'### Sampling with {cfg.eval.ode_stepnum} {cfg.eval.sampler} steps')

    _model, _flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    print("### Model and flow created, loading state dict...")
    _model.load_state_dict(
        dist_util.load_state_dict(cfg.eval.model_path, map_location="cpu"), strict=True
    )

    if cfg.eval.text_guidance_scale != -1.0:
        print(f"### Setting the text guidance scale to {cfg.eval.text_guidance_scale} for evaluation")
        text_scale = th.tensor(cfg.eval.text_guidance_scale, device=dist_util.dev())
        _model = ClassifierFreeSampleModel(_model, text_scale=text_scale)
        output_path = output_path + f"_textscale{cfg.eval.text_guidance_scale}"
    
    if os.path.exists(output_path):
        output_path += time.strftime("_%Y%m%d-%H:%M:%S")
    os.makedirs(output_path, exist_ok=True)

    _model.eval().requires_grad_(False).to(dist_util.dev())
    segments = False

    print("### Loading evaluation dataset...")
    data4eval = load_lmdb_dataset(
        batch_size=cfg.eval.batch_size,
        seq_len=cfg.eval.seq_len,
        data_args=cfg,
        split=cfg.eval.split,
        deterministic=True,
        segments=segments,
        num_workers=0,
        drop_last= True if cfg.eval.text_guidance_scale != -1 else False,
        concat_hands=True if cfg.dataset_feat == "hand" else False,
    )
    data4eval = iter(data4eval)
    all_test_data = []
    n_loaded = 0
    max_samples = cfg.eval.get("max_samples", -1)
    try:
        while True:
            embed, batch_dict = next(data4eval)
            all_test_data.append((embed, batch_dict))
            n_loaded += embed.shape[0]
            if cfg.eval.is_debug:
                logger.log("### Debug mode is on, break..")
                break
            if max_samples is not None and max_samples > 0 and n_loaded >= max_samples:
                logger.log(f"### Reached max_samples={max_samples} ({n_loaded} loaded), stop reading..")
                break
    except StopIteration:
        print("### End of reading iteration...")
    
    print(f"### Length of the evaluation dataset {len(all_test_data)}")

    sample_output_path = os.path.join(
            output_path,
            f"seed{_current_seed}_clampstep{cfg.eval.clamp_step}_{cfg.eval.note}.npy",
        )

    start_time = time.time()
    all_poses, all_lengths, all_texts, all_inputs, all_keyframes, all_names = [], [], [], [], [], []
    for _ith in range(cfg.eval.candidate_num):

        batch_poses, batch_lengths, batch_texts, batch_inputs, batch_keyframes, batch_names = None, None, None, None, None, None
        for _i, (embed, _batch_dict) in enumerate(all_test_data):
            batch_dict = copy.deepcopy(_batch_dict)
            if not batch_dict:
                continue

            if cfg.eval.is_debug and _i >= 1:
                logger.log("### Debug mode is on, break..")
                break

            embed = embed.to(dist_util.dev())
            cond = {'y': copy.deepcopy(batch_dict)}
            if cfg.dataset_feat == "hand":
                # Split hands to left/right and interleave per sample: L0,R0,L1,R1,...
                left_embed = embed[:, :embed.shape[1] // 2, :]
                right_embed = embed[:, embed.shape[1] // 2:, :]
                embed = torch.stack([left_embed, right_embed], dim=1).reshape(
                    -1, left_embed.shape[1], left_embed.shape[2]
                )
                cond['y']['mask'] = cond['y']['mask'].repeat_interleave(2, dim=0)
                cond['y']['lengths'] = cond['y']['lengths'].repeat_interleave(2, dim=0)
                cond['y']['text'] = [text for text in cond['y']['text'] for _ in range(2)]
                cond['y']['keyframes'] = [kf for kf in cond['y']['keyframes'] for _ in range(2)]
                cond['y']['video_names'] = [vn for vn in cond['y']['video_names'] for _ in range(2)]


            cond['obs_x0'] = embed # [bs, n_joints=96, n_frames=224]
            cond['obs_mask'] = get_keyframes_mask(data=embed, keyframe_list=cond['y']['keyframes'], joint_prob=0)  # [bs, n_joints=96, n_frames=224]
            if cfg.eval.keyframes_mask_all:
                print("\033[91mWarning, masking all keyframes for unconditional sampling!\033[0m")
                cond['obs_mask'] = torch.zeros_like(cond['obs_mask']).bool()

            noise = th.randn_like(embed)

            samples, _nfe = _flow.decode(
                _model,
                noise=noise,    # Starting point
                keyframe_mask=cond['obs_mask'],
                x_embed=embed,
                model_kwargs=cond,
                ode_package=cfg.eval.sampler,
                ode_stepnum=cfg.eval.ode_stepnum,
            )

            if batch_poses is None:
                batch_poses = samples.cpu().numpy()
                batch_lengths = cond['y']['lengths'].numpy()
                batch_texts = cond['y']['text']
                batch_names = cond['y']['video_names']
                batch_inputs = embed.cpu().numpy()
                batch_keyframes = cond['obs_mask'].cpu().numpy()
            else:
                batch_poses = np.concatenate((batch_poses, samples.cpu().numpy()), axis=0)
                batch_lengths = np.concatenate((batch_lengths, cond['y']['lengths'].numpy()), axis=0)
                batch_texts = batch_texts + cond['y']['text']  
                batch_names = batch_names + cond['y']['video_names']  
                batch_inputs = np.concatenate((batch_inputs, embed.cpu().numpy()), axis=0)
                batch_keyframes = np.concatenate((batch_keyframes, cond['obs_mask'].cpu().numpy()), axis=0)

        if batch_poses is not None:
            all_poses.append(batch_poses)
            all_lengths.append(batch_lengths)
            all_texts.append(batch_texts)
            all_inputs.append(batch_inputs)
            all_keyframes.append(batch_keyframes)
            all_names.append(batch_names)

    print(f"### Total sampling time for all candidates: {time.time() - start_time} seconds")
    pred_poses = np.stack(all_poses, axis=0)  # (N, n_joints, n_frames)
    pred_lengths = np.stack(all_lengths, axis=0)  # (N,)
    input_texts = np.stack(all_texts, axis=0)  # (N,)
    input_poses = np.stack(all_inputs, axis=0)  # (N, n_joints, n_frames)
    input_keyframes = np.stack(all_keyframes, axis=0)  # (N, n_frames)
    input_names = np.stack(all_names, axis=0)  # (N,)

    print("### Output pred_poses shape", pred_poses.shape, pred_lengths.shape, input_texts.shape, input_poses.shape, input_keyframes.shape)

    np.save(sample_output_path, {
        'pred_poses': pred_poses,
        'lengths': pred_lengths,
        'text': input_texts,
        'num_samples': cfg.batch_size,
        'gt_poses': input_poses,
        'keyframes': input_keyframes,
        'video_names': input_names,
    })

    print(f"### Written the decoded output to {sample_output_path}")


if __name__ == "__main__":
    main()
