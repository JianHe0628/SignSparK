import argparse
from signspark.utils import logger
import torch

from signspark.rflow import RFlow


def unet_init(dataset_type, hidden_dim, arch="unet", **kwargs):
    if dataset_type == "wilor" or dataset_type == "hand":
        n_joints = 90
    elif dataset_type == "bodyonly" or dataset_type == "body":
        n_joints = 60
    elif dataset_type == "face":
        n_joints = 56
    elif dataset_type == "full":
        n_joints = 296
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    from signspark.unet_model import Wilor_UNET
    model = Wilor_UNET(
        njoints=n_joints,
        nfeats=1,
        latent_dim=hidden_dim,
        dim_mults=kwargs.get("dim_mults", (2, 2, 2, 2)),
        attention=kwargs.get("attention", False),
        dataset=kwargs.get("dataset_feat", "hand"),
        keyframe_conditioned=kwargs.get("keyframe_conditioned", True),
        unet_out_mult=kwargs.get("unet_out_mult", 2),
        arch=arch,
        length_mask_conditioned=kwargs.get("length_mask_conditioned", False),
        text_conditioned=kwargs.get("text_conditioned", True),
        text_mask_prob=kwargs.get("text_mask_prob", 0.1),
        text_enc_name=kwargs.get("text_enc_name", "CLIP"),
        text_enc_dim=kwargs.get("text_enc_dim", 640),
        text_enc_ver=kwargs.get("text_enc_ver", 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
    )
    return model, n_joints


def create_model_and_flow(
    hidden_t_dim,
    hidden_dim,
    dropout,
    vocab_size=None,
    config_name=None,
    **kwargs,
):
    logger.log("### Creating model and flow...")

    arch = kwargs.get("model_arch", "unet_large")
    dataset_type = kwargs.get("dataset_feat", "hand")
    model, _ = unet_init(
        dataset_type=dataset_type,
        hidden_dim=hidden_dim,
        arch=arch,
        **kwargs,
    )

    _flow = RFlow(
        lambda_flow=kwargs.get("lambda_flow", 1.0),
        lambda_recon=kwargs.get("lambda_recon", 1.0),
        lambda_velocity=kwargs.get("lambda_velocity", 1.0),
        lambda_acceleration=kwargs.get("lambda_acceleration", 0.1),
        )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"### The parameter count is {pytorch_total_params}")

    return model, _flow


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def _infer_n_joints(dataset_feat: str) -> int:
    if dataset_feat in ["wilor", "hand"]:
        return 90
    if dataset_feat in ["bodyonly", "body"]:
        return 60
    if dataset_feat == "face":
        return 56
    if dataset_feat == "fullbody":
        return 240
    raise ValueError(f"Unknown dataset_feat: {dataset_feat}")


def _run_batch_probe(model, cfg, accelerator):
    device = accelerator.device
    model = model.to(device)
    model.train()

    max_bs = getattr(cfg, "batch_probe_max", cfg.batch_size)
    start_bs = getattr(cfg, "batch_probe_start", 1)
    mode = getattr(cfg, "batch_probe_mode", "fwd_bwd")  # fwd or fwd_bwd

    n_joints = _infer_n_joints(getattr(cfg, "dataset_feat", "hand"))
    seq_len = cfg.seq_len

    last_ok = 0
    last_fail = None
    bs = start_bs
    logger.log(f"### Batch probe start: start_bs={start_bs}, max_bs={max_bs}, mode={mode}")

    while bs <= max_bs:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, n_joints, seq_len, device=device)
            obs_x0 = x
            obs_mask = torch.zeros_like(x, dtype=torch.bool)
            y = {
                "mask": torch.ones((bs, 1, seq_len), device=device, dtype=torch.bool),
                "lengths": torch.full((bs,), seq_len, device=device, dtype=torch.long),
                "text": ["probe"] * bs,
            }
            timesteps = torch.zeros(bs, device=device, dtype=torch.long)

            out = model(x, timesteps, y=y, obs_x0=obs_x0, obs_mask=obs_mask)
            if mode == "fwd_bwd":
                loss = out.float().mean()
                loss.backward()

            last_ok = bs
            logger.log(f"### Batch probe ok: bs={bs}")
            bs += 16
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.log(f"### Batch probe OOM at bs={bs}")
                last_fail = bs
                break
            raise
        finally:
            for obj in ["x", "obs_x0", "obs_mask", "y", "out", "loss", "timesteps"]:
                if obj in locals():
                    del locals()[obj]
            torch.cuda.empty_cache()

    logger.log(f"### Batch probe done. Max ok batch size (local): {last_ok}")