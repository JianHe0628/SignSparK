from datetime import datetime
import os
from signspark.utils import logger
from pathlib import Path

from basic_utils import (
    create_model_and_flow,
    args_to_dict,
    _run_batch_probe
)

from flow_train_util import TrainLoop_Flow
from transformers import set_seed
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import accelerate
from accelerate.utils import AutocastKwargs
from torch.optim import AdamW


def update_cfg(cfg):
    if cfg.is_debug:
        logger.log("### Debug mode is on")
        cfg.batch_size = 128
        cfg.microbatch = 64
        cfg.learning_steps = 102
        cfg.save_interval = 20
        cfg.log_interval = 10
    else:
        logger.log("### Debug mode is off")

    cfg.note = f"{cfg.model_arch}_bs{cfg.batch_size}_mcbs{cfg.microbatch}_lflow{cfg.lambda_flow}_lrecon{cfg.lambda_recon}_lvel{cfg.lambda_velocity}_{cfg.note}"
    cfg.checkpoint_path = cfg.checkpoint_path + "_" + cfg.note
    cfg.checkpoint_path = os.path.join(HydraConfig.get().run.dir, cfg.checkpoint_path)

    if os.path.exists(cfg.checkpoint_path):
        cfg.checkpoint_path = cfg.checkpoint_path + f"_{datetime.now():%Y%m%d-%H:%M:%S}"
        print("checkpoint path already exists,renaming")
    print(f"checkpoint path: {cfg.checkpoint_path}")
    return cfg

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg):
    # cfg = cfg.data  # hydra
    cfg = update_cfg(cfg)
    run_name = Path(cfg.checkpoint_path).name
    set_seed(cfg.seed)
    logger.configure(dir=cfg.checkpoint_path, format_strs=["log", "stdout", "csv"])
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    kwargs = AutocastKwargs(enabled=False)
    # https://github.com/pytorch/pytorch/issues/40497#issuecomment-709846922
    # https://github.com/huggingface/accelerate/issues/2487#issuecomment-1969997224
    
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision=None,
    )
    device = accelerator.device
    accelerate.utils.set_seed(666, device_specific=True)
    rank = accelerator.state.process_index
    logger.log(f"Starting rank={rank}, world_size={accelerator.state.num_processes}, device={device}.")

    if accelerator.is_main_process:
        # W&B is opt-in: set WANDB_MODE=online (+ WANDB_PROJECT/WANDB_ENTITY).
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "SignSparK"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            reinit=True,
            job_type="train",
            mode=os.getenv("WANDB_MODE", "disabled"),
        )
        wandb.config.update(args_to_dict(cfg, cfg.keys()))

    accelerator.wait_for_everyone()
        
    model, _flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    if getattr(cfg, "batch_probe", False):
        _run_batch_probe(model, cfg, accelerator)
        return
    
    load_lmdb = getattr(cfg, "load_lmdb", False)
    logger.log("### Creating data loader...")
    if load_lmdb:
        from signspark.pose_datasets_lmdb import load_lmdb_dataset, infinite_loader
        logger.log("### Using LMDB dataset")
        segments = None

        data_train = load_lmdb_dataset(
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            data_args=cfg,
            split="train",
            segments=segments,
            is_debug=cfg.is_debug,
        )

        data_val = load_lmdb_dataset(
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            data_args=cfg,
            split="dev",
            deterministic=True,
            segments=segments,
        )
    else:
        raise NotImplementedError("Only LMDB dataset is supported in this version. Please set load_lmdb to True.")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data_train, data_val, opt, model = accelerator.prepare(data_train, data_val, opt, model)

    data_train = infinite_loader(data_train)
    data_val = infinite_loader(data_val)
    accelerator.wait_for_everyone()

    TrainLoop_Flow(
        model=model,
        flow=_flow,
        opt=opt,
        accelerator=accelerator,
        data_train=data_train,
        data_val=data_val,
        args=cfg,
        **cfg,
    ).run_loop()


if __name__ == "__main__":
    main()
