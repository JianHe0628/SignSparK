import copy
import functools
import os
import blobfile as bf
import torch
from tqdm import tqdm
import wandb
from signspark.keyframes import get_keyframes_mask
from signspark.utils import dist_util, logger
from signspark.utils.nn import update_ema


EPS = 1e-3
SDE_T = 1.0


def grad_clip(opt, model, max_grad_norm=2.0):
    if hasattr(opt, "clip_grad_norm"):
        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        opt.clip_grad_norm(max_grad_norm)
    else:
        # Revert to normal clipping otherwise, handling Apex or full precision
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
            max_grad_norm,
        )


class TrainLoop_Flow:
    def __init__(
        self,
        *,
        model,
        flow,
        opt,
        accelerator,
        data_train,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        anneal_lr,
        weight_decay=0.0,
        learning_steps=0,
        checkpoint_path="",
        gradient_clipping=-1.0,
        data_val=None,
        eval_interval=-1,
        args=None,
        **kwargs,
    ):
        self.args = args
        self.model = model
        self.flow = flow
        self.accelerator = accelerator
        self.anneal_lr = anneal_lr
        self.data = data_train
        self.eval_data = data_val
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * self.accelerator.state.num_processes

        self.master_params = list(self.accelerator.unwrap_model(self.model).parameters())
        self.checkpoint_path = checkpoint_path
        self.opt = opt
        if self.accelerator.is_main_process:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
        else:
            self.ema_params = []

        logger.log("starting training from scratch")
        logger.log("using keyframe selection mode:", self.args.keyframe_selection_mode)

        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            self._load_model_state()
            self._load_optimizer_state()
            if self.accelerator.is_main_process:
                self.ema_params = [
                    self._load_ema_parameters(rate) for rate in self.ema_rate
                ]

    def _load_model_state(self):
        if not self.resume_checkpoint:
            return
        logger.log(f"loading model from checkpoint: {self.resume_checkpoint}")
        state_dict = dist_util.load_state_dict(
            actual_model_path(self.resume_checkpoint), map_location="cpu"
        )
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(state_dict)
        self.accelerator.wait_for_everyone() 

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint),
            f"opt{parse_resume_step_from_filename(main_checkpoint):06d}.pt",
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                actual_model_path(opt_checkpoint), map_location="cpu"
            )
            self.opt.load_state_dict(state_dict["optimizer"])
            if "step" in state_dict:
                self.resume_step = state_dict["step"]
        else:
            logger.log(f"optimizer checkpoint not found: {opt_checkpoint}")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                actual_model_path(ema_checkpoint), map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def run_loop(self):
        progress_bar = tqdm(desc="training run_loop", total=self.learning_steps)
        while (
            not self.learning_steps
            or self.step + self.resume_step < self.learning_steps
        ):
            batch_embed, batch_dict = next(self.data)
            self.forward_backward(batch_embed, batch_dict)
            if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                logger.dumpkvs()
                logger.log(self.checkpoint_path)
            if (
                self.eval_data is not None
                and self.step % self.eval_interval == 0
                and self.step > 0
            ):
                batch_eval, cond_eval = next(self.eval_data)
                self.do_eval(batch_eval, cond_eval)
                if self.accelerator.is_main_process:
                    logger.log(
                        f"eval on validation set, checkpoint_path = {self.checkpoint_path}"
                    )
                    logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save() 
            self.step += 1
            if self.accelerator.is_main_process:
                wandb.log(dict(global_step=self.step))
            progress_bar.update(1)
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        progress_bar.close()

    @torch.no_grad()
    def do_eval(self, batch_emb, batch_dict):
        for i in range(0, len(batch_emb), self.microbatch):
            micro = batch_emb[i : i + self.microbatch].to(self.accelerator.device)
            micro_dict = {
                k: v[i : i + self.microbatch].to(self.accelerator.device) if torch.is_tensor(v) else v[i : i + self.microbatch]
                for k, v in batch_dict.items()
            }

            cond = {'y': micro_dict}
            cond['obs_x0'] = micro
            cond['obs_mask'] = get_keyframes_mask(data=micro, keyframe_list=cond['y']['keyframes'], joint_prob=self.args.joint_mask_prob)

            if self.args.val_keyframe_mask_prob > 0.:
                random_tensor = torch.rand(cond['obs_mask'].shape[0], device=cond['obs_mask'].device)
                mask_prob = (random_tensor < self.args.val_keyframe_mask_prob).float().view(cond['obs_mask'].shape[0], 1, 1)
                cond['obs_mask'] = (cond['obs_mask'] * (1. - mask_prob)).bool()
            
            if self.args.keyframe_selection_mode != 'random':
                assert torch.all(cond['obs_mask'] == (cond['obs_mask'] * cond['y']['mask']).bool())

            t = (
                torch.rand(len(micro), device=micro.device, dtype=micro.dtype)
                * (SDE_T - EPS)
                + EPS
            )

            compute_losses = functools.partial(
                self.flow.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=cond,
            )

            loss_dict = compute_losses()

            _ = loss_dict.pop("wandb_dict", None)
            loss_dict = {f"eval_{k}": v for k, v in loss_dict.items()}
            loss_gathered = self.accelerator.gather_for_metrics(loss_dict["eval_loss"]).mean()
            loss_dict["loss"] = loss_gathered
            if self.accelerator.is_main_process:
                log_loss_dict(loss_dict)

    def forward_backward(self, batch_embed, batch_dict):
        self.opt.zero_grad()
        for i in range(0, len(batch_embed), self.microbatch):
            _micro_embed = batch_embed[i : i + self.microbatch].to(self.accelerator.device)
            _micro_dict = {
                k: v[i : i + self.microbatch].to(self.accelerator.device) if torch.is_tensor(v) else v[i : i + self.microbatch]
                for k, v in batch_dict.items()
            }
            
            _cond = {'y': _micro_dict}
            ## Keyframe Conditioned 
            _cond['obs_x0'] = _micro_embed # [bs, n_joints=96, n_frames=224]
            _cond['obs_mask'] = get_keyframes_mask(data=_micro_embed, keyframe_list=_cond['y']['keyframes'], joint_prob=self.args.joint_mask_prob) 

            if self.args.train_keyframe_mask_prob > 0.:
                random_tensor = torch.rand(_cond['obs_mask'].shape[0], device=_cond['obs_mask'].device)
                mask_prob = (random_tensor < self.args.train_keyframe_mask_prob).float().view(_cond['obs_mask'].shape[0], 1, 1)
                _cond['obs_mask'] = (_cond['obs_mask'] * (1. - mask_prob)).bool()

            if self.args.keyframe_selection_mode != 'random':
                assert torch.all(_cond['obs_mask'] == (_cond['obs_mask'] * _cond['y']['mask']).bool())

            t = (
                torch.rand(len(_micro_embed), device=_micro_embed.device, dtype=_micro_embed.dtype)
                * (SDE_T - EPS)
                + EPS
            )

            compute_losses = functools.partial(
                self.flow.training_losses,
                self.model,
                _micro_embed,
                t,
                model_kwargs=_cond,
            )

            loss_dict = compute_losses()
            loss = (loss_dict["loss"]).mean()

            wandb_dict = loss_dict.pop("wandb_dict", None)
            wandb_dict.update({k: v for k, v in loss_dict.items()})
            loss_gathered = self.accelerator.gather(loss).mean()
            if self.accelerator.is_main_process:
                wandb_dict["loss"] = loss_gathered.item()
                log_loss_dict(wandb_dict)

            self.accelerator.backward(loss)

        if self.gradient_clipping > 0:
            grad_clip(self.opt, self.accelerator.unwrap_model(self.model), max_grad_norm=self.gradient_clipping)
        self._anneal_lr()
        self.opt.step()

        if self.accelerator.is_main_process:
            for rate, params in zip(self.ema_rate, self.ema_params):
                update_ema(params, self.master_params, rate=rate)
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def _anneal_lr(self):
        if self.anneal_lr == True:
            if not self.learning_steps:
                return
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

        else:
            lr = self.opt.param_groups[0]["lr"]
        if self.accelerator.is_main_process:
            logger.logkv_mean("lr", lr)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if self.accelerator.is_main_process:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                print("writing to", bf.join(get_blob_logdir(), filename))
                print("writing to", bf.join(self.checkpoint_path, filename))

                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                    torch.save(state_dict, f)

        # Save the base model checkpoint regardless of EMA config.
        self.accelerator.wait_for_everyone()  # sync before saving
    
        if self.accelerator.is_main_process:
            save_checkpoint(0, self.master_params)

            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)

            opt_state = {
                "step": self.step + self.resume_step,
                "optimizer": self.opt.state_dict(),
            }
            opt_filename = f"opt{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.checkpoint_path, opt_filename), "wb") as f:
                torch.save(opt_state, f)

        self.accelerator.wait_for_everyone()

    def _master_params_to_state_dict(self, master_params):
        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = unwrapped.state_dict()
        for i, (name, _value) in enumerate(unwrapped.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        unwrapped = self.accelerator.unwrap_model(self.model)
        params = [state_dict[name] for name, _ in unwrapped.named_parameters()]
        return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename[-3:] == ".pt":
        return int(filename[-9:-3])
    else:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        if isinstance(values, torch.Tensor):
            logger.logkv_mean(key, values.mean().item())
        elif isinstance(values, float) or isinstance(values, int):
            logger.logkv_mean(key, values)
        else:
            raise NotImplementedError


def actual_model_path(model_path):
    return model_path