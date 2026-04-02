from __future__ import annotations

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
from tkinter.scrolledtext import example
import statistics

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

from tabicl import TabICL
from tabicl.prior.dataset import PriorDataset
from tabicl.prior.genload import LoadPriorDataset
from tabicl.train.optim import get_scheduler
from tabicl.train.muon import Muon
from tabicl.train.train_config import build_parser
from transformers.optimization import Adafactor

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        self.step_time_list = []
        self.avg_step_1k = None

    def print_1k_step_avg(self):
        """打印前1000步平均时间，仅主进程打印"""
        if not self.master_process:
            return
        if len(self.step_time_list) >= 1000 and self.avg_step_1k is None:
            total = sum(self.step_time_list[:1000])
            self.avg_step_1k = total / 1000
            print(f"\n===== 前1000步平均耗时 =====")
            print(f"总时间: {total:.2f}s")
            print(f"平均每步: {self.avg_step_1k:.4f}s")
            print(f"速度: {1 / self.avg_step_1k:.2f} step/s\n")


    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the TabICL model."""

        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
        }

        model = TabICL(**self.model_config)
        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # Freeze model components if requested
        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            print("Model compile is TRUE")
            if self.master_process:
                print("Model compiled successfully.")
        self.model = model


    def configure_prior(self):
        """
        Sets up a tabular dataset generator that creates synthetic datasets
        during training with controllable properties and data distributions.
        """

        if self.config.prior_dir is None:
            # Generate prior data on the fly
            dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
            )
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        # Create dataloader for efficient loading and prefetching
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device if self.config.prior_device == "cpu" else "",
        )

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""
        row_params = []
        other_params = []
        for name, p in self.raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("row_interactor."):
                row_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {"params": other_params, "lr": self.config.lr},
            {"params": row_params, "lr": self.config.lr * 1.0},
        ]

        optimizer_name = getattr(self.config, "optimizer", "adamw").lower()
        if optimizer_name == "muon":
            # Keep the same key knobs as AdamW (lr/weight_decay).
            self.optimizer = Muon(
                param_groups,
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        elif optimizer_name == "adafactor":
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=self.config.lr,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=self.config.weight_decay,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            # self.amp_ctx = torch.autocast(
            #     device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            # )
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "bfloat16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        # checkpoint_path = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/good.ckpt"
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        # checkpoint_path = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/checkpoint/tabicl-classifier-v1.1-0506.ckpt"
        checkpoint_path = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/checkpoint/dir1/step-100400.ckpt"
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
        state_dict = checkpoint["state_dict"]

        # 这里的逻辑：如果 config 中没有指定具体 load 哪个，默认全加载
        load_col = getattr(self.config, "load_col", True)
        load_row = getattr(self.config, "load_row", True)
        load_icl = getattr(self.config, "load_icl", True)

        def filter_and_load(module, prefix):
            # 提取以 prefix 开头的权重，并去除 prefix 前缀
            filtered_dict = {
                k[len(prefix) + 1 :]: v 
                for k, v in state_dict.items() 
                if k.startswith(prefix)
            }
            if filtered_dict:
                module.load_state_dict(filtered_dict)
                print(f"  Successfully loaded weights for: {prefix}")
            else:
                print(f"  Warning: No weights found for {prefix} in checkpoint")

        # 按需加载
        if load_col: filter_and_load(self.raw_model.col_embedder, "col_embedder")
        if load_row: filter_and_load(self.raw_model.row_interactor, "row_interactor")
        if load_icl: filter_and_load(self.raw_model.icl_predictor, "icl_predictor")

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        # self.raw_model.load_state_dict(checkpoint["state_dict"])

        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            # self.curr_step = checkpoint["curr_step"]
            # print(f"Resuming training at step {self.curr_step}")

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                # try:
                #     os.remove(ckpt_path)
                # except Exception as e:
                #     print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        iterator = iter(self.dataloader)
        if self.master_process:
            prog = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            prog = range(self.curr_step, self.config.max_steps)

        for step in prog:
            # 整步总耗时
            with Timer() as step_timer:
                with Timer() as prior_timer:
                    batch = next(iterator)
                with Timer() as train_timer:
                    results = self.run_batch(batch)
            self.curr_step = step + 1

            # 记录前1000步时间
            if self.curr_step <= 1000:
                self.step_time_list.append(step_timer.elapsed)

            if self.master_process:
                if self.curr_step == 1000:
                    self.print_1k_step_avg()
                results.update({"prior_time": prior_timer.elapsed, "train_time": train_timer.elapsed})
                if isinstance(prog, tqdm):
                    prog.set_postfix(**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in results.items()})
                if self.curr_step % self.config.save_temp_every == 0 or self.curr_step % self.config.save_perm_every == 0:
                    self.save_checkpoint(f"step-{self.curr_step}.ckpt")
                    if self.curr_step % self.config.save_temp_every == 0 and self.curr_step % self.config.save_perm_every != 0:
                        if self.config.max_checkpoints > 0:
                            self.manage_checkpoint()

            # if self.wandb is not None:
            #     results["lr"] = self.scheduler.get_last_lr()[0]
            #     self.wandb.log(results, step=self.curr_step)

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")
        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")
        return micro_seq_len[0].item(), micro_train_size[0].item()

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        if micro_X.shape[1] > seq_len: micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len: micro_y = micro_y[:, :seq_len]
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features: micro_X = micro_X[..., :max_features]
        return micro_X, micro_y

    def get_module_grad_norm(self, module):
        """计算特定模块所有参数梯度的 L2 范数"""
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)  # (B, test_size, C)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {"ce": scaled_loss.item(), "accuracy": (pred.argmax(dim=1) == true).float().mean().item() / num_micro_batches}
        return micro_results
        """
    
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)
        
        micro_X = micro_X.to(self.config.device)
        # micro_X = torch.clamp(micro_X, min=-10.0, max=10.0)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test  = micro_y[:, train_size:]

        # early exit if nothing to predict
        if y_test.numel() == 0:
            return {"ce": 0.0, "accuracy": 0.0}

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            logits = self.model(micro_X, y_train, micro_d)  # (B, Ttest, C)
            B, T, C = logits.shape
            pred  = logits.reshape(-1, C)
            true  = y_test.reshape(-1).long()

            # drop any labels outside [0, C-1] (corrupt/padded labels)
            valid = (true >= 0) & (true < C)
            if not torch.all(valid):
                true = true[valid]
                pred = pred[valid]
            if true.numel() == 0:
                return {"ce": 0.0, "accuracy": 0.0}

            loss = F.cross_entropy(pred, true)

        # if loss blew up, abort this micro and let caller skip the step
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
            
        
        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {
                "ce": scaled_loss.item(),
                "accuracy": (pred.argmax(dim=1) == true).float().mean().item() / num_micro_batches,
            }
        return micro_results

    def run_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # 记录本 step 用来更新参数的 lr（scheduler.step() 之前）
        lr = self.optimizer.param_groups[0]["lr"]

        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # split into micro-batches
        splits = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        all_micros = list(zip(*splits))

        # keep only micros that actually have ANY test rows (seq_len > train_size)
        valid_micros = []
        for mb in all_micros:
            _, _, _, micro_seq_len, micro_train_size = mb
            seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
            if seq_len > train_size:
                valid_micros.append(mb)

        num_micro_batches = len(valid_micros)
        if num_micro_batches == 0:
            # 没有反传也没更新；这里你原本就 step scheduler，我保留
            self.scheduler.step()
            results = {"ce": 0.0, "accuracy": 0.0, "lr": lr, "lr_next": self.optimizer.param_groups[0]["lr"]}
            return results

        micro_batches = valid_micros

        results = {"ce": 0.0, "accuracy": 0.0}
        failed = 0
        for i, micro in enumerate(micro_batches):
            try:
                res = self.run_micro_batch(micro, i, num_micro_batches)
                for k, v in res.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                torch.cuda.empty_cache()
                failed += 1
                continue
            except FloatingPointError:
                print(f"Non-finite loss in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                failed += 1
                continue

        # ---------- 关键：unscale -> clip -> step，保证 clip 影响本次更新 ----------
        # 先把 AMP 缩放过的梯度还原到真实尺度
        # print(f"Scaler _scale: {self.scaler._scale}")
        self.scaler.unscale_(self.optimizer)

        # 2. 计算三个特定组件的梯度范数
        # 使用 self.raw_model 以确保无论是否开启 DDP 都能正确访问子模块
        with torch.no_grad():
            gnorm_col = self.get_module_grad_norm(self.raw_model.col_embedder)
            gnorm_row = self.get_module_grad_norm(self.raw_model.row_interactor)
            gnorm_icl = self.get_module_grad_norm(self.raw_model.icl_predictor)
        # 收集有 grad 的参数
        params = [p for p in self.model.parameters() if p.grad is not None]
        if len(params) == 0:
            # 没梯度就不更新（但你可以选择仍然 step scheduler；这里保留你的行为）
            self.scaler.update()
            self.scheduler.step()
            results.update({"grad_norm_pre": 0.0, "grad_norm_post": 0.0, "lr": round(lr, 5), "lr_next": self.optimizer.param_groups[0]["lr"]})
            return results

        # 计算 pre-clip grad norm
        with torch.no_grad():
            grad_norm_pre = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2)


        # clip（真正修改梯度，保证对更新生效）
        if self.config.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(params, self.config.gradient_clipping)

        # 计算 post-clip grad norm（这是“更新时实际用到”的梯度范数）
        with torch.no_grad():
            grad_norm_post = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2)

        # ---------- spike / NaN 检查（我用 post-clip 来判断更合理） ----------
        if not hasattr(self, "grad_norm_ema"):
            self.grad_norm_ema = grad_norm_post.item()

        ema_decay = 0.90
        spike_threshold = 10.0
        warmup_steps_skip = 500

        is_nan_inf = not torch.isfinite(grad_norm_post)
        is_spike = (
            (not is_nan_inf)
            and (self.curr_step > warmup_steps_skip)
            and (grad_norm_post.item() > self.grad_norm_ema * spike_threshold)
        )

        if is_nan_inf or is_spike:
            if self.master_process:
                what = "NaN/Inf" if is_nan_inf else "Spike"
                print(f"[Warning] Step {self.curr_step}: {what} (post-clip={grad_norm_post.item():.2f}). Skipping update.")

            # 清梯度，推进 scaler（避免下次 unscale 报错）
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()

            # 你原来跳过也 step scheduler，我保留（但如果你想“跳过就不走 scheduler”，把下一行注释掉）
            self.scheduler.step()

            results.update({
                "grad_norm_pre": grad_norm_pre.item() if torch.isfinite(grad_norm_pre) else 0.0,
                "grad_norm_post": grad_norm_post.item() if torch.isfinite(grad_norm_post) else 0.0,
                "lr": round(lr, 3),
                # "lr_next": self.optimizer.param_groups[0]["lr"],
                "ema": self.grad_norm_ema,
            })
            return results

        # 更新 EMA（用 post-clip）
        self.grad_norm_ema = ema_decay * self.grad_norm_ema + (1 - ema_decay) * grad_norm_post.item()

        # ---------- 真正参数更新 ----------
        self.scaler.step(self.optimizer)   # 等价于 optimizer.step()（若检测到 inf 会自动跳过）
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # scheduler 在更新后 step（常见做法）
        self.scheduler.step()

        # results.update({
        #     "grad_norm_pre": grad_norm_pre.item(),
        #     "grad_norm_post": grad_norm_post.item(),
        #     "lr": lr,                                  # 本次更新用的 lr
        #     "lr_next": self.optimizer.param_groups[0]["lr"],  # 下一步的 lr
        #     "ema": self.grad_norm_ema,
        # })
        results.update({
            "gnorm_col": gnorm_col,
            "gnorm_row": gnorm_row,
            "gnorm_icl": gnorm_icl,
            "grad_norm_pre": grad_norm_pre.item(),
            # "grad_norm_post": grad_norm_post.item(),
            "lr": round(lr, 5),
            # "lr_next": self.optimizer.param_groups[0]["lr"]
        })
        return results


    # def run_batch(self, batch):
    #     self.model.train()
    #     self.optimizer.zero_grad(set_to_none=True)

    #     batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        
    #     """
    #     num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
    #     micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
    #     micro_batches = list(zip(*micro_batches))
    #     """
    #     # split into micro-batches
    #     splits = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
    #     all_micros = list(zip(*splits))

    #     # keep only micros that actually have ANY test rows (seq_len > train_size)
    #     valid_micros = []
    #     for mb in all_micros:
    #         _, _, _, micro_seq_len, micro_train_size = mb
    #         seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
    #         if seq_len > train_size:
    #             valid_micros.append(mb)

    #     num_micro_batches = len(valid_micros)
    #     if num_micro_batches == 0:
    #         # nothing to backprop this step; advance scheduler and report zeros
    #         self.scheduler.step()
    #         return {"ce": 0.0, "accuracy": 0.0}

    #     micro_batches = valid_micros
        
    #     results = {"ce": 0.0, "accuracy": 0.0}
    #     failed = 0
    #     for i, micro in enumerate(micro_batches):
    #         try:
    #             res = self.run_micro_batch(micro, i, num_micro_batches)
    #             for k, v in res.items(): results[k] += v
    #         except torch.cuda.OutOfMemoryError:
    #             print(f"OOM in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
    #             torch.cuda.empty_cache(); failed += 1; continue
    #         except FloatingPointError as e:
    #             print(f"Non-finite loss in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
    #             failed += 1; continue

    #     # if failed / max(1, len(micro_batches)) > 0.1:
    #     #     raise RuntimeError("Too many failed micro-batches. Reduce memory usage or check data quality.")
    #     grad_norm = 0.0
    #     if self.config.gradient_clipping > 0:
    #         self.scaler.unscale_(self.optimizer)
    #         total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

    #         is_too_huge = total_norm > (self.config.gradient_clipping * 10) if self.config.gradient_clipping > 0 else False
    #         if not torch.isfinite(total_norm): #or is_too_huge:
    #             # bad grads: skip the update but keep schedule moving
    #             if self.master_process:
    #                 print(f"Non-finite grad norm at step {self.curr_step}; skipping optimizer step.")
    #             self.optimizer.zero_grad(set_to_none=True)
    #             self.scaler.update()
    #             self.scheduler.step()
    #             results["grad_norm"] = 0.0 # 或者 float('nan')
    #             return results
        
    #     grad_norm_val = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     self.optimizer.zero_grad(set_to_none=True)
    #     self.scheduler.step()
    #     results["grad_norm"] = grad_norm_val

        # return results


if __name__ == "__main__":
    parser = build_parser()
    cfg = parser.parse_args()
    import torch
    torch.set_float32_matmul_precision('high')  #AMD 不支持
    trainer = Trainer(cfg)
    trainer.train()


#SBATCH -w auh7-1b-gpu-[226-227,309-314]
