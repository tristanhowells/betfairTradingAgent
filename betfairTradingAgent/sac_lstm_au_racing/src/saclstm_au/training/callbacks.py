# === src/saclstm_au/training/callbacks.py ===
from __future__ import annotations
import os, math, time, datetime, csv
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------
# Utility functions shared across callbacks
# ---------------------------------------------------------------------
def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu().item()
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        return default


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_windowed_metrics_from_env(training_env) -> Dict[str, float]:
    """Safely extract rolling aggregates from env.info_buffer, if available."""
    try:
        infos = training_env.get_attr("info_buffer")
    except Exception:
        infos = []

    flat: List[Dict[str, Any]] = []
    for entry in infos or []:
        if isinstance(entry, (list, tuple)):
            flat.extend([x for x in entry if isinstance(x, dict)])
        elif isinstance(entry, dict):
            flat.append(entry)

    def agg_mean(key: str) -> float:
        vals = [_to_float(d.get(key, float("nan"))) for d in flat]
        vals = [v for v in vals if not math.isnan(v)]
        return float("nan") if len(vals) == 0 else float(np.mean(vals))

    def agg_rate(key: str) -> float:
        vals = [d.get(key, False) for d in flat]
        if len(vals) == 0:
            return float("nan")
        return float(np.mean([1.0 if bool(v) else 0.0 for v in vals]))

    return {
        "mtm_pnl_mean": agg_mean("mtm_pnl_step"),
        "worst_roi_mean": agg_mean("worst_roi_step"),
        "green_rate": agg_rate("green_all_step"),
        "bankrupt_rate": agg_rate("bankrupt_step"),
        "avg_spread_ticks_mean": agg_mean("avg_spread_ticks_step"),
    }


# ---------------------------------------------------------------------
# 1) CSVLoggerCallback  – core structured metrics logger
# ---------------------------------------------------------------------
class CSVLoggerCallback(BaseCallback):
    """
    Logs aggregated metrics to CSV every N steps.
    - Creates unique timestamped filenames automatically.
    - Pulls env.info_buffer stats and selected SB3 metrics.
    """
    def __init__(
        self,
        log_dir: str = "experiments",
        filename_pattern: str = "metrics_%Y%m%d_%H%M%S.csv",
        log_interval: int = 1000,
        rolling_window: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.filename_pattern = filename_pattern
        self.log_interval = int(log_interval)
        self.rolling_window = int(rolling_window)
        self.csv_path = None
        self.writer = None
        self.file = None
        self.start_time = time.time()

    def _init_logger(self):
        _ensure_dir(self.log_dir)
        timestamp = datetime.datetime.now().strftime(self.filename_pattern)
        self.csv_path = os.path.join(self.log_dir, timestamp)
        if not self.csv_path.endswith(".csv"):
            self.csv_path += ".csv"
        if self.verbose:
            print(f"[CSVLogger] Logging to {self.csv_path}")
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        header = [
            "step",
            "elapsed_sec",
            "ep_rew_mean",
            "ep_len_mean",
            "actor_loss",
            "critic_loss",
            "ent_coef",
            "ent_coef_loss",
            "learning_rate",
            "mtm_pnl_mean",
            "worst_roi_mean",
            "green_rate",
            "bankrupt_rate",
            "avg_spread_ticks_mean",
        ]
        self.writer.writerow(header)
        self.file.flush()

    def _gather_metrics(self) -> Dict[str, float]:
        """Collect both rollout/train metrics and env metrics."""
        stats = {
            "ep_rew_mean": _to_float(self.logger.name_to_value.get("rollout/ep_rew_mean", float("nan"))),
            "ep_len_mean": _to_float(self.logger.name_to_value.get("rollout/ep_len_mean", float("nan"))),
            "actor_loss": _to_float(self.logger.name_to_value.get("train/actor_loss", float("nan"))),
            "critic_loss": _to_float(self.logger.name_to_value.get("train/critic_loss", float("nan"))),
            "ent_coef": _to_float(self.logger.name_to_value.get("train/ent_coef", float("nan"))),
            "ent_coef_loss": _to_float(self.logger.name_to_value.get("train/ent_coef_loss", float("nan"))),
            "learning_rate": _to_float(self.logger.name_to_value.get("train/learning_rate", float("nan"))),
        }
        stats.update(_extract_windowed_metrics_from_env(self.training_env))
        return stats

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval != 0:
            return True

        if self.writer is None:
            self._init_logger()

        metrics = self._gather_metrics()
        row = [
            self.num_timesteps,
            time.time() - self.start_time,
            metrics.get("ep_rew_mean"),
            metrics.get("ep_len_mean"),
            metrics.get("actor_loss"),
            metrics.get("critic_loss"),
            metrics.get("ent_coef"),
            metrics.get("ent_coef_loss"),
            metrics.get("learning_rate"),
            metrics.get("mtm_pnl_mean"),
            metrics.get("worst_roi_mean"),
            metrics.get("green_rate"),
            metrics.get("bankrupt_rate"),
            metrics.get("avg_spread_ticks_mean"),
        ]
        self.writer.writerow(row)
        self.file.flush()
        if self.verbose:
            print(f"[CSVLogger] step={self.num_timesteps} logged metrics")
        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()
            if self.verbose:
                print(f"[CSVLogger] Closed {self.csv_path}")


# ---------------------------------------------------------------------
# 2) BestModelCheckpointCallback
# ---------------------------------------------------------------------
class BestModelCheckpointCallback(BaseCallback):
    """
    Save best-so-far model by a selected metric.
    metric_name: "mtm_pnl_mean", "ep_rew_mean", or "-worst_roi_mean".
    """
    def __init__(
        self,
        save_dir: str = "artifacts/models/best",
        metric_name: str = "mtm_pnl_mean",
        check_interval: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.check_interval = int(check_interval)
        self.best_score = None
        _ensure_dir(self.save_dir)

    def _current_metric(self) -> float:
        name = self.metric_name
        negate = False
        if name.startswith("-"):
            negate = True
            name = name[1:]

        val = float("nan")
        if name == "ep_rew_mean":
            try:
                v = self.logger.name_to_value.get("rollout/ep_rew_mean", float("nan"))
                val = _to_float(v, float("nan"))
            except Exception:
                val = float("nan")
        if math.isnan(val):
            env_stats = _extract_windowed_metrics_from_env(self.training_env)
            if name in env_stats:
                val = _to_float(env_stats[name], float("nan"))
        if negate and not math.isnan(val):
            val = -val
        return val

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_interval != 0:
            return True
        score = self._current_metric()
        if math.isnan(score):
            return True
        if (self.best_score is None) or (score > self.best_score):
            self.best_score = score
            path = os.path.join(
                self.save_dir, f"best_{self.metric_name.replace('-', 'neg_')}_{self.num_timesteps}.zip"
            )
            if self.verbose:
                print(f"[BestModel] New best {self.metric_name}={score:.6f} @ {self.num_timesteps}. Saved -> {path}")
            self.model.save(path)
        return True


# ---------------------------------------------------------------------
# 3) PeriodicSnapshotCallback
# ---------------------------------------------------------------------
class PeriodicSnapshotCallback(BaseCallback):
    """Save rolling snapshots every save_interval timesteps."""
    def __init__(self, save_dir: str = "artifacts/models/snapshots", save_interval: int = 20000, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_interval = int(save_interval)
        _ensure_dir(self.save_dir)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            path = os.path.join(self.save_dir, f"ckpt_{self.num_timesteps}.zip")
            if self.verbose:
                print(f"[Snapshot] Saving model snapshot @ {self.num_timesteps} -> {path}")
            self.model.save(path)
        return True


# ---------------------------------------------------------------------
# 4) LRCosineAnnealCallback
# ---------------------------------------------------------------------
class LRCosineAnnealCallback(BaseCallback):
    """Cosine LR schedule with optional warmup."""
    def __init__(
        self,
        total_timesteps: int,
        base_lr: Optional[float] = None,
        min_lr: float = 1e-5,
        warmup_frac: float = 0.05,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total = int(total_timesteps)
        self.base_lr = base_lr
        self.min_lr = float(min_lr)
        self.warmup_frac = float(warmup_frac)

    def _set_lr(self, lr: float) -> None:
        optims = []
        pi = getattr(self.model, "actor", None)
        if pi is not None and hasattr(pi, "optimizer"):
            optims.append(pi.optimizer)
        for name in ["critic", "critic_target"]:
            c = getattr(self.model, name, None)
            opt = getattr(c, "optimizer", None) if c is not None else None
            if opt is not None:
                optims.append(opt)
        for attr_name in dir(self.model):
            if "optim" in attr_name and isinstance(getattr(self.model, attr_name), torch.optim.Optimizer):
                optims.append(getattr(self.model, attr_name))
        for opt in optims:
            for pg in opt.param_groups:
                pg["lr"] = lr

    def _current_lr(self) -> Optional[float]:
        pi = getattr(self.model, "actor", None)
        if pi is not None and hasattr(pi, "optimizer"):
            for pg in pi.optimizer.param_groups:
                return float(pg.get("lr", float("nan")))
        return None

    def _on_training_start(self) -> None:
        if self.base_lr is None:
            cur = self._current_lr()
            self.base_lr = cur if cur and not math.isnan(cur) else 3e-4

    def _on_step(self) -> bool:
        t = min(self.num_timesteps, self.total)
        w_steps = int(self.total * self.warmup_frac)
        if t <= w_steps and w_steps > 0:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (t / w_steps)
        else:
            if self.total <= w_steps:
                lr = self.min_lr
            else:
                prog = (t - w_steps) / max(1, (self.total - w_steps))
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * (1 - prog)))
        self._set_lr(float(lr))
        return True


# ---------------------------------------------------------------------
# 5) TBSummaryCallback
# ---------------------------------------------------------------------
class TBSummaryCallback(BaseCallback):
    """Write custom scalars to TensorBoard."""
    def __init__(self, write_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.write_interval = int(write_interval)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.write_interval != 0:
            return True
        stats = _extract_windowed_metrics_from_env(self.training_env)
        for k, v in stats.items():
            if not math.isnan(_to_float(v, float("nan"))):
                self.logger.record(f"finance/{k}", float(v))
        for key in ["rollout/ep_len_mean", "rollout/ep_rew_mean", "train/actor_loss", "train/critic_loss"]:
            val = self.logger.name_to_value.get(key, None)
            if val is not None:
                fv = _to_float(val, float("nan"))
                if not math.isnan(fv):
                    safe_key = key.replace("rollout/", "tb_rollout/").replace("train/", "tb_train/")
                    self.logger.record(safe_key, fv)
        return True
