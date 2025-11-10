# src/saclstm_au/training/train_sac_lstm.py
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC

# Optional TQC (distributional critic)
try:
    from sb3_contrib import TQC
    HAS_TQC = True
except Exception:
    HAS_TQC = False

# ---- Callbacks (imported at module scope to avoid UnboundLocalError) ----
from saclstm_au.training.callbacks import (
    CSVLoggerCallback,
    BestModelCheckpointCallback,
    PeriodicSnapshotCallback,   # available if you want to add it later
    LRCosineAnnealCallback,     # available if you want to add it later
    TBSummaryCallback,          # available if you want to add it later
)

# ---- Env and wrappers ----
from saclstm_au.envs.au_replay_env_v9 import AUPreOffEnvV9Replay
from saclstm_au.envs.wrappers import FlattenObs, FrameStack1D  # if you want these later


# ----------------------------
# YAML helpers
# ----------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------------
# Env construction
# ----------------------------

def _filtered_env_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt YAML kwargs to AUPreOffEnvV9Replay signature and drop unknowns.
    YAML keys accepted:
      - replay_dir            -> data_root
      - commission_rate       -> commission
      - n_runners, tick, tto_max_seconds, rng_seed (if you add them to YAML)
      - file_glob (optional)
      - All others are ignored here.
    """
    out: Dict[str, Any] = {}
    if "replay_dir" in raw:
        out["data_root"] = raw["replay_dir"]
    if "commission_rate" in raw:
        out["commission"] = raw["commission_rate"]
    if "file_glob" in raw:
        out["file_glob"] = raw["file_glob"]
    for k in ("n_runners", "tick", "tto_max_seconds", "rng_seed"):
        if k in raw:
            out[k] = raw[k]
    return out


def make_env_from_cfg(cfg: Dict[str, Any]):
    # Read env target
    env_cfg = cfg.get("env", {}) or {}
    env_id = env_cfg.get("id", "saclstm_au.envs.au_replay_env_v9:AUPreOffEnvV9Replay")
    env_kwargs_raw = env_cfg.get("kwargs", {}) or {}

    # Import class
    module_name, class_name = env_id.split(":")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    # Map/Filter kwargs for AUPreOffEnvV9Replay
    kwargs = _filtered_env_kwargs(env_kwargs_raw)

    env = cls(**kwargs)

    # Optional max-steps wrapper from YAML
    max_steps = env_cfg.get("kwargs", {}).get("max_steps_per_episode", None)
    if isinstance(max_steps, int) and max_steps > 0:
        env = TimeLimit(env, max_episode_steps=int(max_steps))

    # If you want to add FlattenObs/FrameStack1D later, uncomment:
    # env = FlattenObs(env)
    # env = FrameStack1D(env, k=int(cfg.get("wrappers", {}).get("frame_stack", 1)))

    return env


# ----------------------------
# Algo/model construction
# ----------------------------

def make_model_from_cfg(cfg: Dict[str, Any], env):
    agent = cfg.get("agent", {}) or {}
    algo = str(agent.get("algo", "SAC")).upper()
    policy = agent.get("policy", "MlpPolicy")

    common = dict(
        learning_rate=agent.get("learning_rate", 3e-4),
        buffer_size=agent.get("buffer_size", 1_000_000),
        batch_size=agent.get("batch_size", 512),
        tau=agent.get("tau", 0.005),
        gamma=agent.get("gamma", 0.995),
        train_freq=agent.get("train_freq", 64),
        gradient_steps=agent.get("gradient_steps", 1),
        learning_starts=agent.get("learning_starts", 2048),
        seed=cfg.get("train", {}).get("seed", 42),
        policy_kwargs=agent.get("policy_kwargs", {}) or {},
        verbose=1,
    )

    if algo == "TQC":
        if not HAS_TQC:
            raise RuntimeError("agent.algo=TQC requested but sb3_contrib.TQC is not available.")
        model = TQC(policy, env, **common)
    else:
        ent_coef = agent.get("ent_coef", "auto")
        target_entropy = agent.get("target_entropy", "auto")
        model = SAC(policy, env, ent_coef=ent_coef, target_entropy=target_entropy, **common)
    return model


# ----------------------------
# Callbacks wiring
# ----------------------------

def _instantiate_callback_from_yaml(cb_spec: Dict[str, Any], run_dir: Path):
    """
    Supports your YAML style, especially CSVLoggerCallback with
    {log_dir, filename_pattern, log_interval, rolling_window}.
    Everything else is passed through as-is.
    """
    cb_id = cb_spec.get("id")
    kwargs = dict(cb_spec.get("kwargs", {}) or {})

    # Dynamic import
    module_name, class_name = cb_id.split(":")
    mod = importlib.import_module(module_name)
    cb_cls = getattr(mod, class_name)

    # Adapter for our CSVLoggerCallback API
    if cb_cls is CSVLoggerCallback:
        # Map YAML -> actual args
        log_dir = Path(kwargs.pop("log_dir", run_dir / "logs"))
        filename_pattern = kwargs.pop("filename_pattern", "metrics_%Y%m%d_%H%M%S.csv")
        metrics_csv = log_dir / time.strftime(filename_pattern)
        log_freq = int(kwargs.pop("log_interval", 1000))
        # rolling_window is not used by CSVLoggerCallback in callbacks.py; drop if present
        kwargs.pop("rolling_window", None)
        return cb_cls(metrics_csv=metrics_csv, log_freq=log_freq)

    # Pass-through for other callbacks (if you add them via YAML later)
    return cb_cls(**kwargs)


def build_callbacks(cfg: Dict[str, Any], run_dir: Path, total_timesteps: int):
    cbs = []

    # 1) YAML-provided callbacks (CSV logger etc.)
    for cb_spec in cfg.get("callbacks", []) or []:
        try:
            cbs.append(_instantiate_callback_from_yaml(cb_spec, run_dir))
        except Exception as e:
            print(f"[WARN] Failed to instantiate callback {cb_spec.get('id')}: {e}")

    # 2) Best-model checkpoint based on finance metric (interval from train.checkpoint_interval)
    train_cfg = cfg.get("train", {}) or {}
    check_int = int(train_cfg.get("checkpoint_interval", 100_000))
    # Save best on mtm_pnl_mean (flip to '-worst_roi_mean' if you prefer a drawdown-min metric)
    cbs.append(
        BestModelCheckpointCallback(
            save_dir=str(run_dir / "artifacts" / "models" / "best"),
            metric_name="mtm_pnl_mean",
            check_interval=check_int,
            verbose=1,
        )
    )

    # You can uncomment/add more:
    # cbs.append(PeriodicSnapshotCallback(save_dir=str(run_dir / "artifacts" / "models" / "snapshots"),
    #                                     save_interval=check_int, verbose=1))
    # cbs.append(LRCosineAnnealCallback(total_timesteps=total_timesteps, base_lr=None, min_lr=1e-5, warmup_frac=0.05))
    # cbs.append(TBSummaryCallback(write_interval=1000, verbose=0))

    return cbs


# ----------------------------
# Main
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="train_v9_m5_profit_tuned.yaml")
    # If you want to override run name/output root at launch time:
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--output_root", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    run_name = args.run_name or cfg.get("run_name", time.strftime("exp_%Y%m%d_%H%M%S"))
    output_root = Path(args.output_root or cfg.get("output_root", "experiments"))
    run_dir = output_root / run_name
    (run_dir / "artifacts" / "models").mkdir(parents=True, exist_ok=True)

    # Env & Model
    env = make_env_from_cfg(cfg)
    model = make_model_from_cfg(cfg, env)

    # Training schedule
    train_cfg = cfg.get("train", {}) or {}
    total_timesteps = int(train_cfg.get("total_timesteps", 1_000_000))
    log_interval = int(train_cfg.get("log_interval", 1000))

    # Callbacks
    callbacks = build_callbacks(cfg, run_dir, total_timesteps)

    # Train
    print(f"[INFO] Training {model.__class__.__name__} for {total_timesteps:,} timesteps | run_dir={run_dir}")
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=callbacks)

    # Save final checkpoint
    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    save_path = ckpt_cfg.get("save_path", f"artifacts/models/sac_lstm/{run_name}.zip")
    save_path = (run_dir / save_path) if not Path(save_path).is_absolute() else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[INFO] Saved final model to {save_path}")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
