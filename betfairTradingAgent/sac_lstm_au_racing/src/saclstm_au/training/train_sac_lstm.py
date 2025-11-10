diff --git a/betfairTradingAgent/sac_lstm_au_racing/src/saclstm_au/training/train_sac_lstm.py b/betfairTradingAgent/sac_lstm_au_racing/src/saclstm_au/training/train_sac_lstm.py
index 0cc8f9d16f6fb470085cc99a011a86a974c63925..09fd66d7ea12ce050e4058fc68202df2870f4177 100644
--- a/betfairTradingAgent/sac_lstm_au_racing/src/saclstm_au/training/train_sac_lstm.py
+++ b/betfairTradingAgent/sac_lstm_au_racing/src/saclstm_au/training/train_sac_lstm.py
@@ -1,112 +1,147 @@
 # src/saclstm_au/training/train_sac_lstm.py
 from __future__ import annotations
 
 import argparse
 import importlib
 import time
 from pathlib import Path
 from typing import Any, Dict
-
+# Default config shipped with the repo
+DEFAULT_CONFIG_PATH = (
+    Path(__file__).resolve().parents[3] / "configs" / "train_v9_m5_profit_tuned.yaml"
+)
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
 
-def _filtered_env_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
+def _resolve_relative_path(path_str: str | Path, config_path: Path) -> Path:
+    path = Path(path_str)
+    if path.is_absolute():
+        return path
+
+    base_candidates = [
+        config_path.parent,
+        config_path.parent.parent,
+        Path.cwd(),
+    ]
+
+    for base in base_candidates:
+        candidate = (base / path).resolve()
+        if candidate.exists():
+            return candidate
+
+    return (config_path.parent.parent / path).resolve()
+
+
+def _filtered_env_kwargs(raw: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
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
-        out["data_root"] = raw["replay_dir"]
+        out["data_root"] = str(_resolve_relative_path(raw["replay_dir"], config_path))
     if "commission_rate" in raw:
         out["commission"] = raw["commission_rate"]
     if "file_glob" in raw:
         out["file_glob"] = raw["file_glob"]
+    if "per_market_loss_cap_bps" in raw:
+        out["per_market_loss_cap_bps"] = raw["per_market_loss_cap_bps"]
+    if "reject_unaffordable" in raw:
+        out["reject_unaffordable"] = raw["reject_unaffordable"]
+    if "forward_fill" in raw:
+        out["forward_fill"] = raw["forward_fill"]
     for k in ("n_runners", "tick", "tto_max_seconds", "rng_seed"):
         if k in raw:
             out[k] = raw[k]
     return out
 
 
-def make_env_from_cfg(cfg: Dict[str, Any]):
+def make_env_from_cfg(cfg: Dict[str, Any], config_path: Path):
     # Read env target
     env_cfg = cfg.get("env", {}) or {}
     env_id = env_cfg.get("id", "saclstm_au.envs.au_replay_env_v9:AUPreOffEnvV9Replay")
     env_kwargs_raw = env_cfg.get("kwargs", {}) or {}
 
     # Import class
     module_name, class_name = env_id.split(":")
     mod = importlib.import_module(module_name)
     cls = getattr(mod, class_name)
 
     # Map/Filter kwargs for AUPreOffEnvV9Replay
-    kwargs = _filtered_env_kwargs(env_kwargs_raw)
+    kwargs = _filtered_env_kwargs(env_kwargs_raw, config_path)
+
+    data_root = Path(kwargs.get("data_root", ""))
+    if data_root and not data_root.exists():
+        raise FileNotFoundError(
+            "Replay directory not found. Update env.kwargs.replay_dir in the YAML "
+            f"configuration to point at your local parquet dataset: {data_root}"
+        )
 
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
@@ -183,68 +218,72 @@ def build_callbacks(cfg: Dict[str, Any], run_dir: Path, total_timesteps: int):
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
-    ap.add_argument("--config", type=str, default="train_v9_m5_profit_tuned.yaml")
+    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
     # If you want to override run name/output root at launch time:
     ap.add_argument("--run_name", type=str, default=None)
     ap.add_argument("--output_root", type=str, default=None)
     return ap.parse_args()
 
 
 def main():
     args = parse_args()
-    cfg = load_yaml(args.config)
+    config_path = Path(args.config).expanduser()
+    if not config_path.is_absolute():
+        config_path = Path.cwd() / config_path
+    config_path = config_path.resolve()
+    cfg = load_yaml(config_path)
 
     run_name = args.run_name or cfg.get("run_name", time.strftime("exp_%Y%m%d_%H%M%S"))
     output_root = Path(args.output_root or cfg.get("output_root", "experiments"))
     run_dir = output_root / run_name
     (run_dir / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
 
     # Env & Model
-    env = make_env_from_cfg(cfg)
+    env = make_env_from_cfg(cfg, config_path)
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
 
