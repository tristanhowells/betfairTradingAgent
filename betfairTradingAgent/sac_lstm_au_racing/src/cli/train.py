#!/usr/bin/env python
import argparse, sys, pathlib, shutil, time, csv, json

def main():
    p = argparse.ArgumentParser(description="Train SAC-LSTM agent (stub scaffold).")
    p.add_argument("--config", type=str, default="configs/model.sac_lstm.small.yaml")
    p.add_argument("--run", type=str, default=None, help="Optional run name")
    args = p.parse_args()

    run = args.run or time.strftime("exp_%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(f"experiments/{run}")
    charts = run_dir / "charts"; run_dir.mkdir(parents=True, exist_ok=True); charts.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "metrics.csv"

    # Freeze config for reproducibility if present
    cfg_src = pathlib.Path(args.config)
    if cfg_src.exists():
        shutil.copy2(cfg_src, run_dir / "config.freeze.yaml")

    # Seed a metrics CSV with headers
    if not metrics_csv.exists():
        with metrics_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step","green_rate","worst_roi_p95","mtm_roi_mean","orders_submitted","orders_filled"])

    print(f"[INFO] Training stub started. Run dir: {run_dir}")
    print("[INFO] (Stub) Plug in src.saclstm_au.training.train_sac_lstm to run SB3 training.")
    print("[INFO] Writing metrics to:", metrics_csv)
    return 0

if __name__ == "__main__":
    sys.exit(main())
