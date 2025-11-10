#!/usr/bin/env python
import argparse, sys, pathlib, yaml, time

def main():
    p = argparse.ArgumentParser(description="Paper-trading harness (stub).")
    p.add_argument("--config", type=str, default="configs/execution.paper.yaml")
    args = p.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Execution config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text())

    out_dir = pathlib.Path(cfg.get("logging",{}).get("out_dir","experiments/paper_runs"))
    run = time.strftime("paper_%Y%m%d_%H%M%S")
    run_dir = out_dir / run
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Paper-trade stub run: {run_dir}")
    print(f"[INFO] Risk: per_market_loss_bps_cap={cfg['risk']['per_market_loss_bps_cap']} "
          f"exposure_band={cfg['risk']['per_runner_exposure_band']} reject_unaffordable={cfg['risk']['reject_unaffordable_orders']}")
    print("[INFO] (Stub) Replace with exec.paper_adapter integration and live event loop.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
