#!/usr/bin/env python
import argparse, sys, pathlib

def main():
    p = argparse.ArgumentParser(description="Assemble Gold episodes from Silver (stub).")
    p.add_argument("--input-path", type=str, default="data/silver/runner_ts")
    p.add_argument("--output-path", type=str, default="data/gold/episodes")
    p.add_argument("--reward-config", type=str, default="configs/reward.shaping.yaml")
    p.add_argument("--env-config", type=str, default="configs/env.au_preoff.yaml")
    args = p.parse_args()

    in_p = pathlib.Path(args.input_path); out_p = pathlib.Path(args.output_path)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Gold stub � would read Silver from {in_p} and write episodes to {out_p}")
    print(f"[INFO] Using reward config: {args.reward_config} and env config: {args.env_config}")
    print("[INFO] (Stub) implement to_gold.py to write Parquet rollouts with obs/action/reward/done/info.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
