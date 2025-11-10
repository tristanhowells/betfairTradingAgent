#!/usr/bin/env python
import argparse, sys, pathlib

def main():
    p = argparse.ArgumentParser(description="Build Silver features from Bronze (stub).")
    p.add_argument("--input-path", type=str, default="data/bronze")
    p.add_argument("--output-path", type=str, default="data/silver/runner_ts")
    p.add_argument("--bar-seconds", type=int, default=1)
    args = p.parse_args()

    in_p = pathlib.Path(args.input_path); out_p = pathlib.Path(args.output_path)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Silver stub � would read from {in_p} and write aligned features to {out_p}")
    print(f"[INFO] Using bar size = {args.bar_seconds}s; (Stub) implement ETL in etl/to_silver.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
