#!/usr/bin/env python
import argparse, json, sys, time, pathlib

def main():
    p = argparse.ArgumentParser(description="Ingest Betfair data → Bronze layer (stub).")
    p.add_argument("--login-json", type=str, required=True, help="Path to betfair_login.json")
    p.add_argument("--since", type=str, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--until", type=str, default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--write-path", type=str, default="data/bronze", help="Output root")
    args = p.parse_args()

    login_path = pathlib.Path(args.login_json)
    if not login_path.exists():
        print(f"[ERROR] Login file not found: {login_path}", file=sys.stderr)
        sys.exit(2)
    try:
        creds = json.loads(login_path.read_text())
    except Exception as e:
        print(f"[ERROR] Failed to parse login json: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Ingest stub — would connect as {creds.get('username','<unknown>')} "
          f"for AU/WIN from {args.since} to {args.until or '<open-ended>'}")
    print(f"[INFO] Writing canonicalized parquet to: {args.write_path}")
    print("[INFO] (Stub) No actual API calls are made. Replace with io.betfair_client usage.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
