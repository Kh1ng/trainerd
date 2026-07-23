from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="unknown")
    parser.add_argument("--message", default="hello from trainerd")
    args = parser.parse_args()

    out_dir = Path("examples/work") / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": args.version, "message": args.message}
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
