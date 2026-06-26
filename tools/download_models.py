#!/usr/bin/env python3
"""Download SignSparK checkpoints from the HF Hub into $SIGNSPARK_CKPT_DIR.

    python tools/download_models.py --streams hand body face --dest ./checkpoints
"""

import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "LionelLow/SignSparK"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=os.getenv("SIGNSPARK_MODEL_REPO", DEFAULT_REPO_ID))
    p.add_argument("--streams", nargs="+", default=["hand", "body", "face"])
    p.add_argument("--dest", default=os.getenv("SIGNSPARK_CKPT_DIR", "./checkpoints"))
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        allow_patterns=[f"{s}/*" for s in args.streams],
    )
    print(f"### Done -> {dest.resolve()}  (set SIGNSPARK_CKPT_DIR to this)")


if __name__ == "__main__":
    main()
