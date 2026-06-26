#!/usr/bin/env python3
"""Download the prebuilt SignSparK LMDBs from the HF Hub into ${DATA_ROOT}/lmdb/.

    python tools/download_data.py --datasets CSL-Daily How2Sign --dest ./data
"""

import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "LionelLow/SignSparK_data"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=os.getenv("SIGNSPARK_DATA_REPO", DEFAULT_REPO_ID),
                   help="Hugging Face dataset repo id holding the LMDBs.")
    p.add_argument("--datasets", nargs="+", default=["CSL-Daily", "How2Sign"],
                   help="Which datasets to fetch (matched against the names in split.yaml).")
    p.add_argument("--dest", default=os.getenv("DATA_ROOT", "./data"),
                   help="DATA_ROOT; LMDBs land under <dest>/lmdb/<split>/.")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest) / "lmdb"
    dest.mkdir(parents=True, exist_ok=True)
    # Repo layout is <split>/<Dataset>_*.lmdb/data.mdb; match the requested
    # datasets at any depth.
    patterns = [f"*{d}*" for d in args.datasets]
    print(f"### Downloading {args.datasets} from {args.repo_id} -> {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
    )
    print(f"### Done. Set DATA_ROOT={Path(args.dest).resolve()} (split.yaml reads ${{DATA_ROOT}}/lmdb).")


if __name__ == "__main__":
    main()
