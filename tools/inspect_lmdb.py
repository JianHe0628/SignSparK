#!/usr/bin/env python3
"""Print an LMDB's clip count and one decoded sample.

    python tools/inspect_lmdb.py ${DATA_ROOT}/lmdb/train/CSL-Daily_reopt_train.lmdb
"""

import argparse
import io
import pickle

import lmdb
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lmdb_path", help="Path to a .lmdb directory.")
    parser.add_argument("--index", type=int, default=0, help="Which clip to decode and print.")
    args = parser.parse_args()

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        meta = pickle.loads(txn.get(b"__meta__"))
        clip_ids = meta.get("clip_ids", [])
        print(f"### {args.lmdb_path}")
        print(f"### num_clips (meta): {meta.get('num_clips')}  | clip_ids listed: {len(clip_ids)}")

        if not clip_ids:
            print("### No clip_ids in meta — nothing to decode.")
            return

        idx = max(0, min(args.index, len(clip_ids) - 1))
        clip_id = clip_ids[idx]
        raw = txn.get(clip_id.encode("utf-8"))
        with io.BytesIO(raw) as buf:
            data = np.load(buf, allow_pickle=True)
            print(f"\n### Clip[{idx}] = {clip_id}")
            print(f"    language    : {data['language'][0]}")
            print(f"    translation : {data['translation'][0]}")
            print(f"    gloss       : {data['gloss'][0]}")
            print(f"    segment     : shape={data['segment'].shape} unique={np.unique(data['segment'])}")
            for f in ["left_features", "right_features", "body_features", "face_features"]:
                print(f"    {f:<14}: shape={data[f].shape} dtype={data[f].dtype}")
    env.close()


if __name__ == "__main__":
    main()
