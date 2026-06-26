# Data

SignSparK trains and samples from **prebuilt LMDB databases**. We release the
LMDBs for **CSL-Daily** and **How2Sign** directly — they already contain the 6D
pose features, translations, glosses and segment annotations used in the paper,
so there is **no local build step** for users of this repo.

## 1. Download

```bash
python tools/download_data.py --datasets CSL-Daily How2Sign --dest ./data
export DATA_ROOT=$(pwd)/data
```

This populates the layout the loader expects (globbed from
`configs/data_v2/splits/split.yaml`):

```
${DATA_ROOT}/lmdb/
├── train/   CSL-Daily_reopt_train.lmdb/  How2Sign_reopt_train.lmdb/
├── dev/     CSL-Daily_reopt_dev.lmdb/    ...
└── test/    CSL-Daily_reopt_test.lmdb/   ...
```

Select which dataset a run uses via `train_data` / `dev_data` / `test_data` in
`split.yaml` (names are substring-matched against the `.lmdb` filenames).

Inspect a database to confirm contents:

```bash
python tools/inspect_lmdb.py ${DATA_ROOT}/lmdb/train/CSL-Daily_reopt_train.lmdb
```

## 2. LMDB record schema

Each `.lmdb` is a key/value store with a `__meta__` key
(`pickle.dumps({"clip_ids": [...], "num_clips": N})`) and one key per clip whose
value is an in-memory `.npz` blob (`np.savez` → bytes) with:

| Field | Shape | Type | Notes |
| --- | --- | --- | --- |
| `language` | `(1,)` | str | language tag (prefixed to the text when `specify_lang=True`) |
| `translation` | `(1,)` | str | spoken-language sentence (the generation condition) |
| `gloss` | `(1,)` | str | gloss annotation (`""` if unavailable) |
| `segment` | `(T,)` | int | per-frame segment label in `{0,1,2}`, see below |
| `left_features` | `(T, ≥90)` | float32 | left-hand rot6D (15 × 6) |
| `right_features` | `(T, ≥90)` | float32 | right-hand rot6D (15 × 6) |
| `body_features` | `(T, ≥126)` | float32 | body rot6D; the loader keeps the last 10 joints (`[:, 66:]` → 60) |
| `face_features` | `(T, 56)` | float32 | jaw 6D (6) + expression (50) |

The active modality is chosen by `dataset_feat`: `hand` (90), `body` (60),
`face` (56) or `full` (296 = 90 + 90 + 60 + 56).

### `segment` labels and keyframes

`segment` drives sparse-keyframe selection
([`signspark/pose_datasets_lmdb.py`](signspark/pose_datasets_lmdb.py),
`generate_keyframes`):

* `0` — non-sign / transition frame
* `2` — **first** frame of a sign segment
* `1` — continuation frame within a sign segment

`keyframe_selection_mode` then derives the conditioning keyframes (e.g. mode `3`
takes first/middle/last of each segment).

### Hand convention (`flip_left_hand`)

The released LMDBs store the left hand in true SMPLX-left convention; the
single-hand model operates in right-hand (WiLoR) convention. Set
`flip_left_hand=True` so the loader conjugates the left hand into right-hand
frame — this is also what lets the model **train on both hands** (the left,
once flipped, is an extra right-hand sample).

## 3. How the LMDBs were built (reference only)

For transparency: the released databases are produced by the authors' SMPL-X
fitting + re-optimization pipeline (FAST segmentation → per-clip SMPL-X
parameters → 6D features + segment annotations), which is a separate component
described in the paper and **not required to use this repo**. If you want to
build LMDBs from your own data, replicate the record schema above
(`np.savez` the eight fields per clip; add the `__meta__` index).

## 4. Datasets & licensing

Obtain CSL-Daily and How2Sign from their original sources and follow their
licenses; the released LMDBs contain SMPL-X pose features derived from these
corpora and are provided for research use under the same terms.
