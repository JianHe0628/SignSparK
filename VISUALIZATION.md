# Visualization

[`tools/visualize.py`](tools/visualize.py) renders SignSparK predictions as
side-by-side GT/prediction SMPL-X mesh videos. It consumes the per-stream `.npy`
files written by [`sample.py`](sample.py) / [`sample_all.py`](sample_all.py) and
reconstructs a full-body signer from the **hand**, **body** and (optional)
**face** streams.

## 1. Get the SMPL-X model

We use the official [`smplx`](https://github.com/vchoutas/smplx) package. The
SMPL-X model files are **license-restricted** and are *not* redistributed here —
download them yourself and follow the license:

* SMPL-X: https://smpl-x.is.tue.mpg.de/ (register, then download "SMPL-X v1.1")

Just unzip the download — its default layout is already what's expected. Point
`--model-folder` at the extracted `models/` folder (the parent of `smplx/`); the
`smplx` package appends the `smplx/` subdir itself:

```
${SMPLX_DIR}/              <- pass this as --model-folder
└── smplx/
    └── SMPLX_NEUTRAL.npz
```

We recommend the `.npz` models (no `chumpy` dependency). The text encoder
(M-CLIP) used for sampling is downloaded automatically from the Hugging Face Hub
and needs no license action.

## 2. Render comparison videos

Pass the per-stream `.npy` that sampling wrote (one per stream, named like
`seed<seed>_clampstep<n>_<note>.npy` under each stream's output dir):

```bash
python tools/visualize.py \
    --body  <body_run>/seed102_clampstep0_CSLDaily.npy \
    --hand  <hand_run>/seed102_clampstep0_CSLDaily.npy \
    --face  <face_run>/seed102_clampstep0_CSLDaily.npy \   # optional
    --model-folder ${SMPLX_DIR} \
    --out   outputs/viz \
    --gap 1 --fps 20
```

This writes one `viz_<clip>.mp4` per clip: ground truth (left) vs prediction
(right), side by side, with a coloured border on keyframe frames. Use `--gap N`
to render every Nth frame (faster previews) and `--max-clips N` to cap how many
clips are processed.

## Metrics

This tool is for **visualization only**. For the quantitative metrics
(MPJPE / PA-MPJPE / DTW and the evaluation protocol used in the paper), use the
official evaluation code at **https://github.com/2000ZRL/SOKE**.

## Face rendering

The face is driven by SMPL-X's native `jaw_pose` + `expression` (50 coeffs) on
the standard SMPL-X mesh. The paper's qualitative figures used **SMPLFX**, our
SMPL-X+FLAME renderer with face stitching
([arXiv:2603.23617](https://arxiv.org/abs/2603.23617)), so **the expression
surface here may look slightly different** (jaw motion and body/hands are
identical). Pass `--no-expression` to render jaw-only if your expression
coefficients are in a different basis.

## Notes

* Rendering is offscreen via EGL (`PYOPENGL_PLATFORM=egl`); a GPU/EGL-capable
  node is recommended.
* The left hand is flipped from SMPLX-left to right-hand convention by default
  (matching the released models); pass `--no-flip-left-hand` if your features
  are already right-handed. See [DATA.md](DATA.md#hand-convention-flip_left_hand).
* This tool produces quick comparison renders. For publication-quality figures,
  export the meshes and render them with a dedicated tool such as
  [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox).
