# FAST: Fast and Accurate Sign Language Segmentor

FAST is our temporal sign language segmentor used to build SignSparK's sparse
keyframes. It labels every frame with a BIO tag: `0` non-sign, `2` sign onset,
`1` continuation, following the `segment` field of the LMDB schema
([DATA.md](../DATA.md)).

It ships as part of the **[Sign Language Toolkit](https://sign-language-toolkit.readthedocs.io/)**
(`pip install signlangtk`, import `sltk`). We provide the scripts to wire it to SignSparK, with [`segment.py`](segment.py)
to produce the tags and [`visualize.py`](visualize.py) to inspect them.

> Note: FAST's architecture is based off **[Hands-On](https://github.com/JianHe0628/Hands-On)**

> Note: Additionally, SLTK also covers more than the FAST segmentor. It includes our pose extraction (WiLoR hands, NLF body,
> TEASER face, MediaPipe, RTMPose), SMPL-X fitting, ELAN I/O and evaluation metrics. 
> Worth a look if you require the use of SignSparK on proprietary data.

<div align="center">
  <img src="../assets/FAST_Overview.png" alt="FAST architecture and keyframe selection policy" width="100%">
</div>


## Install

On top of the `signspark` environment. Adds 5 small packages (h5py, hdf5plugin,
nltk, defusedxml, signlangtk) and changes no existing pin:

```bash
pip install signlangtk==0.2.3
```

That is everything needed to segment **WiLoR hand features**. The ~180 MB
segmenter checkpoint downloads to `~/.cache/sltk/weights/` on first use
(`export SLTK_AUTO_DOWNLOAD=1` to skip the prompt).

To segment directly from **raw RGB video**, please install the full extraction stack instead, it requires ~20 more packages
(ultralytics, pytorch-lightning, timm, ...):

```bash
pip install "signlangtk[wilor]==0.2.3"
```


### MANO (RGB video input only)

WiLoR needs two files in one directory:

```
$SLTK_MANO_DIR/
├── MANO_RIGHT.pkl          # mano_v1_2.zip, from mano.is.tue.mpg.de (registration required)
└── mano_mean_params.npz    # from WiLoR: github.com/rolpotamias/WiLoR/tree/main/mano_data
```

Only `MANO_RIGHT.pkl` is needed — WiLoR mirrors left hands — and it is the one
piece we cannot redistribute, so register at
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) for it.
`mano_mean_params.npz` is *not* in that archive; it comes from the
[WiLoR](https://github.com/rolpotamias/WiLoR/tree/main/mano_data) repo and needs
no registration. Then:

```bash
export SLTK_MANO_DIR=/path/to/mano
```

## Usage

One shot, video to keyframes:

```bash
python fast/segment.py clip.mp4 -o segments.pt --keyframes
```

If you already have WiLoR `.h5` features, skip extraction (and MANO entirely) —
pass a file or a directory:

```bash
python fast/segment.py hands.h5   -o segments.pt
python fast/segment.py wilor_h5/  -o segments.pt
```

Export for inspection in ELAN:

```bash
python fast/segment.py clip.mp4 -o segments.eaf --format elan --fps 25 --media clip.mp4
```

Useful flags: `--nlf` (assign hands to the right signer in multi-person video),
`--device`, `--features` (where to keep the extracted `.h5`).

## Visualise

To render the segments on the RGB videos directly, we provide the following script:

```bash
python fast/visualize.py clip.mp4 segments.pt -o viz.mp4
```

The glow fades in and out rather than flashing; `--fade-in` / `--fade-out` (in
frames) control how fast the segments flash. Other arguments also include: `--glow` (border depth), `--alpha` (peak strength), `--color`,
`--width`, `--clip` (which key in `segments.pt`).

## Using the tags

`segments.pt` is `{clip_id: (T,) int64 tensor}`. These tags can be placed into the LMDB
`segment` field without any conversion. SignSparK's dataloader turns them
into sparse conditioning keyframes itself, via `generate_keyframes` in
[`signspark/pose_datasets_lmdb.py`](../signspark/pose_datasets_lmdb.py); the
default `keyframe_selection_mode=3` takes the first, middle and last frame of
each segment. `--keyframes` previews exactly what the loader will derive.

If you build your own LMDB, follow our data schema in
[DATA.md](../DATA.md#2-lmdb-record-schema).

> Note: when writing your own tooling, use SignSparK's `_segment_bounds`, and not SLTK's
> `extract_segments`, as they define the sign segment differently.

## Notes

- **Frame rate.** Native rate is 50 fps with no resampling, but 25 fps works
  fine; `--fps` only affects json/elan timestamps.
- **Do not feed the released LMDB features to FAST.** They are the same shape
  (96 dims/hand) but hold re-optimised SMPL-X fits, whereas FAST was trained on
  raw WiLoR MANO output. That path is off-distribution and unvalidated.
