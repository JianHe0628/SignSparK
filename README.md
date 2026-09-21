<div align="center">

# SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning

**[Jianhe Low](mailto:jianhe.low@surrey.ac.uk), Alexandre Symeonidis-Herzig, Maksym Ivashechkin, Özge Mercanoğlu Sincan, Richard Bowden**

Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey

**ECCV 2026**

[![Paper](https://img.shields.io/badge/arXiv-2603.10446-b31b1b.svg)](https://arxiv.org/abs/2603.10446)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://cogvis-cvssp.github.io/papers/signspark/)

<img src="assets/signspark_qualitatives.gif" alt="SignSparK qualitative results" width="100%">

*Full-resolution video on the [project page](https://cogvis-cvssp.github.io/papers/signspark/).*

</div>

---

## Overview

**SignSparK** is a Conditional Flow Matching framework for **multilingual Sign
Language Production (SLP)**. Instead of regressing dense pose sequences, which
collapses toward the mean and yields under-articulated signing, SignSparK
learns from **sparse keyframes** that capture the underlying kinematic
distribution of human signing, then synthesises fluid 3D signing sequences
conditioned on spoken-language text and those keyframes (Keyframe-to-Pose
generation). The framework supports four sign languages and achieves
state-of-the-art results across multiple benchmarks.

This repository contains the **flow-matching training and sampling code**. The
model is trained per body stream (i.e. **hand**, **body**, and **face**), which are
combined to produce the full signer.

> 📄 Paper: https://arxiv.org/abs/2603.10446 &nbsp;·&nbsp; 🌐 Project page: https://cogvis-cvssp.github.io/papers/signspark/

### Release status

- [x] **Datasets**: prebuilt LMDBs for CSL-Daily, How2Sign and PHOENIX-2014T · 🤗 [LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data)
- [x] **SignSparK checkpoints**: hand / body / face streams · 🤗 [LionelLow/SignSparK](https://huggingface.co/LionelLow/SignSparK)
- [x] **Back-translation code and weights**: [SignSparK-BT Repo](https://github.com/JianHe0628/SignSparK_BT) · 🤗 [LionelLow/SignSparK_BT](https://huggingface.co/LionelLow/SignSparK_BT)
- [ ] **FAST keyframe segmentor** — *pending toolkit embargo, expected end of September 2026* ([details](#fast-keyframe-segmentation))

## Installation

Python 3.10, PyTorch 2.10 with **CUDA 12.8** (required for Blackwell / RTX
50-series GPUs; works on older GPUs too):

```bash
conda env create -f environment.yml
conda activate signspark
```

Or with pip into your own environment:

```bash
pip install -r requirements.txt
```

> Both pull PyTorch from the `cu128` wheel index. For a different CUDA version,
> change the index URL (e.g. `.../whl/cu121`) and drop the `+cu128` tags in
> `environment.yml` / `requirements.txt`. Do **not** upgrade `huggingface_hub`
> past `<1.0` — `transformers 4.56.1` requires it.

## Data & checkpoints

SignSparK reads pose data from **LMDB** databases. We release prebuilt LMDBs for
**CSL-Daily**, **How2Sign** and **PHOENIX-2014T** (6D pose features +
translations + glosses + segment annotations). Thus, no local build step is required:

> **Note:** The released datasets and checkpoints are higher quality than those used in the paper. Datasets have been reoptimized for improved contact points, while checkpoints have been trained on 15x more data, so reproduced results may exceed the originally reported numbers.

```bash
python tools/download_data.py --datasets CSL-Daily How2Sign PHOENIX-2014T --dest ./data
export DATA_ROOT=$(pwd)/data
python tools/inspect_lmdb.py ${DATA_ROOT}/lmdb/train/CSL-Daily_reopt_train.lmdb  # sanity-check
```

Pretrained checkpoints (per stream) download into `$SIGNSPARK_CKPT_DIR`:

> **Note:** Our official checkpoint is trained on CSL-Daily, How2Sign, BOBSL, and CSL-News. PHOENIX14T is excluded due to poor 3D estimation quality on its low-resolution footage, though the processed data remains available on HuggingFace.

```bash
python tools/download_models.py --streams hand body face --dest ./checkpoints
export SIGNSPARK_CKPT_DIR=$(pwd)/checkpoints   # -> <stream>/ema_0.9999_<iter>.pt
```

See **[DATA.md](DATA.md)** for the record schema, directory layout, segment
labels, and the left-hand convention. Datasets and checkpoints live on the Hub:

> 🤗 Data: [LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data) · Models: [LionelLow/SignSparK](https://huggingface.co/LionelLow/SignSparK) · Back-translation: [LionelLow/SignSparK_BT](https://huggingface.co/LionelLow/SignSparK_BT)

Configure paths via environment variables (copy `.env.example` → `.env`):

| Variable | Meaning | Default |
| --- | --- | --- |
| `DATA_ROOT` | root holding `lmdb/<split>/*.lmdb` (downloaded data) | `./data` |
| `SIGNSPARK_CKPT_DIR` | released checkpoints, `{hand,body,face}/ema_*.pt` (read for sampling) | `./checkpoints` |
| `OUTPUT_DIR` | where **training** writes its runs/checkpoints | `./outputs` |

`OUTPUT_DIR` (training output) and `SIGNSPARK_CKPT_DIR` (released weights for
sampling) are deliberately separate — to sample your own freshly-trained model,
point `eval.model_path=` at the file under `OUTPUT_DIR`.

## FAST keyframe segmentation

**FAST**, the temporal sign language segmentor behind SignSparK's sparse
keyframes, labels every frame `0` (non-sign), `2` (sign onset) or `1`
(continuation), following the `segment` field of the LMDB schema.

<div align="center">
  <img src="assets/FAST_Overview.png" alt="FAST architecture and keyframe selection policy" width="100%">
</div>

FAST is currently deployed as an internal toolkit and is under embargo until the
end of September 2026. The public release, along with the scripts that wire it
to SignSparK, will follow here. The currently released LMDBs already carry the `segment`
field, so the repo currently still works without the toolkit.

## Training

Configs are managed with [Hydra](https://hydra.cc/); select the body stream with
`data_v2=<hand|body|face|full>` and override any key on the CLI.

```bash
# Single GPU
python train.py data_v2=hand

# Multi-GPU (via 🤗 Accelerate)
accelerate launch train.py data_v2=hand
```

Train each stream separately (`hand`, `body`, `face`). Checkpoints (including EMA
weights) are written under `${OUTPUT_DIR}`. Common overrides:

```bash
python train.py data_v2=body batch_size=128 learning_steps=500000 \
                train_data='[CSL-Daily]' base_path=${DATA_ROOT}/lmdb
```

Logging is **opt-in**: metrics go to Weights & Biases only if you set
`WANDB_MODE=online` (and `WANDB_PROJECT` / `WANDB_ENTITY`); otherwise it runs
disabled.

## Sampling

Generate poses for a single stream using the inference config and a checkpoint:

```bash
python sample.py data_v2=hand data_v2/common@common=inference \
                 eval.model_path=${SIGNSPARK_CKPT_DIR}/hand/ema_0.9999_200000.pt \
                 eval.split=test eval.ode_stepnum=50
```

Or sample all three streams sequentially with `sample_all.py` (edit its `CONFIG`
dict, or pass the same overrides on the CLI — CLI wins):

```bash
python sample_all.py eval.note=CSLDaily eval.ode_stepnum=50 test_data='[CSL-Daily]'
```

Key sampling options (`eval.*`, set in the inference config or overridden):

| Option | Meaning |
| --- | --- |
| `eval.split` | which split to sample: `test` / `dev` / `train` |
| `test_data` (or `dev_data`/`train_data`) | **which dataset** for that split, e.g. `'[CSL-Daily]'`, `'[How2Sign]'` or `'[PHOENIX-2014T]'` — the eval set is chosen by this list in `split.yaml`, *not* a `dataset=` arg |
| `eval.ode_stepnum` | number of ODE sampling steps |
| `eval.candidate_num` | samples drawn per clip |
| `eval.max_samples` | cap clips sampled (`-1` = whole split) |
| `eval.model_path` | checkpoint to load (per-stream) |
| `flip_left_hand` | flip left hand to right-hand frame if your LMDB stores SMPLX-left (see [DATA.md](DATA.md#hand-convention-flip_left_hand)) |

`sample_all.py` resolves per-stream checkpoints from
`${SIGNSPARK_CKPT_DIR}/<stream>/ema_0.9999_<iter>.pt` and runs the streams one at
a time to keep peak VRAM at a single stream's footprint.

**Classifier-free guidance / text-free sampling.** Set `eval.text_guidance_scale`
above `0` to apply text guidance (uses [`classifier_free.py`](classifier_free.py)).
For fully unconditional (no keyframe) sampling, set `eval.keyframes_mask_all=True`.
With `eval.text_guidance_scale=-1` guidance is off.

Outputs are written as `.npy` dictionaries containing predicted poses, lengths,
keyframe masks, conditioning text, and ground-truth poses.

## Visualization

[`tools/visualize.py`](tools/visualize.py) turns the sampled `.npy` files into
side-by-side (ground truth vs prediction) **SMPL-X mesh videos**:

```bash
python tools/visualize.py --body body_eval.npy --hand hand_eval.npy \
                          --face face_eval.npy --model-folder ${SMPLX_DIR} \
                          --out outputs/viz
```

It uses the official [`smplx`](https://github.com/vchoutas/smplx) package; you
must download the (license-gated) SMPL-X model files yourself. See
**[VISUALIZATION.md](VISUALIZATION.md)** for setup and options.

**Metrics.** For the quantitative evaluation protocol (MPJPE / PA-MPJPE / DTW),
use the official evaluation code at
[github.com/2000ZRL/SOKE](https://github.com/2000ZRL/SOKE). For back-translation
metrics (BLEU / chrF / ROUGE), see below.

## Back-translation evaluation

**[SignSparK-BT](https://github.com/JianHe0628/SignSparK_BT)** is the evaluation
companion to this repo: it translates generated SMPL+MANO poses back to spoken
language and scores them with BLEU, chrF and ROUGE. It reads the **same LMDBs**
as SignSparK, so `DATA_ROOT` can carry without changes.

```bash
git clone https://github.com/JianHe0628/SignSparK_BT.git && cd SignSparK_BT
conda env create -f environment.yml && conda activate signspark-bt && pip install -e .

# same DATA_ROOT as SignSparK; back-translation weights are a separate download
export DATA_ROOT=/path/to/signspark/data
python tools/download_models.py --datasets PHOENIX-2014T CSL-Daily How2Sign --dest ./checkpoints
```

Score a `sample_all.py` run by pointing it at the **body** and **hand** dumps
from that same run. Note that the face stream is unused, so the model takes body 60 + both
hands 90 each = 240 dims. Note also that `sample.py` writes each dump *next to the checkpoint
it loaded*, under `<ckpt dir>/debug0_eulerstepsize<steps>_can<n>_anchor<0|1>_samples/`:

```bash
RUN=debug0_eulerstepsize50_can1_anchor0_samples
signspark-bt score checkpoints/PHOENIX-2014T \
    --body-npy ${SIGNSPARK_CKPT_DIR}/body/${RUN}/seed123_clampstep0_PHOENIX14T.npy \
    --hand-npy ${SIGNSPARK_CKPT_DIR}/hand/${RUN}/seed123_clampstep0_PHOENIX14T.npy
```

(the `seed` and trailing name come from `eval.seed`, `eval.clamp_step` and
`eval.note` of that run.)

It scores ground-truth and generated poses in one pass and reports the drop
between them. Released models and the updated score tables:
🤗 [LionelLow/SignSparK_BT](https://huggingface.co/LionelLow/SignSparK_BT) ·
[full README](https://github.com/JianHe0628/SignSparK_BT#scoring-sign-language-production).

> **Note:** The released back-translation models are retrained on the
> reoptimized LMDBs, so both the ground-truth ceiling and the reported drop
> differ from the paper.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{low2026signspark,
  title={SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning},
  author={Low, Jianhe and Symeonidis-Herzig, Alexandre and Ivashechkin, Maksym and Sincan, Ozge Mercanoglu and Bowden, Richard},
  booktitle={European Conference on Computer Vision},
  pages={648--670},
  year={2026},
  organization={Springer}
}
```

## Acknowledgements

The flow-matching training scaffold builds on
ideas from [FlowSeq](https://github.com/dongzhuoyao/flowseq), while evaluation metrics come from [SOKE](https://github.com/2000ZRL/SOKE). We sincerely thank the authors for open-sourcing their codebases.

## License

The **code** in this repository is released under the [Apache License 2.0](LICENSE).

The released **datasets and model checkpoints** are derived from CSL-Daily,
How2Sign, PHOENIX-2014T and BOBSL and are provided for **non-commercial research
use only**,
under the terms of those source datasets.
