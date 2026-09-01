#!/usr/bin/env python3
"""Run FAST sign segmentation on a video (or on WiLoR hand features).

Emits per-frame BIO tags (0 = non-sign, 2 = sign onset, 1 = continuation) —
the same convention as the LMDB `segment` field (DATA.md).

    python fast/segment.py clip.mp4 -o segments.pt --keyframes
    python fast/segment.py hands.h5 -o segments.eaf --format elan --fps 25

Video input additionally needs `signlangtk[wilor]` and MANO — see fast/README.md.
"""

import argparse
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="A video, a WiLoR .h5, or a directory of .h5 files.")
    parser.add_argument("-o", "--out", default="fast_segments.pt", help="Output file.")
    parser.add_argument("--format", choices=["pt", "json", "elan"], default="pt",
                        help="pt = {clip_id: (T,) int tensor}; json/elan = frame ranges.")
    parser.add_argument("--fps", type=float, default=25.0, help="Only used for json/elan timestamps.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda:0', ...")
    parser.add_argument("--checkpoint", default=None, help="Override segmenter checkpoint.")
    parser.add_argument("--features", default=None,
                        help="Where to write the extracted .h5 for video input (default: alongside --out).")
    parser.add_argument("--nlf", default=None,
                        help="NLF body .h5 (or directory) to assign hands to signers in multi-person video.")
    parser.add_argument("--no-begin", action="store_true",
                        help="Merge BEGIN into IN. SignSparK needs BEGIN, so leave off for LMDB use.")
    parser.add_argument("--keyframes", action="store_true",
                        help="Also report the keyframes SignSparK's loader would derive (mode 3).")
    parser.add_argument("--media", default=None, help="Video path to reference in ELAN output.")
    return parser.parse_args()


def extract_features(video, out_h5, device):
    """Run WiLoR hand extraction on a video. Needs signlangtk[wilor] + SLTK_MANO_DIR."""
    from sltk.extraction.wilor import WiLoRConfig, WiLoRExtractor

    print(f"### Extracting WiLoR hand poses from {video}")
    with WiLoRExtractor(WiLoRConfig(device=device)) as extractor:
        extractor.extract_from_video(video, out_h5)
    return out_h5


def report_keyframes(predictions):
    """Preview keyframes using the loader's own _segment_bounds.

    Not SLTK's extract_segments — the two disagree on a mid-run BEGIN.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from signspark.pose_datasets_lmdb import PoseDataset_Lmdb

    print("\n### Keyframes (keyframe_selection_mode=3)")
    for name, labels in predictions.items():
        seg = labels.tolist()
        bounds = PoseDataset_Lmdb._segment_bounds(seg)
        keyframes = sorted({f for s, e in bounds for f in (s, (s + e) // 2, e)})
        print(f"    {name}: {len(seg)} frames -> {len(bounds)} segments -> {len(keyframes)} keyframes")
        print(f"        {keyframes}")


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    source = Path(args.input)
    clip_name = None
    if source.suffix.lower() in VIDEO_SUFFIXES:
        clip_name = source.stem
        features = Path(args.features) if args.features else out.with_name(f"{clip_name}_wilor.h5")
        features.parent.mkdir(parents=True, exist_ok=True)
        source = extract_features(source, features, args.device)

    from sltk.segmentation import OutputFormat, get_segmenter, save_predictions
    from sltk.segmentation.transformer_segmenter import TransformerSegmenterConfig

    config = TransformerSegmenterConfig(
        device=args.device, checkpoint_path=args.checkpoint, include_begin=not args.no_begin
    )
    segmenter = get_segmenter("transformer", config=config)
    print(f"### Segmenter on {segmenter.config.device} | ckpt {segmenter.config.checkpoint_path}")

    predictions = segmenter.run_h5(source, nlf_dir=args.nlf)
    if not predictions:
        raise SystemExit(f"No .h5 files found at {source}")

    if clip_name is not None:
        # Key by the clip, not by the intermediate <clip>_wilor.h5 filename.
        predictions = {k.replace(source.stem, clip_name, 1): v for k, v in predictions.items()}

    save_predictions(predictions, out, OutputFormat(args.format), fps=args.fps, media_path=args.media)

    onsets = sum(int((labels == 2).sum()) for labels in predictions.values())
    print(f"\n### Wrote {len(predictions)} clip(s), {onsets} sign onsets -> {out}")

    if args.keyframes:
        report_keyframes(predictions)


if __name__ == "__main__":
    main()
