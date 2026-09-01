#!/usr/bin/env python3
"""Render a video with FAST's sign segments as a glow pulse and a timeline.

Frames inside a sign get a soft border glow; the strip along the bottom shows
every segment with a playhead scrubbing through it.

    python fast/visualize.py clip.mp4 segments.pt -o viz.mp4

The glow fades in and out rather than switching per frame, to stay clear of
photosensitivity thresholds.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ACCENT = (56, 176, 240)            # amber, BGR — reads well against grey studio backdrops
TRACK = (48, 44, 40)               # timeline background, BGR
TEXT = (235, 235, 235)

PRESETS = {"amber": (56, 176, 240), "blue": (248, 176, 56), "green": (120, 200, 90), "magenta": (200, 90, 220)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Source video.")
    parser.add_argument("segments", help="segments.pt written by fast/segment.py.")
    parser.add_argument("-o", "--out", default="viz.mp4", help="Output video.")
    parser.add_argument("--clip", default=None, help="Key inside segments.pt (default: match video stem, else the only entry).")
    parser.add_argument("--fade-in", type=float, default=2.0, help="Glow fade-in time, in frames.")
    parser.add_argument("--fade-out", type=float, default=4.0, help="Glow fade-out time, in frames.")
    parser.add_argument("--glow", type=float, default=0.06, help="Glow depth as a fraction of the short edge.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Peak glow strength, 0-1.")
    parser.add_argument("--color", choices=sorted(PRESETS), default="amber", help="Glow/timeline colour.")
    parser.add_argument("--width", type=int, default=None, help="Scale output to this width (default: at least 480).")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality, lower is better.")
    return parser.parse_args()


def load_labels(path, video_stem, clip):
    import torch

    preds = torch.load(path, map_location="cpu", weights_only=False)
    if clip is None:
        for candidate in (video_stem, f"{video_stem}_wilor"):
            if candidate in preds:
                clip = candidate
                break
        else:
            if len(preds) != 1:
                sys.exit(f"Pick one with --clip; {path} holds: {list(preds)}")
            clip = next(iter(preds))
    if clip not in preds:
        sys.exit(f"'{clip}' not in {path}; holds: {list(preds)}")
    return clip, np.asarray(preds[clip]).astype(int)


def segment_bounds(labels):
    """Segment ranges, using SignSparK's own convention (breaks on 0 only)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from signspark.pose_datasets_lmdb import PoseDataset_Lmdb

    return PoseDataset_Lmdb._segment_bounds(labels.tolist())


def smooth_envelope(labels, fade_in=2.0, fade_out=4.0):
    """Binary in-sign mask -> smooth 0..1 ramp, so the glow never flashes.

    Causal, so the glow starts on the sign onset rather than before it.
    """
    inside = (labels > 0).astype(np.float32)
    rise = 1.0 - np.exp(-1.0 / max(fade_in, 1e-3))
    fall = 1.0 - np.exp(-1.0 / max(fade_out, 1e-3))
    env = np.zeros_like(inside)
    level = 0.0
    for i, target in enumerate(inside):
        level += (target - level) * (rise if target > level else fall)
        env[i] = level
    return np.clip(env, 0.0, 1.0)


def flicker_rate(env, fps):
    """Glow direction reversals per second — the rate a viewer perceives as flashing."""
    sign = np.sign(np.diff(env))
    sign = sign[sign != 0]
    if len(sign) < 2 or fps <= 0:
        return 0.0
    return float((np.diff(sign) != 0).sum()) / (len(env) / fps)


def border_mask(h, w, depth=0.06):
    """Soft inward falloff from the frame edge, 1 at the border and 0 inside."""
    thickness = max(4.0, min(h, w) * depth)
    dy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    dx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    d = np.minimum(dy, dx).astype(np.float32)
    t = np.clip(1.0 - d / thickness, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t))[..., None]   # smoothstep, tighter than a square falloff


def draw_timeline(strip, labels, bounds, env, frame_idx, fps):
    """Timeline track: segment blocks, the glow envelope, and a playhead."""
    h, w = strip.shape[:2]
    strip[:] = (26, 24, 22)
    pad = max(8, w // 80)
    x0, x1 = pad, w - pad
    span = max(x1 - x0, 1)
    total = max(len(labels), 1)
    to_x = lambda f: int(x0 + span * f / total)

    track_top, track_bot = int(h * 0.30), int(h * 0.70)
    cv2.rectangle(strip, (x0, track_top), (x1, track_bot), TRACK, -1)

    for start, end in bounds:
        active = start <= frame_idx <= end
        colour = ACCENT if active else tuple(int(c * 0.45) for c in ACCENT)
        cv2.rectangle(strip, (to_x(start), track_top), (max(to_x(end + 1), to_x(start) + 2), track_bot), colour, -1)

    # the smoothed glow envelope, as a thin line over the blocks
    curve = np.array([[to_x(i), int(track_bot - (track_bot - track_top) * e)] for i, e in enumerate(env)])
    if len(curve) > 1:
        cv2.polylines(strip, [curve.astype(np.int32)], False, (225, 225, 225), 1, cv2.LINE_AA)

    # one-second ticks
    if fps > 0:
        for sec in range(1, int(total / fps) + 1):
            x = to_x(sec * fps)
            cv2.line(strip, (x, track_bot + 3), (x, track_bot + 7), (90, 86, 80), 1, cv2.LINE_AA)

    x = to_x(frame_idx)
    cv2.line(strip, (x, track_top - 6), (x, track_bot + 6), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(strip, (x, track_top - 7), 3, (255, 255, 255), -1, cv2.LINE_AA)

    scale = max(0.36, min(0.5, w / 1400))
    inside = labels[frame_idx] > 0 if frame_idx < len(labels) else False
    cv2.putText(strip, f"{frame_idx / fps:5.2f}s" if fps > 0 else f"f{frame_idx}",
                (x0, int(h * 0.94)), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT, 1, cv2.LINE_AA)
    label = f"SIGN  {len(bounds)} segments" if inside else f"{len(bounds)} segments"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(strip, label, (x1 - tw, int(h * 0.94)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                ACCENT if inside else (140, 136, 130), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    global ACCENT
    ACCENT = PRESETS[args.color]
    video = Path(args.video)
    clip, labels = load_labels(args.segments, video.stem, args.clip)
    bounds = segment_bounds(labels)
    env = smooth_envelope(labels, args.fade_in, args.fade_out)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = args.width or max(src_w, 480)
    scale = target_w / src_w
    w = target_w - (target_w % 2)
    h = int(round(src_h * scale))
    h -= h % 2
    strip_h = max(52, int(h * 0.17))
    strip_h -= strip_h % 2

    print(f"### {clip}: {len(labels)} frames, {len(bounds)} segments, {fps:g} fps -> {w}x{h + strip_h}")

    rate = flicker_rate(env, fps)
    if rate > 3.0:
        print(f"### WARNING: glow reverses at {rate:.1f} Hz, above the 3 Hz photosensitivity "
              f"guidance. Raise --fade-in/--fade-out to smooth it further.")

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h + strip_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf), str(args.out)],
        stdin=subprocess.PIPE,
    )

    mask = border_mask(h, w, args.glow)
    glow = np.zeros((h, w, 3), np.float32)
    glow[:] = ACCENT
    canvas = np.zeros((h + strip_h, w, 3), np.uint8)

    idx = 0
    while idx < len(labels):
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

        a = float(env[idx]) * args.alpha * mask
        canvas[:h] = np.clip(frame.astype(np.float32) * (1 - a) + glow * a, 0, 255).astype(np.uint8)
        draw_timeline(canvas[h:], labels, bounds, env, idx, fps)
        ffmpeg.stdin.write(canvas.tobytes())
        idx += 1

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"### Wrote {idx} frames -> {args.out}")


if __name__ == "__main__":
    main()
