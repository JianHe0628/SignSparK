#!/usr/bin/env python3
"""Render sample.py predictions as side-by-side GT/pred SMPL-X mesh videos.

Reads the per-stream .npy (body/hand/optional face) and renders one comparison
video per clip. Uses the official `smplx` package; download the SMPL-X models
yourself (see VISUALIZATION.md). For quantitative metrics, use the official
evaluation at https://github.com/2000ZRL/SOKE.

    python tools/visualize.py --body body.npy --hand hand.npy [--face face.npy] \
        --model-folder /path/to/smplx_models --out viz
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless offscreen rendering

NUM_HAND_JOINTS = 15        # MANO joints per hand (15 × 6D = 90)
NUM_LEG_SPINE_JOINTS = 11   # leading body joints the model does not predict
R_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))


# ── Rotation helpers ─────────────────────────────────────────────────────────
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation (Zhou et al.) -> (..., 3, 3)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _mats_to_rotvec(mats_np: np.ndarray) -> np.ndarray:
    """(M,3,3) -> (M,3) axis-angle, mapping degenerate matrices (zero-padded /
    masked joints) to identity so scipy doesn't reject them."""
    finite = np.isfinite(mats_np).all(axis=(-2, -1))
    det = np.zeros(mats_np.shape[0])
    det[finite] = np.linalg.det(mats_np[finite])
    bad = (~finite) | (det <= 1e-6)
    mats_np = mats_np.copy()
    mats_np[bad] = np.eye(3)
    rotvec = Rotation.from_matrix(mats_np).as_rotvec()
    rotvec[bad] = 0.0
    return rotvec


def rot6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) 6D rotations -> (..., 3) axis-angle (rotation vectors)."""
    mats = rot6d_to_matrix(d6.reshape(-1, 6)).cpu().numpy()
    rotvec = _mats_to_rotvec(mats)
    return torch.from_numpy(rotvec).float().reshape(*d6.shape[:-1], 3)


def flip_left_hand_6d(left_6d: torch.Tensor) -> torch.Tensor:
    """Conjugate left-hand rot6D by diag(1,-1,-1) (SMPLX-left -> right-hand frame)."""
    mats = rot6d_to_matrix(left_6d.reshape(-1, 6)).reshape(-1, NUM_HAND_JOINTS, 3, 3)
    flipped = (R_FLIP @ mats @ R_FLIP).reshape(-1, 3, 3).cpu().numpy()
    rotvec = _mats_to_rotvec(flipped)
    return torch.from_numpy(rotvec).float().reshape(-1, NUM_HAND_JOINTS, 3)


# ── Feature loading / assembly ───────────────────────────────────────────────
def load_stream(npy_path, repetition):
    """Load one stream .npy and return (gt, pred) as (B, T, C) tensors."""
    feats = np.load(npy_path, allow_pickle=True).item()
    gt = feats["gt_poses"][repetition]
    pred = feats["pred_poses"][repetition]
    gt = torch.from_numpy(gt.reshape(gt.shape[0], -1, gt.shape[-1])).float().permute(0, 2, 1)
    pred = torch.from_numpy(pred.reshape(pred.shape[0], -1, pred.shape[-1])).float().permute(0, 2, 1)
    return feats, gt, pred


def build_smplx_input(body_60d, lhand_90d, rhand_90d, face=None, num_expression=50,
                      flip_left_hand=True, use_expression=True, device="cpu"):
    """rot6D stream features -> SMPL-X forward kwargs. body_60d (N,60),
    lhand/rhand_90d (N,90), face (N, jaw6 + num_expression) or None."""
    n = body_60d.shape[0]

    # Body: prepend the 11 unpredicted leading joints as zeros, then 6D -> axis-angle.
    body_full_6d = torch.cat([torch.zeros(n, NUM_LEG_SPINE_JOINTS * 6), body_60d], dim=-1)
    body_pose = rot6d_to_axis_angle(body_full_6d.reshape(n, -1, 6)).reshape(n, -1)
    body_pose[:, : NUM_LEG_SPINE_JOINTS * 3] = 0.0  # keep legs/spine neutral

    if flip_left_hand:
        left_pose = flip_left_hand_6d(lhand_90d).reshape(n, -1)
    else:
        left_pose = rot6d_to_axis_angle(lhand_90d.reshape(n, NUM_HAND_JOINTS, 6)).reshape(n, -1)
    right_pose = rot6d_to_axis_angle(rhand_90d.reshape(n, NUM_HAND_JOINTS, 6)).reshape(n, -1)

    if face is not None:
        pose_dim = face.shape[1] - num_expression
        jaw_pose = rot6d_to_axis_angle(face[:, :pose_dim].reshape(n, -1, 6)).reshape(n, -1)
        expression = face[:, pose_dim:] if use_expression else torch.zeros(n, num_expression)
    else:
        jaw_pose = torch.zeros(n, 3)
        expression = torch.zeros(n, num_expression)

    zeros3 = torch.zeros(n, 3)
    out = {
        "betas": torch.zeros(n, 10),
        "global_orient": zeros3.clone(),
        "transl": zeros3.clone(),
        "body_pose": body_pose,
        "left_hand_pose": left_pose,
        "right_hand_pose": right_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": zeros3.clone(),
        "reye_pose": zeros3.clone(),
        "expression": expression,
    }
    return {k: v.to(device) for k, v in out.items()}


# ── Rendering ────────────────────────────────────────────────────────────────
def render_mesh(verts, faces, image_size, label):
    import pyrender, trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = verts - verts.mean(0)
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts.cpu().numpy(), faces, process=False), smooth=True)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh)
    cam_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0.1], [0, 0, 1, 1.0], [0, 0, 0, 1]], dtype=float)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
    for dx, intensity in [(0.5, 3.0), (-0.5, 1.8), (0.0, 2.4)]:
        lp = np.eye(4); lp[0, 3] = dx; lp[1, 3] = 0.4
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=lp)
    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    color, _ = renderer.render(scene)
    renderer.delete()

    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    plt.imshow(color); plt.text(10, 24, label, color="red", fontsize=15, fontweight="bold")
    plt.axis("off"); plt.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return img


def write_comparison_video(pred_verts, gt_verts, keyframes, faces, out_path, gap, image_size, fps):
    import cv2
    border = 10
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for idx in tqdm(range(0, len(gt_verts), gap), desc=out_path.stem, leave=False):
        gt_img = render_mesh(gt_verts[idx], faces, image_size, "Ground Truth")
        pred_img = render_mesh(pred_verts[idx], faces, image_size, "Prediction")
        frame = cv2.cvtColor(np.concatenate([gt_img, pred_img], axis=1), cv2.COLOR_RGB2BGR)
        on_kf = keyframes[idx] > 0
        frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT,
                                   value=(0, 0, 224) if on_kf else (0, 0, 0))
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer is not None:
        writer.release()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", required=True, help="body stream .npy from sample.py")
    p.add_argument("--hand", required=True, help="hand stream .npy from sample.py (L/R interleaved)")
    p.add_argument("--face", default=None, help="optional face stream .npy")
    p.add_argument("--model-folder", required=True, help="folder with SMPL-X model files")
    p.add_argument("--out", required=True, help="output dir for the comparison videos")
    p.add_argument("--repetition", type=int, default=0, help="which sampling candidate to use")
    p.add_argument("--gap", type=int, default=1, help="render every Nth frame")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--num-expression", type=int, default=50)
    p.add_argument("--num-betas", type=int, default=10)
    p.add_argument("--gender", default="neutral")
    p.add_argument("--no-flip-left-hand", action="store_true", help="skip the SMPLX-left -> right-hand flip")
    p.add_argument("--no-expression", action="store_true", help="render jaw-only (zero face expression)")
    p.add_argument("--max-clips", type=int, default=-1, help="cap number of clips processed")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import smplx
    model = smplx.create(
        args.model_folder, model_type="smplx", gender=args.gender, use_pca=False,
        flat_hand_mean=True, num_betas=args.num_betas, num_expression_coeffs=args.num_expression,
    ).to(device).eval()
    faces = model.faces

    body_feats, gt_body, pred_body = load_stream(args.body, args.repetition)
    _, gt_hand, pred_hand = load_stream(args.hand, args.repetition)
    if args.face:
        _, gt_face, pred_face = load_stream(args.face, args.repetition)

    # Hands are stored interleaved L0,R0,L1,R1,...  -> de-interleave.
    gt_lh, gt_rh = gt_hand[::2], gt_hand[1::2]
    pred_lh, pred_rh = pred_hand[::2], pred_hand[1::2]
    b = min(len(gt_body), len(gt_lh))
    if args.max_clips > 0:
        b = min(b, args.max_clips)

    lengths = body_feats["lengths"][args.repetition, :b]
    names = [str(x) for x in body_feats["video_names"][args.repetition, :b]]
    kfs = torch.from_numpy(np.asarray(body_feats["keyframes"]))[args.repetition, :b]
    kfs = kfs.float().reshape(kfs.shape[0], -1, kfs.shape[-1]).permute(0, 2, 1).sum(-1)  # (b, T)

    out_dir = Path(args.out)
    for i in tqdm(range(b), desc="clips"):
        L = int(lengths[i])
        smplx_in_pred = build_smplx_input(
            pred_body[i, :L], pred_lh[i, :L][:, :90], pred_rh[i, :L][:, :90],
            face=pred_face[i, :L] if args.face else None,
            num_expression=args.num_expression, flip_left_hand=not args.no_flip_left_hand,
            use_expression=not args.no_expression, device=device)
        smplx_in_gt = build_smplx_input(
            gt_body[i, :L], gt_lh[i, :L][:, :90], gt_rh[i, :L][:, :90],
            face=gt_face[i, :L] if args.face else None,
            num_expression=args.num_expression, flip_left_hand=not args.no_flip_left_hand,
            use_expression=not args.no_expression, device=device)

        with torch.no_grad():
            out_pred, out_gt = model(**smplx_in_pred), model(**smplx_in_gt)

        write_comparison_video(
            out_pred.vertices.cpu(), out_gt.vertices.cpu(), kfs[i], faces,
            out_dir / f"viz_{names[i]}.mp4", args.gap, args.image_size, args.fps)


if __name__ == "__main__":
    main()
