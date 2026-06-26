# import blobfile as bf
from unittest import case

from signspark.utils import logger
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import sys
import torch
from pathlib import Path
import pickle, lmdb, io
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from signspark.helper import skip_list
from scipy.signal import savgol_filter

numpy_major_version = int(np.__version__.split('.')[0])
if numpy_major_version < 2:
    print(f"Warning: Detected numpy version {np.__version__}. For optimal performance, please upgrade to numpy 2.x or later.")
    sys.modules['numpy._core'] = sys.modules['numpy.core']
    sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
    sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']


# ── Left-hand convention flip ─────────────────────────────────────────────────
# LMDBs store the left hand in true SMPLX-left convention; the model operates in
# right-hand (WiLoR) convention, so the left hand is converted by conjugating
# each joint rotation R with the MANO-frame flip diag(1, -1, -1) (a 180 deg
# rotation about the joint-local x-axis). The element-wise mask below is that
# conjugation: (R_flip @ R @ R_flip)[i,j] = sign_i * sign_j * R[i,j],
# sign = [1, -1, -1]. Uses the Zhou et al. row-based rot6D convention.
_LEFT_HAND_FLIP_MASK = np.array([
    [ 1.0, -1.0, -1.0],
    [-1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0],
], dtype=np.float32)


def _rot6d_to_matrix_np(d6):
    """6D rotation (Zhou et al., Gram-Schmidt) -> (..., 3, 3). Rows are b1,b2,b3."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _matrix_to_rot6d_np(matrix):
    """(..., 3, 3) -> 6D by dropping the last row (rows 0,1)."""
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(*batch_dim, 6)


def flip_left_hand_features(left_feats):
    """Flip left-hand rot6D features (T, >=90) from SMPLX-left into right-hand
    convention. Operates on the first 90 dims (15 joints x 6D); trailing dims
    pass through. The 6D -> matrix -> 6D round-trip also re-orthonormalizes,
    identical to how the training data was prepared."""
    left = left_feats.astype(np.float32, copy=False)
    T_len = left.shape[0]
    pose_6d = left[:, :90].reshape(T_len * 15, 6)
    mats = _rot6d_to_matrix_np(pose_6d) * _LEFT_HAND_FLIP_MASK
    flipped = _matrix_to_rot6d_np(mats).reshape(T_len, 90)
    if left.shape[-1] > 90:
        return np.concatenate([flipped, left[:, 90:]], axis=-1)
    return flipped

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def custom_collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b[0] for b in notnone_batches]
    lengths = [b[1] for b in notnone_batches]

    databatchTensor = torch.stack(databatch, dim=0).permute(0,2,1) # (B,T,J) → (B,J,T)
    lenbatchTensor = torch.tensor(lengths, dtype=torch.long) # (B,)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.size(-1)).unsqueeze(1) # (B,1,T)

    cond = {'mask': maskbatchTensor, 'lengths': lenbatchTensor}

    textbatch = [b[2] for b in notnone_batches]
    keyframes_list = [b[3] for b in notnone_batches]
    video_names = [b[4] for b in notnone_batches]

    cond.update({'text': textbatch, 'keyframes': keyframes_list, 'video_names': video_names})

    return databatchTensor, cond

def build_concat_dataset(
    lmdb_paths: list[str],
    seq_len,
    data_args=None,
    split="train",
    is_debug=False,
    segments=None,
    concat_hands=False,
) -> ConcatDataset:
    if data_args.specify_lang:
        print("\033[91mPlease Note Language Specifying Tokens are on!\033[0m")
    logger.log(f"Keyframe selection mode: {data_args.keyframe_selection_mode} | Language Token Specifier: {data_args.specify_lang}")
    datasets = [
        PoseDataset_Lmdb(data_args, split=split, seq_len=seq_len, is_debug=is_debug, data=path, segments=segments, concat_hands=concat_hands)
        for path in lmdb_paths
    ]
    return ConcatDataset(datasets)

def load_lmdb_dataset(
    batch_size,
    seq_len,
    deterministic=False,
    data_args=None,
    split="train",
    is_debug=False,
    segments=None,
    num_workers=4,
    drop_last=False,
    concat_hands=False,
):
    dataset_names = {"train": data_args.train_data, "dev": data_args.dev_data, "test": data_args.test_data}[split]
    lmdb_paths = [str(l) for l in (Path(data_args.base_path) / split).rglob("*.lmdb")]
    lmdb_paths = [p for p in lmdb_paths if any(d in p for d in dataset_names)]

    if len(lmdb_paths) > 1:
        logger.log(f"Multiple LMDB paths ({dataset_names}) found for split '{split}': Building concatenated dataset.")
        dataset = build_concat_dataset(lmdb_paths, seq_len, data_args=data_args, split=split, is_debug=is_debug, segments=segments, concat_hands=concat_hands)
    elif len(lmdb_paths) == 1:
        logger.log(f"Single LMDB path found for split '{split}': {lmdb_paths[0]}")
        dataset = PoseDataset_Lmdb(data_args, split=split, seq_len=seq_len, is_debug=is_debug, data=lmdb_paths[0], segments=segments, concat_hands=concat_hands)
    else:
        raise ValueError(f"No valid LMDB paths found for split '{split}'")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,  # Disable multiprocessing to avoid file handle issues
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.log(f"dataloader length {len(data_loader)}")

    return data_loader


def infinite_loader(data_loader):
    while True:
        yield from data_loader

class PoseDataset_Lmdb(Dataset):
    def __init__(self, data_args, split="train", seq_len=224, is_debug=False, data=None, segments=None, concat_hands=False):
        super().__init__()

        assert split in ["train", "dev", "test"], "invalid split for dataset"

        self.data_args = data_args
        self.dataset_feat = data_args.dataset_feat
        self.specify_lang = getattr(data_args, 'specify_lang', True)
        self.max_feat_length = seq_len
        self.concat_hands = concat_hands
        self.flip_left_hand = getattr(data_args, 'flip_left_hand', False)
        if self.flip_left_hand:
            logger.log("### flip_left_hand=True: converting left hand from SMPLX-left to right-hand convention")

        self.keyframe_selection = int(data_args.keyframe_selection_mode)
        if Path(data_args.custom_keyframe_file).exists():
            self.custom_keyframe_file = torch.load(data_args.custom_keyframe_file)
            self.custom_keyframe_file = self.custom_keyframe_file[split]
            self.keyframe_selection = 999
            logger.log(f"Loaded custom keyframe file from {data_args.custom_keyframe_file}")
        else:
            self.custom_keyframe_file = None

        # Load Metadata
        self.lmdb_path = str(data)  
        with lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1024) as tmp_env:
            with tmp_env.begin(write=False) as txn:
                meta = txn.get(b'__meta__')
        
        meta = pickle.loads(meta)
        self.name_list = meta.get("clip_ids", [])
        self.length = meta.get("num_clips", len(self.name_list))
        self._env = None
        logger.log("#" * 15, f"\nLoading [Dataset: {data_args.dataset_feat} @ {self.length} items | Seq_Len: {seq_len}] from {data}")
    
    def _open_env(self) -> lmdb.Environment:
        return lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,           # False → no lockfile, safe for multi-worker
            readahead=False,          # Better for random access patterns
            meminit=False,            # Faster opens
            max_readers=1024,         # Enough headroom for many workers
            map_size=1 << 40,         # 1 TiB virtual; actual disk usage is sparse
        )
 
    @property
    def env(self) -> lmdb.Environment:
        """
        Lazy env open: each DataLoader worker gets its own handle.
        This avoids the 'lmdb opened before fork' deadlock.
        """
        if self._env is None:
            self._env = self._open_env()
        return self._env
    
    def __del__(self):
        # Best-effort cleanup when the worker process exits
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.name_list[idx]
        with self.env.begin(write=False) as txn:
            raw = txn.get(name.encode('utf-8'))

        if raw is None:
            raise KeyError(f"Key {name} not found in LMDB dataset.")

        data = self.deserialize_minimal_npz(raw)
        if self.flip_left_hand and 'left_features' in data:
            data['left_features'] = flip_left_hand_features(data['left_features'])
        data_len = len(data['segment'])
        text = f"<{data['language']}> {data['translation']}" if self.specify_lang else data['translation']

        # Generate the sparse conditioning keyframes from the segment labels.
        keyframes = self.generate_keyframes(
            data['segment'], selection_mode=self.keyframe_selection,
            custom_file=self.custom_keyframe_file, key=name)

        # Load Features     
        match self.dataset_feat:
            case "hand":
                if self.concat_hands:
                    features = np.concatenate([data['left_features'][:, :90], data['right_features'][:, :90]], axis=-1) # (T, 180)
                else:
                    # Train on both hands: left is loaded in right-hand convention
                    # (flip_left_hand), so sample either per clip (mirror aug).
                    side = 'left_features' if np.random.rand() < 0.5 else 'right_features'
                    features = data[side][:, :90]
                    name += f"_{side.split('_')[0]}"
            case "body":
                features = data['body_features'][:, 11*6:] # Last 10 joints (no legs) T x 60
            case "face":
                features = data['face_features']
            case "full":
                features = np.concatenate([data['left_features'][:,:90], data['right_features'][:,:90], data['body_features'][:, 11*6:], data['face_features']], axis=-1) # (T, 180 + 60 + 56)

        if data_len < self.max_feat_length:
            features = np.concatenate([
                features,
                np.tile(features[-1:], (self.max_feat_length - data_len, 1))
            ], axis=0)
        elif data_len > self.max_feat_length: 
            data_len = self.max_feat_length
            keyframes = [kf for kf in keyframes if kf < self.max_feat_length]
            features = features[:self.max_feat_length]

        features = torch.tensor(features, dtype=torch.float32)

        return features, data_len, text, keyframes, name
        
    @staticmethod
    def _segment_bounds(segment):
        """Return [(start, end), ...] frame ranges for each sign segment.

        ``segment`` is a per-frame label: 0 = non-sign, 1/2 = within a sign
        (2 marks a segment start).
        """
        padded = [0] + list(segment) + [0]
        bounds, start = [], None
        for i in range(1, len(padded)):
            cur, prev = padded[i], padded[i - 1]
            if cur in (1, 2) and prev == 0:
                start = i - 1
            elif cur == 0 and prev in (1, 2) and start is not None:
                bounds.append((start, i - 2))
                start = None
        return bounds

    def generate_keyframes(self, segment_annotations, selection_mode=3, custom_file=None, key=None):
        """Pick the sparse conditioning keyframes from the segment labels.

        Mode 3 (default) takes the first, middle and last frame of each sign
        segment. Mode 999 uses a user-supplied keyframe file. To add your own
        scheme, add a branch below returning a list of frame indices.
        """
        if isinstance(segment_annotations, (torch.Tensor, np.ndarray)):
            segment_annotations = segment_annotations.tolist()

        if selection_mode == 999:  # user-supplied keyframes
            if custom_file is None or key is None:
                raise ValueError("Custom keyframe selection requires custom_file and key.")
            return sorted(set(custom_file[key]))

        if selection_mode == 3:  # first / middle / last frame of each segment
            keyframes = []
            for start, end in self._segment_bounds(segment_annotations):
                keyframes.extend([start, (start + end) // 2, end])
            return sorted(set(keyframes))

        raise ValueError(
            f"Unknown keyframe_selection_mode: {selection_mode} "
            "(add your own branch in generate_keyframes)."
        )

    def deserialize_minimal_npz(self, data: bytes):
        with io.BytesIO(data) as buffer:
            npz_file = np.load(buffer, allow_pickle=True)
            return {
                "language": npz_file["language"][0],
                "translation": npz_file["translation"][0],
                "gloss": npz_file["gloss"][0],
                "segment": npz_file["segment"],
                "left_features": npz_file["left_features"],
                "right_features": npz_file["right_features"],       
                "body_features": npz_file["body_features"],
                "face_features": npz_file["face_features"],
                # "keyframes": npz_file["keyframes"].tolist(),
            }
