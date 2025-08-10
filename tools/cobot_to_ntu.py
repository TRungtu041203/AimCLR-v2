#!/usr/bin/env python3
import argparse
import pickle
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert COBOT continuous skeleton dataset to NTU-style isolated samples '
            'for 3-stream AimCLR++ (no 64-frame pre-slicing; fixed-length clips).'
        )
    )

    parser.add_argument('--raw_root', type=Path, default=Path('pose_new_v2'))
    parser.add_argument('--ann_root', type=Path, default=Path('Annotation_v4'))
    parser.add_argument(
        '--ignored',
        type=Path,
        default=None,
        help='Optional path to a text file with one video_id per line to ignore',
    )
    parser.add_argument('--out_root', type=Path, default=Path('output'))
    parser.add_argument('--max_frame', type=int, default=300)
    parser.add_argument(
        '--train_subjects',
        type=int,
        nargs='+',
        default=[1, 2, 4, 5, 8],
        help='Subject IDs used for xsub train split; others go to val',
    )
    parser.add_argument(
        '--train_subjects_mode',
        type=str,
        choices=['manual', 'auto', 'file', 'all'],
        default='manual',
        help='manual: use --train_subjects; auto: split all subjects by --train_ratio; file: read from --train_subjects_file; all: all subjects go to train',
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='When --train_subjects_mode=auto, fraction of subjects assigned to train',
    )
    parser.add_argument(
        '--train_subjects_file',
        type=Path,
        default=None,
        help='When --train_subjects_mode=file, path to a file containing subject IDs (ints) separated by spaces/newlines/commas',
    )
    parser.add_argument(
        '--resample',
        type=str,
        choices=['pad', 'center-crop', 'uniform-sample'],
        default='uniform-sample',
        help='Policy for segments longer than max_frame',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for determinism')
    parser.add_argument(
        '--emit_clips',
        action='store_true',
        help='Optionally also write 64-frame center-crop clips to output/clips/',
    )
    parser.add_argument('--clip_len', type=int, default=64, help='Clip length when emitting clips')
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='If outputs exist but mismatch expected content, rebuild from scratch',
    )
    parser.add_argument(
        '--export_actions_root',
        type=Path,
        default=None,
        help='If set, also export isolated samples grouped by action name folders to this root',
    )
    parser.add_argument(
        '--export_actions_mode',
        type=str,
        choices=['raw', 'fitted'],
        default='raw',
        help='raw: save (L,48,3) segments; fitted: save (3,max_frame,48,1) segments',
    )

    args = parser.parse_args()
    return args


def read_ignored_list(path: Optional[Path]) -> set:
    if path is None or not path.exists():
        return set()
    ignored = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                ignored.add(token)
    return ignored


def list_pose_files(pose_root: Path) -> List[Path]:
    if not pose_root.exists():
        logging.error(f'Pose root not found: {pose_root}')
        return []
    return sorted([p for p in pose_root.iterdir() if p.suffix == '.npy'])


def discover_subject_ids_from_pose_files(pose_files: List[Path]) -> List[int]:
    subject_ids = []
    for p in pose_files:
        parsed = parse_pose_filename(p.stem)
        if parsed is None:
            continue
        sid, _sname, _vid = parsed
        subject_ids.append(sid)
    unique = sorted(set(subject_ids))
    return unique


def discover_subject_ids_from_annotations(ann_root: Path) -> List[int]:
    # Support both base dir and Annotation_v4 dir
    base = ann_root / 'Annotation_v4'
    scan_dir = base if base.exists() else ann_root
    subject_ids: List[int] = []
    if not scan_dir.exists():
        return []
    try:
        for entry in scan_dir.iterdir():
            if entry.is_dir():
                name = entry.name
                try:
                    sid_token = name.split('_', 1)[0]
                    digits = ''.join(ch for ch in sid_token if ch.isdigit())
                    sid = int(digits)
                    subject_ids.append(sid)
                except Exception:
                    continue
            elif entry.is_file() and entry.suffix.lower() == '.csv':
                # Fallback: parse CSV file prefix like '<sid>_<sname>_<vid>.csv'
                stem = entry.stem
                try:
                    sid_token = stem.split('_', 1)[0]
                    digits = ''.join(ch for ch in sid_token if ch.isdigit())
                    sid = int(digits)
                    subject_ids.append(sid)
                except Exception:
                    continue
    except Exception:
        return []
    return sorted(set(subject_ids))


def parse_pose_filename(stem: str) -> Optional[Tuple[int, str, str]]:
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    raw_subject = parts[0]
    digits = ''.join(ch for ch in raw_subject if ch.isdigit())
    if digits == '':
        return None
    try:
        subject_id = int(digits)
    except Exception:
        return None
    subject_name = parts[1]
    video_id = '_'.join(parts[2:])
    return subject_id, subject_name, video_id


def build_annotation_path(ann_root: Path, subject_id: int, subject_name: str, video_id: str) -> Path:
    # find CSV inside per-subject subfolder
    base = ann_root / 'Annotation_v4'
    scan_root = base if base.exists() else ann_root
    candidates = [f's{subject_id}_{subject_name}', f'{subject_id}_{subject_name}']
    for folder_name in candidates:
        folder = scan_root / folder_name
        if not (folder.exists() and folder.is_dir()):
            continue
        # Try matching filenames
        possibles = [
            folder / f'{folder_name}_{video_id}.csv',
        ]
        if folder_name.startswith('s'):
            possibles.append(folder / f'{folder_name[1:]}_{video_id}.csv')
        for p in possibles:
            if p.exists():
                return p
        # Fallback: any file ending with _<video_id>.csv
        try:
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() == '.csv' and f.name.endswith(f'_{video_id}.csv'):
                    return f
        except Exception:
            pass
    # Default expected path (for logging)
    return scan_root / f's{subject_id}_{subject_name}' / f's{subject_id}_{subject_name}_{video_id}.csv'


def validate_and_warn_overlaps_gaps(actions: pd.DataFrame) -> None:
    if actions.empty:
        return
    actions_sorted = actions.sort_values('start').reset_index(drop=True)
    for i in range(len(actions_sorted) - 1):
        cur_end = int(actions_sorted.loc[i, 'stop'])
        next_start = int(actions_sorted.loc[i + 1, 'start'])
        if next_start <= cur_end:
            logging.warning('Detected overlapping actions in CSV (start <= previous stop).')
        elif next_start > cur_end + 1:
            logging.warning('Detected gap between actions in CSV (start > previous stop + 1).')


def uniform_sample_indices(length: int, target: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((target,), dtype=np.int64)
    indices = np.linspace(0, max(0, length - 1), num=target)
    indices = np.clip(indices.round().astype(np.int64), 0, length - 1)
    return indices


def center_crop_indices(length: int, target: int) -> np.ndarray:
    if length <= target:
        # Will be padded elsewhere
        return np.arange(length, dtype=np.int64)
    start_index = (length - target) // 2
    return np.arange(start_index, start_index + target, dtype=np.int64)


def to_ntu_format(segment_xyz: np.ndarray) -> np.ndarray:
    # Input: (L, 48, 3) -> Output: (3, L, 48, 1)
    data = np.transpose(segment_xyz, (2, 0, 1)).astype(np.float32, copy=False)
    data = data[..., np.newaxis]
    return data


def fit_to_length(data: np.ndarray, max_frame: int, policy: str) -> Tuple[np.ndarray, bool, bool]:
    # data: (3, L, 48, 1)
    length = data.shape[1]
    did_pad = False
    did_resample = False

    if length == max_frame:
        return data, did_pad, did_resample

    if length < max_frame:
        out = np.empty((3, max_frame, 48, 1), dtype=np.float32)
        out[:, :length] = data
        last = data[:, length - 1 : length]
        if max_frame > length:
            out[:, length:] = last
        did_pad = True
        return out, did_pad, did_resample

    # length > max_frame
    if policy == 'uniform-sample':
        indices = uniform_sample_indices(length, max_frame)
        out = data[:, indices]
        did_resample = True
        return out, did_pad, did_resample
    elif policy == 'center-crop':
        indices = center_crop_indices(length, max_frame)
        if indices.shape[0] == max_frame:
            out = data[:, indices]
        else:
            # length < max_frame path handled above; here we still may need padding (edge case)
            out = np.empty((3, max_frame, 48, 1), dtype=np.float32)
            out[:, : indices.shape[0]] = data[:, indices]
            last = out[:, indices.shape[0] - 1 : indices.shape[0]]
            out[:, indices.shape[0] :] = last
        did_resample = True
        return out, did_pad, did_resample
    elif policy == 'pad':
        # Truncate tail to max_frame
        out = data[:, :max_frame]
        did_resample = True
        return out, did_pad, did_resample
    else:
        raise ValueError(f'Unknown resample policy: {policy}')


def write_labels(path: Path, names: List[str], labels: List[int]) -> None:
    with path.open('wb') as f:
        pickle.dump((names, labels), f)


def load_labels(path: Path) -> Optional[Tuple[List[str], List[int]]]:
    if not path.exists():
        return None
    try:
        with path.open('rb') as f:
            names, labels = pickle.load(f)
        return names, labels
    except Exception:
        return None


def open_memmap_writer(path: Path, shape: Tuple[int, int, int, int, int]) -> np.memmap:
    # Creates/overwrites using numpy.lib.format.open_memmap
    from numpy.lib.format import open_memmap as np_open_memmap

    path.parent.mkdir(parents=True, exist_ok=True)
    mem = np_open_memmap(filename=str(path), mode='w+', dtype='float32', shape=shape)
    return mem


def scan_dataset(
    raw_root: Path,
    ann_root: Path,
    ignored_video_ids: set,
    train_subjects: List[int],
) -> Tuple[List[Dict], Dict[str, int]]:
    # Support both when raw_root is the base directory containing pose_new_v2/ and when it is pose_new_v2 itself
    pose_root_candidate = raw_root / 'pose_new_v2'
    pose_root = pose_root_candidate if pose_root_candidate.exists() else raw_root
    logging.info(f'Scanning pose root: {pose_root}')
    pose_files = list_pose_files(pose_root)

    samples: List[Dict] = []
    action_histogram: Dict[str, int] = {}

    for pose_path in tqdm(pose_files, desc='Scanning pose files'):
        parsed = parse_pose_filename(pose_path.stem)
        if parsed is None:
            logging.warning(f'Skipping unrecognized filename: {pose_path.name}')
            continue
        subject_id, subject_name, video_id = parsed

        if video_id in ignored_video_ids:
            continue

        csv_path = build_annotation_path(ann_root, subject_id, subject_name, video_id)
        if not csv_path.exists():
            logging.warning(f'Missing annotation CSV for {pose_path.name} -> {csv_path}')
            continue

        try:
            # Lightweight shape read
            skel = np.load(str(pose_path), mmap_mode='r')
        except Exception as e:
            logging.error(f'Failed to load skeleton: {pose_path} ({e})')
            continue

        if skel.ndim != 3 or skel.shape[1] != 48 or skel.shape[2] != 3:
            logging.warning(f'Unexpected skeleton shape {skel.shape} in {pose_path.name}, skipping')
            continue

        total_frames = skel.shape[0]

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f'Failed to read CSV: {csv_path} ({e})')
            continue

        # Normalize column names
        df_columns = {c.strip().lower(): c for c in df.columns}
        required = ['id', 'start', 'stop']
        if not all(col in df_columns for col in required):
            logging.warning(f'CSV columns missing among {required} in {csv_path.name}, columns={list(df.columns)}')
            # Try best-effort mapping
            # If columns like 'label' exist, we keep but we need id,start,stop to continue
            possible = [c for c in df.columns if c.lower() in {'id', 'start', 'stop'}]
            if len(possible) < 3:
                continue

        id_col = df_columns.get('id', 'ID')
        start_col = df_columns.get('start', 'start')
        stop_col = df_columns.get('stop', 'stop')
        label_col = df_columns.get('label')

        # Warnings for overlap/gap
        try:
            tmp_df = pd.DataFrame({
                'start': df[start_col].astype(int),
                'stop': df[stop_col].astype(int),
            })
            validate_and_warn_overlaps_gaps(tmp_df)
        except Exception:
            pass

        for _, row in df.iterrows():
            try:
                action_id = int(row[id_col])
                start = int(row[start_col])
                end = int(row[stop_col])
            except Exception:
                continue

            if start > end:
                logging.warning(f'Invalid segment start>end in {csv_path.name}: {start}>{end}')
                continue
            if start < 0 or end >= total_frames:
                logging.warning(
                    f'Segment out of range in {csv_path.name}: [0,{total_frames-1}] vs [{start},{end}]'
                )
                continue

            length = end - start + 1
            split = 'train' if subject_id in train_subjects else 'val'

            sample_name = f'{subject_id}_{video_id}_A{action_id}_S{start}_E{end}'
            action_name = None
            if label_col is not None and label_col in df.columns:
                try:
                    action_name = str(row[label_col]).strip()
                except Exception:
                    action_name = None

            samples.append({
                'pose_path': pose_path,
                'subject_id': subject_id,
                'subject_name': subject_name,
                'video_id': video_id,
                'csv_path': csv_path,
                'action_id': action_id,
                'action_name': action_name,
                'start': start,
                'end': end,
                'length': length,
                'split': split,
                'sample_name': sample_name,
            })

            action_histogram[str(action_id)] = action_histogram.get(str(action_id), 0) + 1

    return samples, action_histogram


def write_memmaps(
    samples: List[Dict],
    out_root: Path,
    max_frame: int,
    resample_policy: str,
    emit_clips: bool,
    clip_len: int,
) -> Dict[str, Dict[str, int]]:
    xsub_dir = out_root / 'xsub'
    xsub_dir.mkdir(parents=True, exist_ok=True)

    # Prepare splits
    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples = [s for s in samples if s['split'] == 'val']

    # Resume/idempotency handling
    train_label_path = xsub_dir / 'train_label.pkl'
    val_label_path = xsub_dir / 'val_label.pkl'
    train_data_path = xsub_dir / 'train_data.npy'
    val_data_path = xsub_dir / 'val_data.npy'

    expected_train_names = [s['sample_name'] for s in train_samples]
    expected_train_labels = [int(s['action_id']) for s in train_samples]
    expected_val_names = [s['sample_name'] for s in val_samples]
    expected_val_labels = [int(s['action_id']) for s in val_samples]

    def memmap_shape_ok(path: Path, expected_n: int) -> bool:
        try:
            arr = np.load(str(path), mmap_mode='r')
            return (
                arr.shape == (expected_n, 3, max_frame, 48, 1)
                and arr.dtype == np.float32
            )
        except Exception:
            return False

    existing_train = load_labels(train_label_path)
    existing_val = load_labels(val_label_path)

    # Case 1: perfect match -> skip all writing
    if (
        existing_train is not None and existing_val is not None and
        existing_train[0] == expected_train_names and existing_train[1] == expected_train_labels and
        existing_val[0] == expected_val_names and existing_val[1] == expected_val_labels and
        train_data_path.exists() and val_data_path.exists() and
        memmap_shape_ok(train_data_path, len(train_samples)) and memmap_shape_ok(val_data_path, len(val_samples))
    ):
        logging.info('Outputs already exist with matching sample names and shapes; skipping write.')
        return {
            'train': {'count': len(train_samples)},
            'val': {'count': len(val_samples)},
            'padded': {'count': 0},
            'resampled': {'count': 0},
        }

    # Case 2: partial resume -> we will copy rows by name if possible
    old_train_map: Dict[str, int] = {}
    old_val_map: Dict[str, int] = {}
    old_train_mem = None
    old_val_mem = None

    if existing_train is not None and train_data_path.exists():
        try:
            old_train_names, _old_train_labels = existing_train
            old_train_mem = np.load(str(train_data_path), mmap_mode='r')
            if old_train_mem.shape[1:] == (3, max_frame, 48, 1):
                old_train_map = {name: i for i, name in enumerate(old_train_names)}
            else:
                old_train_mem = None
        except Exception:
            old_train_mem = None
            old_train_map = {}

    if existing_val is not None and val_data_path.exists():
        try:
            old_val_names, _old_val_labels = existing_val
            old_val_mem = np.load(str(val_data_path), mmap_mode='r')
            if old_val_mem.shape[1:] == (3, max_frame, 48, 1):
                old_val_map = {name: i for i, name in enumerate(old_val_names)}
            else:
                old_val_mem = None
        except Exception:
            old_val_mem = None
            old_val_map = {}

    # Create new memmaps
    train_mem = open_memmap_writer(train_data_path, (len(train_samples), 3, max_frame, 48, 1))
    val_mem = open_memmap_writer(val_data_path, (len(val_samples), 3, max_frame, 48, 1))

    # Optional clips
    if emit_clips:
        clips_dir = out_root / 'clips' / 'xsub'
        clips_dir.mkdir(parents=True, exist_ok=True)
        train_clips_path = clips_dir / 'train_data.npy'
        val_clips_path = clips_dir / 'val_data.npy'
        train_clips_mem = open_memmap_writer(train_clips_path, (len(train_samples), 3, clip_len, 48, 1))
        val_clips_mem = open_memmap_writer(val_clips_path, (len(val_samples), 3, clip_len, 48, 1))
    else:
        train_clips_mem = None
        val_clips_mem = None

    # For progress and stats
    padded_count = 0
    resampled_count = 0
    copied_count = 0

    # Group samples by pose file to minimize reloads
    by_pose: Dict[Path, List[Tuple[int, Dict]]] = {}
    for idx, s in enumerate(train_samples):
        by_pose.setdefault(s['pose_path'], []).append((('train', idx), s))
    for idx, s in enumerate(val_samples):
        by_pose.setdefault(s['pose_path'], []).append((('val', idx), s))

    for pose_path, entries in tqdm(by_pose.items(), desc='Writing samples'):
        # Determine if all entries can be copied from previous run
        all_copyable = True
        for (split_tag, index_in_split), s in entries:
            sname = s['sample_name']
            if split_tag == 'train':
                if old_train_mem is None or sname not in old_train_map:
                    all_copyable = False
                    break
            else:
                if old_val_mem is None or sname not in old_val_map:
                    all_copyable = False
                    break

        skel = None
        if not all_copyable:
            try:
                skel = np.load(str(pose_path))  # load fully for slicing speed
            except Exception as e:
                logging.error(f'Failed to load skeleton: {pose_path} ({e})')
                # Even if load fails, still try to copy any copyable ones
                skel = None

        for (split_tag, index_in_split), s in entries:
            sname = s['sample_name']
            # Copy path
            if split_tag == 'train' and old_train_mem is not None and sname in old_train_map:
                train_mem[index_in_split] = old_train_mem[old_train_map[sname]]
                copied_count += 1
                continue
            if split_tag == 'val' and old_val_mem is not None and sname in old_val_map:
                val_mem[index_in_split] = old_val_mem[old_val_map[sname]]
                copied_count += 1
                continue

            # Compute path (needs skel)
            if skel is None:
                # Cannot compute; skip
                logging.error(f'Cannot compute sample {sname} due to missing skeleton load.')
                continue

            start = s['start']
            end = s['end']
            segment = skel[start : end + 1]  # (L, 48, 3)
            ntu = to_ntu_format(segment)     # (3, L, 48, 1)
            fitted, did_pad, did_resample = fit_to_length(ntu, max_frame, resample_policy)
            if did_pad:
                padded_count += 1
            if did_resample:
                resampled_count += 1

            if split_tag == 'train':
                train_mem[index_in_split] = fitted
                if emit_clips and train_clips_mem is not None:
                    if fitted.shape[1] >= clip_len:
                        cc_idx = center_crop_indices(fitted.shape[1], clip_len)
                        train_clips_mem[index_in_split] = fitted[:, cc_idx]
                    else:
                        # Already padded to max_frame >= clip_len by design
                        cc_idx = center_crop_indices(fitted.shape[1], fitted.shape[1])
                        tmp = np.empty((3, clip_len, 48, 1), dtype=np.float32)
                        tmp[:, : cc_idx.shape[0]] = fitted[:, cc_idx]
                        last = tmp[:, cc_idx.shape[0] - 1 : cc_idx.shape[0]]
                        tmp[:, cc_idx.shape[0] :] = last
                        train_clips_mem[index_in_split] = tmp
            else:
                val_mem[index_in_split] = fitted
                if emit_clips and val_clips_mem is not None:
                    if fitted.shape[1] >= clip_len:
                        cc_idx = center_crop_indices(fitted.shape[1], clip_len)
                        val_clips_mem[index_in_split] = fitted[:, cc_idx]
                    else:
                        cc_idx = center_crop_indices(fitted.shape[1], fitted.shape[1])
                        tmp = np.empty((3, clip_len, 48, 1), dtype=np.float32)
                        tmp[:, : cc_idx.shape[0]] = fitted[:, cc_idx]
                        last = tmp[:, cc_idx.shape[0] - 1 : cc_idx.shape[0]]
                        tmp[:, cc_idx.shape[0] :] = last
                        val_clips_mem[index_in_split] = tmp

    # Flush memmaps
    del train_mem
    del val_mem
    if emit_clips and train_clips_mem is not None and val_clips_mem is not None:
        del train_clips_mem
        del val_clips_mem

    # Write labels at the end to reflect final expected sets
    write_labels(train_label_path, expected_train_names, expected_train_labels)
    write_labels(val_label_path, expected_val_names, expected_val_labels)

    return {
        'train': {'count': len(train_samples)},
        'val': {'count': len(val_samples)},
        'padded': {'count': padded_count},
        'resampled': {'count': resampled_count},
        'copied': {'count': copied_count},
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    np.random.seed(args.seed)

    # Roots
    raw_root = args.raw_root
    ann_root = args.ann_root
    out_root = args.out_root

    # Ignored list
    ignored_video_ids = read_ignored_list(args.ignored)
    if ignored_video_ids:
        logging.info(f'Loaded ignored list with {len(ignored_video_ids)} entries')

    # Determine subject split according to mode
    # First discover all pose files to infer subjects if needed
    pose_root_candidate = raw_root / 'pose_new_v2'
    pose_root = pose_root_candidate if pose_root_candidate.exists() else raw_root
    pose_files = list_pose_files(pose_root)

    if args.train_subjects_mode == 'manual':
        train_subjects = args.train_subjects
    elif args.train_subjects_mode == 'all':
        train_subjects = discover_subject_ids_from_pose_files(pose_files)
        if not train_subjects:
            # Try discover from annotations
            train_subjects = discover_subject_ids_from_annotations(ann_root)
    elif args.train_subjects_mode == 'file':
        if not args.train_subjects_file or not args.train_subjects_file.exists():
            logging.error('--train_subjects_mode=file but --train_subjects_file is missing')
            return
        file_txt = args.train_subjects_file.read_text(encoding='utf-8')
        tokens = [t.strip() for t in file_txt.replace(',', ' ').split() if t.strip()]
        try:
            train_subjects = sorted(set(int(t) for t in tokens))
        except Exception:
            logging.error('Failed to parse integers from --train_subjects_file')
            return
    elif args.train_subjects_mode == 'auto':
        all_subjects = discover_subject_ids_from_pose_files(pose_files)
        if not all_subjects:
            all_subjects = discover_subject_ids_from_annotations(ann_root)
        if not all_subjects:
            logging.error('No subjects discovered for auto split')
            return
        rng = np.random.RandomState(args.seed)
        shuffled = all_subjects.copy()
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * max(0.0, min(1.0, args.train_ratio)))
        train_subjects = sorted(shuffled[:split_idx])
        if not train_subjects:
            # ensure at least one subject in train if possible
            train_subjects = [shuffled[0]]
    else:
        logging.error(f'Unknown --train_subjects_mode: {args.train_subjects_mode}')
        return

    logging.info(f'Train subjects: {train_subjects}')

    # Scan and collect
    samples, action_hist = scan_dataset(
        raw_root=raw_root,
        ann_root=ann_root,
        ignored_video_ids=ignored_video_ids,
        train_subjects=train_subjects,
    )

    if not samples:
        logging.error('No valid samples found. Nothing to do.')
        return

    # Summary before writing
    num_train = sum(1 for s in samples if s['split'] == 'train')
    num_val = len(samples) - num_train
    avg_len = np.mean([s['length'] for s in samples]) if samples else 0.0

    logging.info(f'Total samples: {len(samples)} (train={num_train}, val={num_val})')
    logging.info(f'Average segment length (frames): {avg_len:.2f}')
    logging.info(f'Unique actions: {len(action_hist)}')

    # Optionally force rebuild by deleting outputs
    xsub_dir = out_root / 'xsub'
    if args.force_rebuild and xsub_dir.exists():
        for p in [xsub_dir / 'train_data.npy', xsub_dir / 'val_data.npy', xsub_dir / 'train_label.pkl', xsub_dir / 'val_label.pkl']:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    stats = write_memmaps(
        samples=samples,
        out_root=out_root,
        max_frame=args.max_frame,
        resample_policy=args.resample,
        emit_clips=args.emit_clips,
        clip_len=args.clip_len,
    )

    # Final report
    counts_per_action = ', '.join([f'{aid}:{cnt}' for aid, cnt in sorted(action_hist.items(), key=lambda x: int(x[0]))])
    logging.info(f'Counts per split: train={stats["train"]["count"]}, val={stats["val"]["count"]}')
    logging.info(f'Per-action counts: {counts_per_action}')
    logging.info(f'Padded segments: {stats["padded"]["count"]}, Resampled segments: {stats["resampled"]["count"]}')

    # Optional export grouped by action name
    if args.export_actions_root is not None:
        export_root: Path = args.export_actions_root
        export_root.mkdir(parents=True, exist_ok=True)
        # Group by pose to minimize reloads
        by_pose: Dict[Path, List[Dict]] = {}
        for s in samples:
            by_pose.setdefault(s['pose_path'], []).append(s)

        for pose_path, entries in tqdm(by_pose.items(), desc='Exporting per-action segments'):
            try:
                skel = np.load(str(pose_path))
            except Exception as e:
                logging.error(f'Failed to load skeleton: {pose_path} ({e})')
                continue

            for s in entries:
                start = s['start']
                end = s['end']
                act_name = s.get('action_name') or f'A{s["action_id"]}'
                safe_name = str(act_name).strip().replace('/', '-')
                dst_dir = export_root / safe_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                filename = f'{s["sample_name"]}.npy'
                dst_path = dst_dir / filename
                if dst_path.exists():
                    continue
                seg = skel[start : end + 1]
                if args.export_actions_mode == 'raw':
                    np.save(str(dst_path), seg.astype(np.float32))
                else:
                    ntu = to_ntu_format(seg)
                    fitted, _did_pad, _did_resample = fit_to_length(ntu, args.max_frame, args.resample)
                    np.save(str(dst_path), fitted.astype(np.float32))


if __name__ == '__main__':
    main()


