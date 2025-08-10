#!/usr/bin/env python3
"""
Improved skeleton cleaning that actually performs interpolation
Strategy: Use different thresholds and joint-wise approach
"""

import os, argparse
from pathlib import Path
import numpy as np

def clean_pose_improved(arr: np.ndarray,
                       severe_frame_threshold: float = 0.8,  # Drop only very bad frames
                       joint_interpolation_threshold: int = 100,  # Max gap to interpolate
                       filename: str = "") -> tuple:
    """
    Improved cleaning approach:
    1. Only drop severely corrupted frames (>80% missing)
    2. For remaining frames, interpolate missing joints
    3. Use more sophisticated gap handling
    """
    original_frames = arr.shape[0]
    
    # Step 1: Find missing values
    miss = (~np.isfinite(arr)) | (arr == 0.)
    
    # Step 2: Only drop severely corrupted frames (>80% missing)
    frame_missing_ratio = miss.reshape(arr.shape[0], -1).mean(axis=1)
    frame_keep = frame_missing_ratio <= severe_frame_threshold
    frames_dropped = (~frame_keep).sum()
    
    print(f"    [FRAMES] Original: {original_frames}, Severely corrupted dropped: {frames_dropped}, Kept: {frame_keep.sum()}")
    
    if frames_dropped > 0:
        worst_frames = np.where(~frame_keep)[0][:5]
        worst_ratios = frame_missing_ratio[worst_frames]
        print(f"    [FRAMES] Worst frames dropped: {list(zip(worst_frames, [f'{r:.1%}' for r in worst_ratios]))}")
    
    arr = arr[frame_keep]
    miss = miss[frame_keep]
    F = arr.shape[0]
    
    if F == 0:
        print("    [ERROR] No frames left after dropping severely corrupted ones")
        return arr, {}
    
    # Step 3: More sophisticated joint-wise interpolation
    interpolation_stats = {
        'joints_never_detected': 0,
        'joints_with_long_gaps': 0,
        'joints_interpolated': 0,
        'joints_forward_filled': 0,
        'total_gaps_filled': 0,
        'frames_with_missing_after_cleaning': 0
    }
    
    for j in range(arr.shape[1]):  # 48 joints
        for c in range(3):  # x, y, z
            col = arr[:, j, c]
            bad = miss[:, j, c]
            
            if bad.all():
                # Joint never detected - use small random values
                arr[:, j, c] = 1e-3 * np.random.randn(F)
                interpolation_stats['joints_never_detected'] += 1
                continue
            
            if not bad.any():
                # Joint always detected, no cleaning needed
                continue
            
            # Find good indices and gap sizes
            good_idx = np.where(~bad)[0]
            
            if len(good_idx) == 0:
                continue
                
            # Handle different gap scenarios
            bad_indices = np.where(bad)[0]
            gaps_filled = 0
            
            # Method 1: Linear interpolation for gaps within good detections
            if len(good_idx) > 1:
                first_good = good_idx[0]
                last_good = good_idx[-1]
                
                # Interpolate gaps between first and last good detection
                for bad_idx in bad_indices:
                    if first_good < bad_idx < last_good:
                        # This is a gap that can be interpolated
                        gaps_filled += 1
                
                if gaps_filled > 0:
                    arr[bad, j, c] = np.interp(bad_indices, good_idx, col[good_idx])
                    interpolation_stats['joints_interpolated'] += 1
            
            # Method 2: Forward/backward fill for edges
            remaining_bad = miss[:, j, c] & (arr[:, j, c] == 0)  # Still missing after interpolation
            if remaining_bad.any():
                # Forward fill from first good value
                if len(good_idx) > 0:
                    fill_value = col[good_idx[0]]
                    for t in range(F):
                        if remaining_bad[t]:
                            arr[t, j, c] = fill_value
                    interpolation_stats['joints_forward_filled'] += 1
            
            interpolation_stats['total_gaps_filled'] += gaps_filled
    
    # Check remaining missing values
    final_missing = ((arr == 0) | (~np.isfinite(arr)))
    frames_still_missing = final_missing.reshape(F, -1).any(axis=1).sum()
    interpolation_stats['frames_with_missing_after_cleaning'] = frames_still_missing
    
    print(f"    [INTERP] Never detected: {interpolation_stats['joints_never_detected']}, "
          f"Interpolated: {interpolation_stats['joints_interpolated']}, "
          f"Forward filled: {interpolation_stats['joints_forward_filled']}, "
          f"Total gaps filled: {interpolation_stats['total_gaps_filled']}")
    
    print(f"    [RESULT] Frames with missing data after cleaning: {frames_still_missing}/{F}")
    
    return arr, {
        'original_frames': original_frames,
        'frames_dropped': frames_dropped,
        'final_frames': F,
        **interpolation_stats
    }

def process_file_improved(npy_path: Path,
                         dst_dir: Path,
                         drop_file_ratio: float,
                         severe_frame_threshold: float,
                         joint_interpolation_threshold: int) -> bool:
    
    print(f"\n[PROCESSING] {npy_path.name}")
    
    raw = np.load(npy_path).astype(np.float32)
    miss_ratio = ((raw == 0) | (~np.isfinite(raw))).mean()
    
    print(f"  [FILE] Shape: {raw.shape}, Missing ratio: {miss_ratio:.2%}")
    
    if miss_ratio >= drop_file_ratio:
        print(f"  [SKIP] Too much missing data ({miss_ratio:.2%} >= {drop_file_ratio:.1%})")
        return False
    
    cleaned, stats = clean_pose_improved(raw, severe_frame_threshold, joint_interpolation_threshold, npy_path.name)
    
    if cleaned.shape[0] < 10:
        print(f"  [SKIP] Too few frames after cleaning ({cleaned.shape[0]} < 10)")
        return False
    
    np.save(dst_dir / npy_path.name, cleaned.astype(np.float32))
    print(f"  [SAVED] Final shape: {cleaned.shape}")
    
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with raw *.npy pose files")
    ap.add_argument("--dst", required=True, help="Output folder for cleaned *.npy")
    ap.add_argument("--drop_file", type=float, default=0.90,
                    help="Skip file if >ratio coords missing (0-1)")
    ap.add_argument("--severe_frame", type=float, default=0.80,
                    help="Drop frame if >ratio coords missing (0-1)")
    ap.add_argument("--gap_interp", type=int, default=100,
                    help="Max consecutive missing frames to interpolate")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"IMPROVED CLEANING PARAMETERS:")
    print(f"  - Drop file if >{args.drop_file:.1%} missing")
    print(f"  - Drop frame if >{args.severe_frame:.1%} missing (severe corruption only)")
    print(f"  - Interpolate gaps up to {args.gap_interp} frames")
    print(f"  - Source: {src}")
    print(f"  - Destination: {dst}")

    kept, total = 0, 0
    total_frames_processed = 0
    total_gaps_filled = 0

    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".npy"): 
                continue
            total += 1
            if process_file_improved(Path(root)/f, dst,
                                   args.drop_file, args.severe_frame, args.gap_interp):
                kept += 1

    print(f"\n{'='*60}")
    print(f"IMPROVED CLEANING SUMMARY:")
    print(f"  Files processed: {kept}/{total}")
    print(f"  Success rate: {kept/total:.1%}")
    print(f"  Strategy: Keep more frames, interpolate missing joints")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 