#!/usr/bin/env python3
"""
Advanced skeleton visualization with matplotlib
Includes 2D plotting, 3D plotting, animations, and image saving
Handles numpy version conflicts gracefully
"""

import numpy as np
import os
import argparse
from pathlib import Path

# Try to import matplotlib with version checking
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend first
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib loaded successfully")
except ImportError as e:
    print(f"⚠️  Matplotlib not available: {e}")
    print("📝 Will use text-based visualization only")

# Define skeleton connections for 2D/3D plotting
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20)
]

ARM_CONNECTIONS = [
    # Arms: [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    (0, 2), (2, 4),  # Left arm: shoulder -> elbow -> wrist
    (1, 3), (3, 5),  # Right arm: shoulder -> elbow -> wrist
    (0, 1)           # Shoulders connection
]

def print_skeleton_stats(data, name="Data"):
    """Print comprehensive skeleton statistics (always available)"""
    
    print(f"\n{'='*60}")
    print(f"{name.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    # Basic info
    print(f"📊 Basic Info:")
    print(f"   Shape: {data.shape}")
    print(f"   Frames: {data.shape[0]}")
    print(f"   Keypoints per frame: {data.shape[1]}")
    print(f"   Coordinates per keypoint: {data.shape[2]}")
    
    # Missing data analysis
    missing_mask = (data == 0).all(axis=2)  # (frames, keypoints)
    total_missing = missing_mask.sum()
    total_possible = data.shape[0] * data.shape[1]
    
    print(f"\n🔍 Missing Data Overview:")
    print(f"   Total missing keypoints: {total_missing:,}")
    print(f"   Total possible keypoints: {total_possible:,}")
    print(f"   Missing percentage: {total_missing/total_possible:.1%}")
    
    # Frame-level analysis
    frame_missing_counts = missing_mask.sum(axis=1)
    frame_missing_ratios = frame_missing_counts / data.shape[1]
    
    print(f"\n📈 Frame-level Analysis:")
    print(f"   Perfect frames (0% missing): {(frame_missing_ratios == 0).sum():,}")
    print(f"   Good frames (1-25% missing): {((frame_missing_ratios > 0) & (frame_missing_ratios <= 0.25)).sum():,}")
    print(f"   Fair frames (26-50% missing): {((frame_missing_ratios > 0.25) & (frame_missing_ratios <= 0.5)).sum():,}")
    print(f"   Poor frames (51-75% missing): {((frame_missing_ratios > 0.5) & (frame_missing_ratios <= 0.75)).sum():,}")
    print(f"   Bad frames (76-100% missing): {(frame_missing_ratios > 0.75).sum():,}")

def plot_skeleton_frame_2d(ax, pose_data, frame_idx, title="", show_missing=True):
    """
    Plot a single frame of skeleton data with 2D visualization
    Only works if matplotlib is available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for 2D plotting")
        return False
    
    ax.clear()
    
    if frame_idx >= pose_data.shape[0]:
        ax.text(0.5, 0.5, f"Frame {frame_idx} not available", 
                ha='center', va='center', transform=ax.transAxes)
        return True
    
    frame = pose_data[frame_idx]  # (48, 3)
    
    # Extract different body parts
    right_hand = frame[0:21]      # (21, 3)
    left_hand = frame[21:42]      # (21, 3)
    arms = frame[42:48]           # (6, 3)
    
    # Find missing keypoints (all zeros)
    missing_mask = (frame == 0).all(axis=1)  # (48,) boolean mask
    
    # Plot arms first (background)
    for connection in ARM_CONNECTIONS:
        i, j = connection
        if i < len(arms) and j < len(arms):
            p1, p2 = arms[i], arms[j]
            if not (missing_mask[42+i] or missing_mask[42+j]):  # Both points exist
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=4, alpha=0.8, label='Arms' if i == 0 else "")
    
    # Plot arm keypoints
    for i, point in enumerate(arms):
        if not missing_mask[42+i]:  # Point exists
            ax.plot(point[0], point[1], 'bo', markersize=10, label='Arms' if i == 0 else "")
        elif show_missing:
            ax.plot(0.5, 0.5, 'rx', markersize=12, label='Missing Arms' if i == 0 else "")
    
    # Plot right hand
    for connection in HAND_CONNECTIONS:
        i, j = connection
        p1, p2 = right_hand[i], right_hand[j]
        if not (missing_mask[i] or missing_mask[j]):  # Both points exist
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.9)
    
    # Plot right hand keypoints
    for i, point in enumerate(right_hand):
        if not missing_mask[i]:  # Point exists
            ax.plot(point[0], point[1], 'ro', markersize=6, label='Right Hand' if i == 0 else "")
        elif show_missing:
            ax.plot(0.1, 0.9, 'rx', markersize=8, label='Missing R.Hand' if i == 0 else "")
    
    # Plot left hand
    for connection in HAND_CONNECTIONS:
        i, j = connection
        p1, p2 = left_hand[i], left_hand[j]
        if not (missing_mask[21+i] or missing_mask[21+j]):  # Both points exist
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, alpha=0.9)
    
    # Plot left hand keypoints
    for i, point in enumerate(left_hand):
        if not missing_mask[21+i]:  # Point exists
            ax.plot(point[0], point[1], 'go', markersize=6, label='Left Hand' if i == 0 else "")
        elif show_missing:
            ax.plot(0.9, 0.9, 'gx', markersize=8, label='Missing L.Hand' if i == 0 else "")
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # MediaPipe coordinates have origin at top-left
    ax.set_title(f"{title}\nFrame {frame_idx}", fontsize=14, fontweight='bold')
    
    # Add statistics
    total_missing = missing_mask.sum()
    missing_pct = (total_missing / len(missing_mask)) * 100
    ax.text(0.02, 0.02, f"Missing: {total_missing}/48 ({missing_pct:.1f}%)", 
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            fontsize=12, fontweight='bold')
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3)
    
    return True

def plot_skeleton_frame_3d(ax, pose_data, frame_idx, title="", show_missing=True):
    """
    Plot a single frame of skeleton data with 3D visualization
    Only works if matplotlib is available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for 3D plotting")
        return False
    
    ax.clear()
    
    if frame_idx >= pose_data.shape[0]:
        ax.text2D(0.5, 0.5, f"Frame {frame_idx} not available", 
                  ha='center', va='center', transform=ax.transAxes)
        return True
    
    frame = pose_data[frame_idx]  # (48, 3)
    
    # Extract different body parts
    right_hand = frame[0:21]      # (21, 3)
    left_hand = frame[21:42]      # (21, 3)
    arms = frame[42:48]           # (6, 3)
    
    # Find missing keypoints (all zeros)
    missing_mask = (frame == 0).all(axis=1)  # (48,) boolean mask
    
    # Plot arms first (background)
    for connection in ARM_CONNECTIONS:
        i, j = connection
        if i < len(arms) and j < len(arms):
            p1, p2 = arms[i], arms[j]
            if not (missing_mask[42+i] or missing_mask[42+j]):  # Both points exist
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=4, alpha=0.8, label='Arms' if i == 0 else "")
    
    # Plot arm keypoints
    for i, point in enumerate(arms):
        if not missing_mask[42+i]:  # Point exists
            ax.scatter(point[0], point[1], point[2], c='blue', s=100, label='Arms' if i == 0 else "")
        elif show_missing:
            ax.scatter(0.5, 0.5, 0.5, c='red', marker='x', s=200, label='Missing Arms' if i == 0 else "")
    
    # Plot right hand
    for connection in HAND_CONNECTIONS:
        i, j = connection
        p1, p2 = right_hand[i], right_hand[j]
        if not (missing_mask[i] or missing_mask[j]):  # Both points exist
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2, alpha=0.9)
    
    # Plot right hand keypoints
    for i, point in enumerate(right_hand):
        if not missing_mask[i]:  # Point exists
            ax.scatter(point[0], point[1], point[2], c='red', s=60, label='Right Hand' if i == 0 else "")
        elif show_missing:
            ax.scatter(0.1, 0.9, 0.5, c='red', marker='x', s=100, label='Missing R.Hand' if i == 0 else "")
    
    # Plot left hand
    for connection in HAND_CONNECTIONS:
        i, j = connection
        p1, p2 = left_hand[i], left_hand[j]
        if not (missing_mask[21+i] or missing_mask[21+j]):  # Both points exist
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-', linewidth=2, alpha=0.9)
    
    # Plot left hand keypoints
    for i, point in enumerate(left_hand):
        if not missing_mask[21+i]:  # Point exists
            ax.scatter(point[0], point[1], point[2], c='green', s=60, label='Left Hand' if i == 0 else "")
        elif show_missing:
            ax.scatter(0.9, 0.9, 0.5, c='green', marker='x', s=100, label='Missing L.Hand' if i == 0 else "")
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\nFrame {frame_idx}", fontsize=14, fontweight='bold')
    
    # Add statistics
    total_missing = missing_mask.sum()
    missing_pct = (total_missing / len(missing_mask)) * 100
    ax.text2D(0.02, 0.02, f"Missing: {total_missing}/48 ({missing_pct:.1f}%)", 
              transform=ax.transAxes, 
              bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
              fontsize=12, fontweight='bold')
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3)
    
    return True

def create_comparison_plot(raw_file, cleaned_file, frame_idx=50, save_path=None, plot_3d=False):
    """Create side-by-side before/after comparison plot (2D or 3D)"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for comparison plots")
        print("💡 Use: python simple_skeleton_viz.py for text comparison")
        return False
    
    # Load data
    raw_data = np.load(raw_file)
    cleaned_data = np.load(cleaned_file)
    
    print(f"📊 Raw data shape: {raw_data.shape}")
    print(f"📊 Cleaned data shape: {cleaned_data.shape}")
    
    # Adjust frame index if cleaned data has fewer frames
    adjusted_frame = frame_idx
    if frame_idx >= cleaned_data.shape[0]:
        adjusted_frame = int(frame_idx * cleaned_data.shape[0] / raw_data.shape[0])
        print(f"📝 Adjusted frame index to {adjusted_frame} for cleaned data")
    
    # Create comparison plot
    if plot_3d:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot raw data
    if plot_3d:
        success1 = plot_skeleton_frame_3d(ax1, raw_data, frame_idx, 
                                         f"🔴 BEFORE: Raw Data\n{Path(raw_file).name}", 
                                         show_missing=True)
    else:
        success1 = plot_skeleton_frame_2d(ax1, raw_data, frame_idx, 
                                         f"🔴 BEFORE: Raw Data\n{Path(raw_file).name}", 
                                         show_missing=True)
    
    # Plot cleaned data  
    if plot_3d:
        success2 = plot_skeleton_frame_3d(ax2, cleaned_data, adjusted_frame, 
                                         f"🟢 AFTER: Cleaned Data\n{Path(cleaned_file).name}", 
                                         show_missing=True)
    else:
        success2 = plot_skeleton_frame_2d(ax2, cleaned_data, adjusted_frame, 
                                         f"🟢 AFTER: Cleaned Data\n{Path(cleaned_file).name}", 
                                         show_missing=True)
    
    if success1 and success2:
        # Add legend
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Add overall title
        plot_type = "3D" if plot_3d else "2D"
        fig.suptitle(f'🦴 SKELETON DATA: Before vs After Preprocessing ({plot_type})', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved comparison to {save_path}")
        
        # Try to show plot
        try:
            plt.show()
        except:
            print("📱 Plot created but display not available (headless mode)")
    
    return True

def create_animation_2d(data, output_path, title="Skeleton Animation", frames_to_show=100):
    """Create animated 2D skeleton movement"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for 2D animations")
        return False
    
    print(f"🎬 Creating 2D animation with {frames_to_show} frames...")
    
    # Limit frames for reasonable file size
    frames_to_show = min(frames_to_show, data.shape[0])
    data_subset = data[:frames_to_show]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def animate(frame_idx):
        plot_skeleton_frame_2d(ax, data_subset, frame_idx, f"{title}\nFrame {frame_idx}")
        
    try:
        anim = FuncAnimation(fig, animate, frames=frames_to_show, interval=150, repeat=True)
        
        if output_path:
            print(f"💾 Saving 2D animation to {output_path} (this may take a while...)")
            anim.save(output_path, writer='pillow', fps=8, dpi=100)
            print(f"✅ 2D animation saved successfully!")
        
        try:
            plt.show()
        except:
            print("📱 2D animation created but display not available (headless mode)")
        
        return anim
    except Exception as e:
        print(f"❌ 2D animation creation failed: {e}")
        return False

def create_animation_3d(data, output_path, title="Skeleton Animation", frames_to_show=100):
    """Create animated 3D skeleton movement"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for 3D animations")
        return False
    
    print(f"🎬 Creating 3D animation with {frames_to_show} frames...")
    
    # Limit frames for reasonable file size
    frames_to_show = min(frames_to_show, data.shape[0])
    data_subset = data[:frames_to_show]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame_idx):
        plot_skeleton_frame_3d(ax, data_subset, frame_idx, f"{title}\nFrame {frame_idx}")
        
    try:
        anim = FuncAnimation(fig, animate, frames=frames_to_show, interval=200, repeat=True)
        
        if output_path:
            print(f"💾 Saving 3D animation to {output_path} (this may take a while...)")
            anim.save(output_path, writer='pillow', fps=6, dpi=100)
            print(f"✅ 3D animation saved successfully!")
        
        try:
            plt.show()
        except:
            print("📱 3D animation created but display not available (headless mode)")
        
        return anim
    except Exception as e:
        print(f"❌ 3D animation creation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Advanced skeleton visualization with matplotlib support")
    parser.add_argument("--raw", required=True, help="Path to raw .npy file")
    parser.add_argument("--cleaned", help="Path to cleaned .npy file (for comparison)")
    parser.add_argument("--frame", type=int, default=50, help="Frame index to visualize (or number of frames for animation)")
    parser.add_argument("--animate", action="store_true", help="Create animation (requires matplotlib)")
    parser.add_argument("--animate_3d", action="store_true", help="Create 3D animation (requires matplotlib)")
    parser.add_argument("--save", help="Save comparison image to this path")
    parser.add_argument("--analyze", action="store_true", help="Print detailed analysis")
    parser.add_argument("--plot_3d", action="store_true", help="Use 3D plotting instead of 2D")
    
    args = parser.parse_args()
    
    print(f"🦴 ADVANCED SKELETON VISUALIZATION")
    print("=" * 50)
    print(f"📊 Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"🎬 Frame parameter: {args.frame}")
    print("💡 Note: --frame controls:")
    print("   - For static plots: specific frame index to visualize")
    print("   - For animations: number of frames to include in animation")
    
    if not os.path.exists(args.raw):
        print(f"❌ Error: Raw file {args.raw} not found")
        return
    
    # Load and analyze raw data
    raw_data = np.load(args.raw)
    
    if args.analyze:
        print_skeleton_stats(raw_data, f"Raw Data ({Path(args.raw).name})")
        if args.cleaned and os.path.exists(args.cleaned):
            cleaned_data = np.load(args.cleaned)
            print_skeleton_stats(cleaned_data, f"Cleaned Data ({Path(args.cleaned).name})")
    
    # Create visualizations if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        if args.animate_3d:
            # Create 3D animation
            if args.cleaned and os.path.exists(args.cleaned):
                cleaned_data = np.load(args.cleaned)
                create_animation_3d(cleaned_data, "cleaned_skeleton_3d_animation.gif", "Cleaned Skeleton 3D", args.frame)
            else:
                create_animation_3d(raw_data, "raw_skeleton_3d_animation.gif", "Raw Skeleton 3D", args.frame)
        elif args.animate:
            # Create 2D animation
            if args.cleaned and os.path.exists(args.cleaned):
                cleaned_data = np.load(args.cleaned)
                create_animation_2d(cleaned_data, "cleaned_skeleton_2d_animation.gif", "Cleaned Skeleton 2D", args.frame)
            else:
                create_animation_2d(raw_data, "raw_skeleton_2d_animation.gif", "Raw Skeleton 2D", args.frame)
        elif args.cleaned and os.path.exists(args.cleaned):
            # Create before/after comparison
            save_path = args.save if args.save else "skeleton_comparison.png"
            create_comparison_plot(args.raw, args.cleaned, args.frame, save_path, args.plot_3d)
        else:
            # Show only raw data
            if args.plot_3d:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                plot_skeleton_frame_3d(ax, raw_data, args.frame, f"Raw Data: {Path(args.raw).name}")
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
                plot_skeleton_frame_2d(ax, raw_data, args.frame, f"Raw Data: {Path(args.raw).name}")
            plt.show()
    else:
        print("\n💡 FALLBACK: Use simple_skeleton_viz.py for text-based analysis")
        print("💡 To enable graphics, fix the numpy/matplotlib version conflict")
    
    print(f"\n✅ Visualization complete!")

if __name__ == "__main__":
    main() 