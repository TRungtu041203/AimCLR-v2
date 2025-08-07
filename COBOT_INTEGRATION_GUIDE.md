# COBOT Dataset Integration Guide for 3s-AimCLR++

## Overview

This guide explains how to integrate your COBOT human-action recognition dataset with the 3s-AimCLR++ project for self-supervised learning.

## Dataset Structure

Your COBOT dataset has the following structure:
- **48 joints total**: 42 hand joints (21 per hand) + 6 arm/shoulder joints
- **File format**: `s{subject_id}_{person_name}_{action_id}.npy`
- **Data shape**: `(frames, 48, 3)` - (T, V, C) format
- **Actions**: 3 different actions (0, 1, 2)
- **Subjects**: 47 unique subjects

## Bone Connections

The COBOT skeleton uses the following bone connections:

### Arm and Shoulder (Joints 43-48)
- (43, 44), (44, 45), (45, 46), (46, 47), (47, 48)

### Right Hand (Joints 1-21)
- (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)
- (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17)
- (17, 18), (18, 19), (19, 20), (20, 21)

### Left Hand (Joints 22-42)
- (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
- (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38)
- (38, 39), (39, 40), (40, 41), (41, 42)

### Hand-Arm Connections
- (21, 43) - Right hand to right arm
- (42, 48) - Left hand to left arm

## Integration Steps

### Step 1: Data Conversion
```bash
python tools/cobot_gendata.py --data_path pose_clean --out_folder cobot_dataset --benchmark xsub
```

This converts your raw `.npy` files to the format expected by 3s-AimCLR++.

### Step 2: Data Preprocessing
```bash
python feeder/preprocess_cobot.py --dataset_path cobot_dataset --out_folder cobot_dataset_frame50
```

This generates position and motion data for the 3-stream architecture.

### Step 3: Test Integration
```bash
python test_cobot_integration.py
```

This verifies that all components are working correctly.

### Step 4: Pretrain Model
```bash
python main.py pretrain_aimclr_v2_3views --config config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml
```

### Step 5: Linear Evaluation
```bash
python main.py linear_evaluation --config config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml
```

## Complete Pipeline

Run the entire pipeline at once:
```bash
./run_cobot.sh
```

## Configuration Files

### Pretraining Config
- **File**: `config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml`
- **Key settings**:
  - `num_class: 3` (COBOT has 3 actions)
  - `graph_args.layout: 'cobot'` (Custom COBOT layout)
  - `device: [0]` (Adjust for your GPU)

### Linear Evaluation Config
- **File**: `config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml`
- **Key settings**:
  - `weights: ./work_dir/cobot_3views_xsub/pretext/epoch_399_model.pt`
  - `num_class: 3`

## Data Flow

```
pose_clean/ (your raw data)
    ↓
cobot_dataset/ (converted format)
    ↓
cobot_dataset_frame50/ (preprocessed for 3-stream)
    ↓
3s-AimCLR++ Training
    ↓
Results in work_dir/
```

## Key Features

### 3-Stream Architecture
1. **Joint Stream**: Original skeleton joint coordinates
2. **Motion Stream**: Temporal differences between frames
3. **Bone Stream**: Spatial relationships between connected joints

### Self-Supervised Learning
- **Extreme Augmentation**: Strong data augmentations for contrastive learning
- **Multi-Stream Mining**: Cross-stream interaction and mining strategies
- **Momentum Encoder**: MoCo-style momentum encoder for stable training

## Customization Options

### Bone Connections
If you need to modify the bone connections, update these files:
- `net/utils/graph.py` - Graph layout
- `net/aimclr_v2_3views.py` - Bone connections in model
- `processor/pretrain_aimclr_v2.py` - Training bone connections
- `processor/linear_evaluation.py` - Evaluation bone connections

### Number of Actions
Update `num_class` in the config files to match your actual number of actions.

### GPU Configuration
Update `device` in the config files to use your available GPU.

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config files
- Use smaller `queue_size` for contrastive learning

### GPU Issues
- Change `device` in config files
- Ensure CUDA is properly installed

### Data Format Issues
- Verify your `.npy` files have shape `(frames, 48, 3)`
- Check that all files follow the naming convention

## Expected Results

With the current setup:
- **Training samples**: 96
- **Validation samples**: 45
- **Actions**: 3 classes
- **Joints**: 48 (42 hand + 6 arm/shoulder)
- **Frames**: 50 (downsampled from variable length)

## Analysis Tools

### Data Analysis
```bash
python tools/analyze_cobot.py
```

### Skeleton Visualization
```bash
python tools/visualize_cobot_skeleton.py
```

## Files Created

### Data Processing
- `tools/cobot_gendata.py` - Raw data conversion
- `feeder/preprocess_cobot.py` - 3-stream preprocessing
- `tools/analyze_cobot.py` - Data analysis
- `tools/visualize_cobot_skeleton.py` - Skeleton visualization

### Configuration
- `config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml` - Pretraining config
- `config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml` - Evaluation config

### Network Updates
- `net/utils/graph.py` - Added COBOT graph layout
- `net/aimclr_v2_3views.py` - Updated bone connections
- `processor/pretrain_aimclr_v2.py` - Updated bone connections
- `processor/linear_evaluation.py` - Updated bone connections

### Testing
- `test_cobot_integration.py` - Integration test script
- `run_cobot.sh` - Complete pipeline script

## Next Steps

1. **Run the complete pipeline**: `./run_cobot.sh`
2. **Monitor training**: Check logs in `work_dir/`
3. **Evaluate results**: Compare with baseline methods
4. **Customize**: Adjust hyperparameters based on your needs

The integration is now complete and ready for training! 