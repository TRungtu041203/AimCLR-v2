# COBOT Integration Debug Summary

## Issues Encountered

### 1. Missing Dependencies
**Problem**: `ModuleNotFoundError: No module named 'h5py'`
**Solution**: Installed missing packages:
```bash
python -m pip install h5py tensorboardX opencv-python PyYAML
```

### 2. Label File Format
**Problem**: Config expected `.pkl` files but preprocessing created `.npy` files
**Solution**: Created `tools/convert_labels.py` to convert labels to correct format

### 3. Training Instability (NaN Loss)
**Problem**: Training produces NaN values in loss
**Solutions Applied**:
- Reduced learning rate from 0.1 to 0.01
- Reduced batch size from 128 to 32
- Reduced queue size from 32768 to 8192
- Increased temperature from 0.07 to 0.1
- Reduced dropout from 0.5 to 0.3

## Current Status

✅ **Dependencies**: All required packages installed
✅ **Data Conversion**: Raw data converted to 3s-AimCLR++ format
✅ **Data Preprocessing**: Position and motion data generated
✅ **Label Conversion**: Labels converted to correct format
✅ **Configuration**: Updated for stability

## Remaining Issues

❌ **Training Stability**: Still experiencing NaN values during training
❌ **Data Loading**: Potential issues with large data files

## Recommended Next Steps

### 1. Data Validation
```bash
# Check if data files are corrupted
python -c "import numpy as np; data = np.load('cobot_dataset_frame50/xsub/train_position.npy'); print('Data loaded successfully')"
```

### 2. Reduce Data Size for Testing
```bash
# Create a smaller test dataset
python tools/cobot_gendata.py --data_path pose_clean --out_folder cobot_dataset_test --benchmark xsub
# Use only first 10 files for testing
```

### 3. Alternative Training Approach
```bash
# Try single-stream training first
python main.py pretrain_aimclr_v2 --config config/cobot/pretext/pretext_aimclr_v2_cobot_xsub.yaml
```

### 4. Memory Optimization
- Reduce `batch_size` to 16
- Reduce `queue_size` to 4096
- Use CPU training for debugging

## Configuration Changes Made

### `config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml`
```yaml
# Reduced for stability
base_lr: 0.01  # Was 0.1
batch_size: 32  # Was 128
queue_size: 8192  # Was 32768
Temperature: 0.1  # Was 0.07
dropout: 0.3  # Was 0.5
```

## Files Created for Debugging

1. `tools/convert_labels.py` - Convert label formats
2. `tools/debug_data.py` - Check data integrity
3. `test_simple.py` - Basic data loading test
4. `DEBUG_SUMMARY.md` - This summary

## Potential Root Causes

1. **Data Scale**: COBOT data might have different scale than NTU data
2. **Graph Structure**: 48-joint graph might need different parameters
3. **Memory Issues**: Large data files causing memory problems
4. **Numerical Instability**: Learning rate or model parameters too aggressive

## Quick Fixes to Try

1. **Normalize Data**: Add data normalization in preprocessing
2. **Smaller Model**: Reduce hidden dimensions
3. **Gradient Clipping**: Add gradient clipping to prevent explosion
4. **Different Optimizer**: Try Adam instead of SGD
5. **Warmup**: Add learning rate warmup

## Success Criteria

- [ ] Training runs without NaN values
- [ ] Loss decreases over time
- [ ] Model saves successfully
- [ ] Linear evaluation works
- [ ] Reasonable accuracy achieved 