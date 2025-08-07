# Simplified File Interactions for COBOT Integration

## 🎯 Core File Interaction Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    COBOT INTEGRATION FLOW                      │
└─────────────────────────────────────────────────────────────────┘

📁 RAW DATA
pose_clean/ (.npy files, T×48×3)
    ↓
🛠️ DATA PROCESSING
tools/cobot_gendata.py
    ↓
📊 STANDARDIZED DATA
cobot_dataset/xsub/ (N×3×300×48×1)
    ↓
⚙️ PREPROCESSING
feeder/preprocess_cobot.py
    ↓
🎯 TRAINING READY
cobot_dataset_frame50/xsub/ (N×3×50×48×1)
    ↓
🏷️ LABEL CONVERSION
tools/convert_labels.py
    ↓
🚀 TRAINING
main.py + config files
    ↓
📈 RESULTS
work_dir/cobot_3views_xsub/
```

## 🔧 Key File Modifications

### **Network Architecture Files**
```
net/aimclr_v2_3views.py
├── Updated Bone connections for 48 joints
└── COBOT-specific bone structure

net/utils/graph.py
├── Added 'cobot' layout
└── 48-joint graph definition

processor/pretrain_aimclr_v2.py
└── Updated Bone list for bone stream

processor/linear_evaluation.py
└── Updated Bone list for bone stream
```

### **New Files Created**
```
tools/cobot_gendata.py          # Raw data conversion
feeder/preprocess_cobot.py      # Data preprocessing
tools/convert_labels.py         # Label format conversion
config/cobot/pretext/*.yaml     # Pretraining config
config/cobot/linear/*.yaml      # Evaluation config
run_cobot.sh                    # Automation script
```

## 📊 Data Shape Evolution

```
Raw: (T, 48, 3)           # Variable length sequences
  ↓
Standardized: (N, 3, 300, 48, 1)  # Fixed length, padded
  ↓
Preprocessed: (N, 3, 50, 48, 1)   # Downsampled to 50 frames
  ↓
Training: (N, 3, 50, 48, 1)       # Ready for model
```

## 🔄 Execution Pipeline

```bash
# Step 1: Convert raw data
python tools/cobot_gendata.py --data_path pose_clean --out_folder cobot_dataset

# Step 2: Preprocess data
python feeder/preprocess_cobot.py --dataset_path cobot_dataset --out_folder cobot_dataset_frame50

# Step 3: Convert labels
python tools/convert_labels.py

# Step 4: Pretrain model
python main.py pretrain_aimclr_v2_3views --config config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml

# Step 5: Evaluate model
python main.py linear_evaluation --config config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml
```

## 🎯 File Responsibilities

### **Data Processing**
- `cobot_gendata.py` → Converts raw data to standard format
- `preprocess_cobot.py` → Generates motion/bone data, handles variable lengths
- `convert_labels.py` → Converts label format for compatibility

### **Model Architecture**
- `aimclr_v2_3views.py` → 3-stream contrastive learning model
- `st_gcn.py` → Spatial-Temporal Graph CNN backbone
- `graph.py` → Defines skeleton graph structures

### **Training Logic**
- `pretrain_aimclr_v2_3views.py` → 3-stream pretraining processor
- `linear_evaluation.py` → Linear evaluation processor
- `ntu_feeder.py` → Data loading and augmentation

### **Configuration**
- `*.yaml` files → Define model parameters, data paths, training settings

## 🔍 Key Interactions

```
main.py
├── Reads config files
├── Calls appropriate processor
└── Orchestrates training/evaluation

processor/*.py
├── Loads data via feeder
├── Instantiates model
└── Handles training loop

feeder/ntu_feeder.py
├── Loads data from disk
├── Applies augmentations
└── Returns batches to processor

net/aimclr_v2_3views.py
├── Defines 3-stream model
├── Uses ST-GCN backbone
└── Implements contrastive learning

net/st_gcn.py
├── Spatial-Temporal Graph CNN
├── Uses graph structure from graph.py
└── Processes skeleton sequences

net/utils/graph.py
├── Defines skeleton layouts
├── NTU: 25 joints, 2 persons
└── COBOT: 48 joints, 1 person
```

This simplified view shows the essential file interactions and data flow for integrating COBOT data into the 3s-AimCLR++ framework.
