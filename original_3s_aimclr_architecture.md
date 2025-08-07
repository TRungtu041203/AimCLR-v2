# Original 3s-AimCLR++ Code Architecture & File Interactions

## 🏗️ Overall Architecture Flow

```
Raw NTU Data → Data Processing → Preprocessing → Training → Evaluation
    ↓              ↓              ↓           ↓         ↓
skeleton files → ntu_gendata → preprocess → main.py → Results
```

## 📊 Detailed File Interaction Diagram

### 1. **Original Data Processing Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NTU skeleton  │    │ ntu_gendata.py   │    │ ntu_dataset/    │
│   files (.skeleton)│───▶│ (converts to     │───▶│ (standardized   │
│   (raw format)  │    │  standard format)│    │  format)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ ntu_dataset/    │
                       │ xsub/xview/     │
                       │ ├─ train_data.npy│
                       │ ├─ val_data.npy  │
                       │ ├─ train_label.pkl│
                       │ └─ val_label.pkl │
                       └─────────────────┘
```

### 2. **Original Preprocessing Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ntu_dataset/    │    │ preprocess_ntu.py│    │ ntu_dataset_    │
│ xsub/xview/     │───▶│ (generates motion │───▶│ frame50/        │
│ (raw data)      │    │  & bone data)    │    │ xsub/xview/     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ ntu_dataset_    │
                       │ frame50/xsub/   │
                       │ ├─ train_position.npy│
                       │ ├─ val_position.npy  │
                       │ ├─ train_motion.npy  │
                       │ ├─ val_motion.npy    │
                       │ ├─ train_label.npy   │
                       │ └─ val_label.npy     │
                       └─────────────────┘
```

### 3. **Original Training Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ config/three-   │    │ main.py          │    │ work_dir/       │
│ stream/pretext/ │───▶│ pretrain_aimclr_ │───▶│ ntu_3views_     │
│ *.yaml          │    │ v2_3views        │    │ xsub/pretext/   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ net/            │
                       │ ├─ aimclr_v2_3views.py│
                       │ ├─ st_gcn.py    │
                       │ └─ utils/graph.py│
                       └─────────────────┘
```

### 4. **Original Evaluation Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ config/three-   │    │ main.py          │    │ work_dir/       │
│ stream/linear/  │───▶│ linear_evaluation│───▶│ ntu_3views_     │
│ *.yaml          │    │                  │    │ xsub/linear/    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ processor/      │
                       │ ├─ linear_evaluation.py│
                       │ └─ pretrain_aimclr_v2.py│
                       └─────────────────┘
```

## 🔄 Core File Dependencies

### **Data Processing Files**
```
tools/ntu_gendata.py
├── Depends on: NTU skeleton files (.skeleton)
├── Uses: utils/ntu_read_skeleton.py, numpy, open_memmap
└── Outputs: ntu_dataset/xsub/xview/

feeder/preprocess_ntu.py
├── Depends on: ntu_dataset/xsub/xview/
├── Uses: NTUDatasets.py, torch, numpy
└── Outputs: ntu_dataset_frame50/xsub/xview/
```

### **Network Architecture Files**
```
net/aimclr_v2_3views.py
├── Depends on: net/st_gcn.py, net/utils/graph.py
├── Uses: torch, numpy
├── Implements: 3-stream contrastive learning
└── Bone connections: 25 joints (NTU format)

net/aimclr_v2.py
├── Depends on: net/st_gcn.py, net/utils/graph.py
├── Uses: torch, numpy
└── Implements: single-stream contrastive learning

net/st_gcn.py
├── Depends on: net/utils/graph.py
├── Uses: torch
└── Implements: Spatial-Temporal Graph CNN

net/utils/graph.py
├── Depends on: None
├── Uses: numpy
└── Defines: skeleton graph structures (NTU-RGB+D, OpenPose, NW-UCLA)
```

### **Processor Files**
```
processor/pretrain_aimclr_v2_3views.py
├── Depends on: net/aimclr_v2_3views.py
├── Uses: feeder/ntu_feeder.py, torch
└── Implements: 3-stream pretraining

processor/pretrain_aimclr_v2.py
├── Depends on: net/aimclr_v2.py
├── Uses: feeder/ntu_feeder.py, torch
└── Implements: single-stream pretraining

processor/linear_evaluation.py
├── Depends on: net/aimclr_v2_3views.py
├── Uses: feeder/ntu_feeder.py, torch
└── Implements: linear evaluation
```

### **Data Feeder Files**
```
feeder/ntu_feeder.py
├── Depends on: feeder/tools.py
├── Uses: torch, numpy, pickle
├── Implements: data loading and augmentation
└── Classes: Feeder_single, Feeder_triple

feeder/tools.py
├── Depends on: None
├── Uses: numpy, scipy
└── Implements: data augmentation functions

feeder/NTUDatasets.py
├── Depends on: None
├── Uses: torch, numpy, pickle
├── Implements: advanced data preprocessing
└── Classes: SimpleLoader, NTUMotionProcessor
```

### **Utility Files**
```
tools/utils/ntu_read_skeleton.py
├── Depends on: None
├── Uses: numpy
└── Implements: NTU skeleton file reading

tools/ntu_gendata.py
├── Depends on: utils/ntu_read_skeleton.py
├── Uses: numpy, open_memmap
└── Implements: NTU data conversion
```

## 🚀 Original Execution Flow

### **Step 1: NTU Data Conversion**
```bash
python tools/ntu_gendata.py --data_path /path/to/ntu/skeleton --out_folder ntu_dataset --benchmark xsub
```
**Files Involved:**
- `tools/ntu_gendata.py` ← reads NTU skeleton files
- Uses `tools/utils/ntu_read_skeleton.py`
- Outputs to `ntu_dataset/xsub/`

### **Step 2: NTU Data Preprocessing**
```bash
python feeder/preprocess_ntu.py --dataset_path ntu_dataset --out_folder ntu_dataset_frame50
```
**Files Involved:**
- `feeder/preprocess_ntu.py` ← reads `ntu_dataset/xsub/`
- Uses `feeder/NTUDatasets.py` (NTUMotionProcessor)
- Outputs to `ntu_dataset_frame50/xsub/`

### **Step 3: Pretraining**
```bash
python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml
```
**Files Involved:**
- `main.py` ← reads config file
- `processor/pretrain_aimclr_v2_3views.py` ← loads data
- `feeder/ntu_feeder.py` ← data loading
- `net/aimclr_v2_3views.py` ← model definition
- `net/st_gcn.py` ← backbone network
- `net/utils/graph.py` ← graph structure
- Outputs to `work_dir/ntu_3views_xsub/pretext/`

### **Step 4: Linear Evaluation**
```bash
python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml
```
**Files Involved:**
- `main.py` ← reads config file
- `processor/linear_evaluation.py` ← evaluation logic
- `feeder/ntu_feeder.py` ← data loading
- `net/aimclr_v2_3views.py` ← model definition
- Outputs to `work_dir/ntu_3views_xsub/linear/`

## 📊 Original Data Shape Transformations

### **Raw NTU Data → Standardized**
```
NTU skeleton files (raw)
├── Format: .skeleton files
└── Contains: frame info, body info, joint coordinates

ntu_gendata.py
└── Converts to: (N, 3, 300, 25, 2)
    ├── N = number of samples
    ├── 3 = x,y,z coordinates
    ├── 300 = max frames
    ├── 25 = joints (NTU format)
    └── 2 = max persons
```

### **Standardized → Preprocessed**
```
ntu_dataset/xsub/
├── Shape: (N, 3, 300, 25, 2)
└── Format: .npy + .pkl

preprocess_ntu.py
└── Converts to: (N, 3, 50, 25, 2)
    ├── Downsampled to 50 frames
    ├── Generated motion data
    └── Handled variable lengths
```

## 📁 Original File Organization by Purpose

### **Data Processing**
- `tools/ntu_gendata.py` - NTU raw data conversion
- `feeder/preprocess_ntu.py` - NTU data preprocessing
- `tools/utils/ntu_read_skeleton.py` - NTU skeleton file reading

### **Network Architecture**
- `net/aimclr_v2_3views.py` - 3-stream model (NTU bone connections)
- `net/aimclr_v2.py` - Single-stream model
- `net/st_gcn.py` - ST-GCN backbone
- `net/utils/graph.py` - Graph definitions (NTU-RGB+D layout)

### **Training Processors**
- `processor/pretrain_aimclr_v2_3views.py` - 3-stream pretraining
- `processor/pretrain_aimclr_v2.py` - Single-stream pretraining
- `processor/linear_evaluation.py` - Linear evaluation

### **Data Loading**
- `feeder/ntu_feeder.py` - Data feeders (Feeder_single, Feeder_triple)
- `feeder/NTUDatasets.py` - Dataset classes (SimpleLoader, NTUMotionProcessor)
- `feeder/tools.py` - Augmentation tools

### **Configuration**
- `config/three-stream/pretext/` - Pretraining configs
- `config/three-stream/linear/` - Evaluation configs

### **Orchestration**
- `main.py` - Main entry point
- `run_3views.sh` - Automation script

## 🔗 Original Key Data Flow Connections

```
NTU skeleton files (raw)
    ↓
ntu_gendata.py
    ↓
ntu_dataset/xsub/ (standardized)
    ↓
preprocess_ntu.py
    ↓
ntu_dataset_frame50/xsub/ (preprocessed)
    ↓
main.py + config files
    ↓
work_dir/ (trained models)
```

## 🎯 Original File Responsibilities

### **Data Processing**
- `ntu_gendata.py` → Converts NTU skeleton files to standard format
- `preprocess_ntu.py` → Generates motion/bone data, handles variable lengths
- `ntu_read_skeleton.py` → Reads NTU skeleton file format

### **Model Architecture**
- `aimclr_v2_3views.py` → 3-stream contrastive learning model (NTU bone connections)
- `st_gcn.py` → Spatial-Temporal Graph CNN backbone
- `graph.py` → Defines skeleton graph structures (NTU-RGB+D)

### **Training Logic**
- `pretrain_aimclr_v2_3views.py` → 3-stream pretraining processor
- `linear_evaluation.py` → Linear evaluation processor
- `ntu_feeder.py` → Data loading and augmentation

### **Configuration**
- `*.yaml` files → Define model parameters, data paths, training settings

## 🔍 Original Key Interactions

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
├── Implements contrastive learning
└── NTU bone connections (25 joints)

net/st_gcn.py
├── Spatial-Temporal Graph CNN
├── Uses graph structure from graph.py
└── Processes skeleton sequences

net/utils/graph.py
├── Defines skeleton layouts
├── NTU-RGB+D: 25 joints, 2 persons
├── OpenPose: 18 joints, 2 persons
└── NW-UCLA: 20 joints, 1 person
```

## 📊 Original NTU Data Specifications

### **NTU Dataset Parameters**
```
max_body = 2          # 2 persons max
num_joint = 25        # 25 joints per person
max_frame = 300       # 300 frames max
num_class = 60        # 60 action classes
```

### **NTU Bone Connections (Original)**
```python
# From net/aimclr_v2_3views.py (original)
self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
             (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
             (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
             (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
             (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
```

This architecture shows how the original 3s-AimCLR++ framework processes NTU RGB+D data through multiple stages, with each file having specific responsibilities in the pipeline.
