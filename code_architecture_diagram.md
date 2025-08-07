# 3s-AimCLR++ Code Architecture & File Interactions

## 🏗️ Overall Architecture Flow

```
Raw Data → Data Processing → Preprocessing → Training → Evaluation
    ↓           ↓              ↓           ↓         ↓
pose_clean → cobot_gendata → preprocess → main.py → Results
```

## 📊 Detailed File Interaction Diagram

### 1. **Data Processing Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   pose_clean/   │    │ cobot_gendata.py │    │ cobot_dataset/  │
│   (raw .npy)    │───▶│ (converts to     │───▶│ (standardized   │
│   (T, 48, 3)    │    │  standard format)│    │  format)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ cobot_dataset/  │
                       │ xsub/           │
                       │ ├─ train_data.npy│
                       │ ├─ val_data.npy  │
                       │ ├─ train_label.pkl│
                       │ └─ val_label.pkl │
                       └─────────────────┘
```

### 2. **Preprocessing Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ cobot_dataset/  │    │ preprocess_cobot.py│   │ cobot_dataset_  │
│ xsub/           │───▶│ (generates motion │───▶│ frame50/        │
│ (raw data)      │    │  & bone data)    │    │ xsub/           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ cobot_dataset_  │
                       │ frame50/xsub/   │
                       │ ├─ train_position.npy│
                       │ ├─ val_position.npy  │
                       │ ├─ train_motion.npy  │
                       │ ├─ val_motion.npy    │
                       │ ├─ train_label.npy   │
                       │ └─ val_label.npy     │
                       └─────────────────┘
```

### 3. **Label Conversion**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ train_label.npy │    │ convert_labels.py│    │ train_label.pkl │
│ val_label.npy   │───▶│ (converts .npy   │───▶│ val_label.pkl   │
│ (.npy format)   │    │  to .pkl format)│    │ (.pkl format)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 4. **Training Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ config/cobot/   │    │ main.py          │    │ work_dir/       │
│ pretext/        │───▶│ pretrain_aimclr_ │───▶│ cobot_3views_   │
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

### 5. **Evaluation Pipeline**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ config/cobot/   │    │ main.py          │    │ work_dir/       │
│ linear/         │───▶│ linear_evaluation│───▶│ cobot_3views_   │
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
tools/cobot_gendata.py
├── Depends on: pose_clean/ (raw data)
├── Uses: numpy, open_memmap
└── Outputs: cobot_dataset/xsub/

feeder/preprocess_cobot.py
├── Depends on: cobot_dataset/xsub/
├── Uses: NTUDatasets.py, torch, numpy
└── Outputs: cobot_dataset_frame50/xsub/

tools/convert_labels.py
├── Depends on: cobot_dataset_frame50/xsub/
├── Uses: numpy, pickle
└── Outputs: .pkl label files
```

### **Network Architecture Files**
```
net/aimclr_v2_3views.py
├── Depends on: net/st_gcn.py, net/utils/graph.py
├── Uses: torch, numpy
└── Implements: 3-stream contrastive learning

net/st_gcn.py
├── Depends on: net/utils/graph.py
├── Uses: torch
└── Implements: Spatial-Temporal Graph CNN

net/utils/graph.py
├── Depends on: None
├── Uses: numpy
└── Defines: skeleton graph structures (NTU, COBOT)
```

### **Processor Files**
```
processor/pretrain_aimclr_v2_3views.py
├── Depends on: net/aimclr_v2_3views.py
├── Uses: feeder/ntu_feeder.py, torch
└── Implements: 3-stream pretraining

processor/linear_evaluation.py
├── Depends on: net/aimclr_v2_3views.py
├── Uses: feeder/ntu_feeder.py, torch
└── Implements: linear evaluation

processor/pretrain_aimclr_v2.py
├── Depends on: net/aimclr_v2.py
├── Uses: feeder/ntu_feeder.py, torch
└── Implements: single-stream pretraining
```

### **Data Feeder Files**
```
feeder/ntu_feeder.py
├── Depends on: feeder/tools.py
├── Uses: torch, numpy, pickle
└── Implements: data loading and augmentation

feeder/tools.py
├── Depends on: None
├── Uses: numpy, scipy
└── Implements: data augmentation functions

feeder/NTUDatasets.py
├── Depends on: None
├── Uses: torch, numpy, pickle
└── Implements: advanced data preprocessing
```

## 🚀 Execution Flow

### **Step 1: Data Conversion**
```bash
python tools/cobot_gendata.py --data_path pose_clean --out_folder cobot_dataset --benchmark xsub
```
**Files Involved:**
- `tools/cobot_gendata.py` ← reads `pose_clean/`
- Outputs to `cobot_dataset/xsub/`

### **Step 2: Data Preprocessing**
```bash
python feeder/preprocess_cobot.py --dataset_path cobot_dataset --out_folder cobot_dataset_frame50
```
**Files Involved:**
- `feeder/preprocess_cobot.py` ← reads `cobot_dataset/xsub/`
- Uses `feeder/NTUDatasets.py` (COBOTMotionProcessor)
- Outputs to `cobot_dataset_frame50/xsub/`

### **Step 3: Label Conversion**
```bash
python tools/convert_labels.py
```
**Files Involved:**
- `tools/convert_labels.py` ← reads `cobot_dataset_frame50/xsub/`
- Outputs `.pkl` files to `cobot_dataset_frame50/xsub/`

### **Step 4: Pretraining**
```bash
python main.py pretrain_aimclr_v2_3views --config config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml
```
**Files Involved:**
- `main.py` ← reads config file
- `processor/pretrain_aimclr_v2_3views.py` ← loads data
- `feeder/ntu_feeder.py` ← data loading
- `net/aimclr_v2_3views.py` ← model definition
- `net/st_gcn.py` ← backbone network
- `net/utils/graph.py` ← graph structure
- Outputs to `work_dir/cobot_3views_xsub/pretext/`

### **Step 5: Linear Evaluation**
```bash
python main.py linear_evaluation --config config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml
```
**Files Involved:**
- `main.py` ← reads config file
- `processor/linear_evaluation.py` ← evaluation logic
- `feeder/ntu_feeder.py` ← data loading
- `net/aimclr_v2_3views.py` ← model definition
- Outputs to `work_dir/cobot_3views_xsub/linear/`

## 📁 File Organization by Purpose

### **Data Processing**
- `tools/cobot_gendata.py` - Raw data conversion
- `feeder/preprocess_cobot.py` - Data preprocessing
- `tools/convert_labels.py` - Label format conversion

### **Network Architecture**
- `net/aimclr_v2_3views.py` - 3-stream model
- `net/aimclr_v2.py` - Single-stream model
- `net/st_gcn.py` - ST-GCN backbone
- `net/utils/graph.py` - Graph definitions

### **Training Processors**
- `processor/pretrain_aimclr_v2_3views.py` - 3-stream pretraining
- `processor/pretrain_aimclr_v2.py` - Single-stream pretraining
- `processor/linear_evaluation.py` - Linear evaluation

### **Data Loading**
- `feeder/ntu_feeder.py` - Data feeders
- `feeder/NTUDatasets.py` - Dataset classes
- `feeder/tools.py` - Augmentation tools

### **Configuration**
- `config/cobot/pretext/` - Pretraining configs
- `config/cobot/linear/` - Evaluation configs

### **Orchestration**
- `main.py` - Main entry point
- `run_cobot.sh` - Automation script

## 🔗 Key Data Flow Connections

```
pose_clean/ (raw)
    ↓
cobot_gendata.py
    ↓
cobot_dataset/xsub/ (standardized)
    ↓
preprocess_cobot.py
    ↓
cobot_dataset_frame50/xsub/ (preprocessed)
    ↓
convert_labels.py
    ↓
cobot_dataset_frame50/xsub/ (.pkl labels)
    ↓
main.py + config files
    ↓
work_dir/ (trained models)
```

This architecture shows how data flows from raw format through multiple processing stages to final training and evaluation, with each file having specific responsibilities in the pipeline.
