# NTU vs COBOT: Original vs Modified Architecture Comparison

## 🔄 Data Flow Comparison

### **Original NTU Workflow**
```
NTU skeleton files (.skeleton)
    ↓
tools/ntu_gendata.py
    ↓
ntu_dataset/xsub/ (N×3×300×25×2)
    ↓
feeder/preprocess_ntu.py
    ↓
ntu_dataset_frame50/xsub/ (N×3×50×25×2)
    ↓
main.py + config files
    ↓
work_dir/ntu_3views_xsub/
```

### **COBOT Integration Workflow**
```
pose_clean/ (.npy files, T×48×3)
    ↓
tools/cobot_gendata.py
    ↓
cobot_dataset/xsub/ (N×3×300×48×1)
    ↓
feeder/preprocess_cobot.py
    ↓
cobot_dataset_frame50/xsub/ (N×3×50×48×1)
    ↓
tools/convert_labels.py
    ↓
main.py + config files
    ↓
work_dir/cobot_3views_xsub/
```

## 📊 Key Differences

| Aspect | Original NTU | COBOT Integration |
|--------|-------------|-------------------|
| **Raw Data Format** | `.skeleton` files | `.npy` files |
| **Data Shape** | `(T, 25, 3)` | `(T, 48, 3)` |
| **Joints** | 25 joints | 48 joints |
| **Persons** | 2 max | 1 person |
| **Actions** | 60 classes | 3 classes |
| **Data Processing** | `ntu_gendata.py` | `cobot_gendata.py` |
| **Preprocessing** | `preprocess_ntu.py` | `preprocess_cobot.py` |
| **Label Conversion** | Not needed | `convert_labels.py` |

## 🔧 File Modifications Summary

### **Files Modified for COBOT**

#### **1. Network Architecture Files**
```diff
# net/aimclr_v2_3views.py
- self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), ...]  # 25 joints
+ self.Bone = [(43, 44), (44, 45), ..., (21, 43), (42, 48)]      # 48 joints

# net/utils/graph.py
+ elif layout == 'cobot':
+     self.num_node = 48
+     # COBOT-specific bone connections

# processor/pretrain_aimclr_v2.py
- Bone = [(1, 2), (2, 21), (3, 21), ...]  # 25 joints
+ Bone = [(43, 44), (44, 45), ..., (21, 43), (42, 48)]  # 48 joints

# processor/linear_evaluation.py
- Bone = [(1, 2), (2, 21), (3, 21), ...]  # 25 joints
+ Bone = [(43, 44), (44, 45), ..., (21, 43), (42, 48)]  # 48 joints
```

#### **2. Configuration Files**
```diff
# Original: config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml
- num_class: 60
- graph_args:
-   layout: 'ntu-rgb+d'

# New: config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml
+ num_class: 3
+ graph_args:
+   layout: 'cobot'
```

### **New Files Created for COBOT**

```
tools/cobot_gendata.py              # Raw data conversion
feeder/preprocess_cobot.py          # Data preprocessing
tools/convert_labels.py             # Label format conversion
config/cobot/pretext/*.yaml         # Pretraining config
config/cobot/linear/*.yaml          # Evaluation config
run_cobot.sh                        # Automation script
tools/analyze_cobot.py              # Data analysis
tools/visualize_cobot_skeleton.py   # Skeleton visualization
test_cobot_integration.py           # Integration testing
tools/debug_data.py                 # Data debugging
```

## 🎯 Data Shape Evolution Comparison

### **NTU Data Shapes**
```
Raw: NTU skeleton files
  ↓
Standardized: (N, 3, 300, 25, 2)  # 25 joints, 2 persons
  ↓
Preprocessed: (N, 3, 50, 25, 2)   # Downsampled to 50 frames
```

### **COBOT Data Shapes**
```
Raw: (T, 48, 3)                   # Variable length, 48 joints
  ↓
Standardized: (N, 3, 300, 48, 1)  # 48 joints, 1 person
  ↓
Preprocessed: (N, 3, 50, 48, 1)   # Downsampled to 50 frames
```

## 🔍 Execution Commands Comparison

### **Original NTU Commands**
```bash
# Step 1: Convert NTU data
python tools/ntu_gendata.py --data_path /path/to/ntu/skeleton --out_folder ntu_dataset --benchmark xsub

# Step 2: Preprocess NTU data
python feeder/preprocess_ntu.py --dataset_path ntu_dataset --out_folder ntu_dataset_frame50

# Step 3: Pretrain
python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml

# Step 4: Evaluate
python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml
```

### **COBOT Integration Commands**
```bash
# Step 1: Convert COBOT data
python tools/cobot_gendata.py --data_path pose_clean --out_folder cobot_dataset --benchmark xsub

# Step 2: Preprocess COBOT data
python feeder/preprocess_cobot.py --dataset_path cobot_dataset --out_folder cobot_dataset_frame50

# Step 3: Convert labels
python tools/convert_labels.py

# Step 4: Pretrain
python main.py pretrain_aimclr_v2_3views --config config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml

# Step 5: Evaluate
python main.py linear_evaluation --config config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml
```

## 📁 File Organization Comparison

### **Original NTU Structure**
```
tools/
├── ntu_gendata.py
└── utils/
    └── ntu_read_skeleton.py

feeder/
├── preprocess_ntu.py
├── ntu_feeder.py
├── NTUDatasets.py
└── tools.py

config/
└── three-stream/
    ├── pretext/
    └── linear/

net/
├── aimclr_v2_3views.py (NTU bone connections)
├── aimclr_v2.py
├── st_gcn.py
└── utils/
    └── graph.py (NTU-RGB+D layout)

processor/
├── pretrain_aimclr_v2_3views.py
├── pretrain_aimclr_v2.py
└── linear_evaluation.py
```

### **COBOT Integration Structure**
```
tools/
├── cobot_gendata.py (NEW)
├── convert_labels.py (NEW)
├── analyze_cobot.py (NEW)
├── visualize_cobot_skeleton.py (NEW)
├── debug_data.py (NEW)
└── utils/
    └── ntu_read_skeleton.py

feeder/
├── preprocess_cobot.py (NEW)
├── ntu_feeder.py
├── NTUDatasets.py
└── tools.py

config/
├── three-stream/ (original)
└── cobot/ (NEW)
    ├── pretext/
    └── linear/

net/
├── aimclr_v2_3views.py (MODIFIED - COBOT bone connections)
├── aimclr_v2.py
├── st_gcn.py
└── utils/
    └── graph.py (MODIFIED - added COBOT layout)

processor/
├── pretrain_aimclr_v2_3views.py
├── pretrain_aimclr_v2.py (MODIFIED - COBOT bone connections)
└── linear_evaluation.py (MODIFIED - COBOT bone connections)

run_cobot.sh (NEW)
test_cobot_integration.py (NEW)
```

## 🔧 Key Technical Differences

### **Bone Connection Differences**

#### **NTU Bone Connections (25 joints)**
```python
# Original NTU bone connections
self.Bone = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
    (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)
]
```

#### **COBOT Bone Connections (48 joints)**
```python
# COBOT bone connections
self.Bone = [
    # Arm and shoulder connections (6 joints: 43-48)
    (43, 44), (44, 45), (45, 46), (46, 47), (47, 48),
    # Right hand connections (21 joints: 1-21)
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
    (17, 18), (18, 19), (19, 20), (20, 21),
    # Left hand connections (21 joints: 22-42)
    (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
    (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
    (38, 39), (39, 40), (40, 41), (41, 42),
    # Connect hands to arms
    (21, 43), (42, 48),  # Connect right hand to right arm, left hand to left arm
]
```

### **Graph Layout Differences**

#### **NTU Graph Layout**
```python
# net/utils/graph.py (original)
elif layout == 'ntu-rgb+d':
    self.num_node = 25
    self_link = [(i, i) for i in range(self.num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), ...]  # 25 joints
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    self.edge = self_link + neighbor_link
    self.center = 21 - 1
```

#### **COBOT Graph Layout**
```python
# net/utils/graph.py (added)
elif layout == 'cobot':
    self.num_node = 48
    self_link = [(i, i) for i in range(self.num_node)]
    neighbor_1base = [(43, 44), (44, 45), ..., (21, 43), (42, 48)]  # 48 joints
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    self.edge = self_link + neighbor_link
    self.center = 45  # Use middle arm joint as center
```

## 🎯 Summary of Changes

### **What Stayed the Same**
- Core architecture (`main.py`, `processor/*.py`, `feeder/ntu_feeder.py`)
- ST-GCN backbone (`net/st_gcn.py`)
- Data augmentation tools (`feeder/tools.py`)
- Training logic and evaluation procedures

### **What Changed**
- **Data processing**: New scripts for COBOT data format
- **Network architecture**: Updated bone connections and graph layout
- **Configuration**: New YAML files for COBOT parameters
- **Data shapes**: From 25 joints to 48 joints, from 2 persons to 1 person
- **Action classes**: From 60 classes to 3 classes

### **What Was Added**
- COBOT-specific data processing pipeline
- Label conversion step
- COBOT skeleton visualization tools
- Debugging and testing scripts
- COBOT configuration files

This comparison shows how the original NTU workflow was systematically adapted to handle the COBOT dataset while maintaining the core 3s-AimCLR++ architecture.
