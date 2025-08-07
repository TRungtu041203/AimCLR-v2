# File Interaction Flowchart for COBOT Integration

## 🔄 Complete Data Flow Pipeline

```mermaid
graph TD
    A[pose_clean/ raw .npy files<br/>Shape: T×48×3] --> B[tools/cobot_gendata.py]
    B --> C[cobot_dataset/xsub/<br/>train_data.npy, val_data.npy<br/>train_label.pkl, val_label.pkl<br/>Shape: N×3×300×48×1]
    
    C --> D[feeder/preprocess_cobot.py]
    D --> E[cobot_dataset_frame50/xsub/<br/>train_position.npy, val_position.npy<br/>train_motion.npy, val_motion.npy<br/>train_label.npy, val_label.npy<br/>Shape: N×3×50×48×1]
    
    E --> F[tools/convert_labels.py]
    F --> G[cobot_dataset_frame50/xsub/<br/>train_label.pkl, val_label.pkl]
    
    G --> H[main.py + config files]
    H --> I[work_dir/cobot_3views_xsub/]
    
    style A fill:#e1f5fe
    style I fill:#c8e6c9
```

## 🏗️ Detailed File Dependencies

### **Phase 1: Data Processing**
```mermaid
graph LR
    A[pose_clean/] --> B[cobot_gendata.py]
    B --> C[cobot_dataset/]
    
    subgraph "Data Processing"
        B
    end
    
    subgraph "Input"
        A
    end
    
    subgraph "Output"
        C
    end
```

**File: `tools/cobot_gendata.py`**
- **Input**: `pose_clean/` (raw .npy files)
- **Process**: Converts `(T, 48, 3)` → `(N, 3, 300, 48, 1)`
- **Output**: `cobot_dataset/xsub/` (standardized format)
- **Dependencies**: `numpy`, `open_memmap`

### **Phase 2: Preprocessing**
```mermaid
graph LR
    A[cobot_dataset/] --> B[preprocess_cobot.py]
    B --> C[cobot_dataset_frame50/]
    
    subgraph "Preprocessing"
        B
    end
    
    subgraph "Input"
        A
    end
    
    subgraph "Output"
        C
    end
```

**File: `feeder/preprocess_cobot.py`**
- **Input**: `cobot_dataset/xsub/`
- **Process**: 
  - Downsample to 50 frames
  - Generate motion data
  - Handle variable lengths
- **Output**: `cobot_dataset_frame50/xsub/`
- **Dependencies**: `NTUDatasets.py`, `torch`, `numpy`

### **Phase 3: Label Conversion**
```mermaid
graph LR
    A[cobot_dataset_frame50/] --> B[convert_labels.py]
    B --> C[.pkl labels]
    
    subgraph "Label Conversion"
        B
    end
    
    subgraph "Input"
        A
    end
    
    subgraph "Output"
        C
    end
```

**File: `tools/convert_labels.py`**
- **Input**: `.npy` label files
- **Process**: Convert to `.pkl` format
- **Output**: `.pkl` label files
- **Dependencies**: `numpy`, `pickle`

### **Phase 4: Training**
```mermaid
graph TD
    A[config/cobot/pretext/*.yaml] --> B[main.py]
    B --> C[processor/pretrain_aimclr_v2_3views.py]
    C --> D[feeder/ntu_feeder.py]
    D --> E[net/aimclr_v2_3views.py]
    E --> F[net/st_gcn.py]
    F --> G[net/utils/graph.py]
    G --> H[work_dir/]
    
    subgraph "Configuration"
        A
    end
    
    subgraph "Entry Point"
        B
    end
    
    subgraph "Training Logic"
        C
    end
    
    subgraph "Data Loading"
        D
    end
    
    subgraph "Model Architecture"
        E
    end
    
    subgraph "Backbone Network"
        F
    end
    
    subgraph "Graph Structure"
        G
    end
    
    subgraph "Output"
        H
    end
```

## 📊 Data Shape Transformations

### **Raw Data → Standardized**
```
pose_clean/ (raw)
├── Shape: (T, 48, 3) - variable length
└── Format: .npy files

cobot_gendata.py
└── Converts to: (N, 3, 300, 48, 1)
    ├── N = number of samples
    ├── 3 = x,y,z coordinates
    ├── 300 = max frames (padded/cropped)
    ├── 48 = joints
    └── 1 = single person
```

### **Standardized → Preprocessed**
```
cobot_dataset/xsub/
├── Shape: (N, 3, 300, 48, 1)
└── Format: .npy + .pkl

preprocess_cobot.py
└── Converts to: (N, 3, 50, 48, 1)
    ├── Downsampled to 50 frames
    ├── Generated motion data
    └── Handled variable lengths
```

### **Preprocessed → Training Ready**
```
cobot_dataset_frame50/xsub/
├── Position: (N, 3, 50, 48, 1)
├── Motion: (N, 3, 50, 48, 1)
└── Labels: .pkl format

main.py + config
└── Loads into training pipeline
```

## 🔧 Key File Modifications for COBOT

### **Network Architecture Changes**
```mermaid
graph TD
    A[net/aimclr_v2_3views.py] --> B[Updated Bone connections]
    C[net/utils/graph.py] --> D[Added COBOT layout]
    E[processor/pretrain_aimclr_v2.py] --> F[Updated Bone list]
    G[processor/linear_evaluation.py] --> H[Updated Bone list]
    
    subgraph "Bone Connection Updates"
        B
        F
        H
    end
    
    subgraph "Graph Structure Updates"
        D
    end
```

**Modified Files:**
1. **`net/aimclr_v2_3views.py`**
   - Updated `self.Bone` list for 48 joints
   - Added COBOT-specific bone connections

2. **`net/utils/graph.py`**
   - Added `cobot` layout
   - Defined 48-joint graph structure

3. **`processor/pretrain_aimclr_v2.py`**
   - Updated `Bone` list for bone stream processing

4. **`processor/linear_evaluation.py`**
   - Updated `Bone` list for bone stream processing

### **Configuration Files**
```mermaid
graph LR
    A[config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml] --> B[Pretraining config]
    C[config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml] --> D[Evaluation config]
    
    subgraph "Configuration"
        A
        C
    end
    
    subgraph "Usage"
        B
        D
    end
```

**New Configuration Files:**
- `config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml`
  - Sets `num_class: 3` (COBOT actions)
  - Sets `graph_args.layout: 'cobot'`
  - Adjusted hyperparameters for stability

- `config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml`
  - Sets `num_class: 3`
  - Sets `graph_args.layout: 'cobot'`

## 🚀 Execution Commands Flow

```mermaid
graph TD
    A[python tools/cobot_gendata.py] --> B[Data conversion]
    B --> C[python feeder/preprocess_cobot.py] --> D[Preprocessing]
    D --> E[python tools/convert_labels.py] --> F[Label conversion]
    F --> G[python main.py pretrain_aimclr_v2_3views] --> H[Pretraining]
    H --> I[python main.py linear_evaluation] --> J[Evaluation]
    
    subgraph "Data Pipeline"
        A
        C
        E
    end
    
    subgraph "Training Pipeline"
        G
        I
    end
    
    subgraph "Results"
        J
    end
```

## 🔍 Debugging Tools Integration

```mermaid
graph LR
    A[tools/debug_data.py] --> B[Data validation]
    C[tools/analyze_cobot.py] --> D[Data analysis]
    E[tools/visualize_cobot_skeleton.py] --> F[Skeleton visualization]
    G[test_cobot_integration.py] --> H[Integration testing]
    
    subgraph "Debugging Tools"
        A
        C
        E
        G
    end
    
    subgraph "Purpose"
        B
        D
        F
        H
    end
```

This comprehensive flowchart shows how each file interacts with others, the data transformations at each step, and the complete pipeline from raw data to final results.
