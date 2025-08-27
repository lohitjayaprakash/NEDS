# NEDS: Neural Encoding and Decoding at Scale - Architecture Documentation

## Overview

NEDS is a revolutionary multimodal transformer-based model that simultaneously handles neural signals (spike data + LFP) and behavioral data for both encoding and decoding tasks using a novel multi-task masking strategy.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Multi-Task Masking Strategy](#multi-task-masking-strategy)
- [Session Stitching](#session-stitching)
- [Training Modes](#training-modes)
- [Data Specifications](#data-specifications)
- [Key Innovations](#key-innovations)

---

## Model Architecture

### Core Components Overview

```mermaid
graph TB
    subgraph "Input Data"
        A1[Neural Data<br/>Spike Trains]
        A2[LFP Data<br/>Optional]
        A3[Static Behavior<br/>choice, block]
        A4[Dynamic Behavior<br/>wheel, whisker]
    end
    
    subgraph "Encoder Embeddings"
        B1[Session Embeddings<br/>Handle variable neurons]
        B2[Modality Embeddings<br/>5 modality types]
        B3[Positional Embeddings<br/>Temporal positions]
        B4[Stitching Layer<br/>Variable dimensions]
    end
    
    subgraph "Multi-Task Masker"
        C1[Encoding Mask<br/>Neural → Behavior]
        C2[Decoding Mask<br/>Behavior → Neural]
        C3[Self-Modality Mask<br/>Within modality]
        C4[Cross-Modality Mask<br/>Between modalities]
        C5[Random Token Mask<br/>Temporal masking]
    end
    
    subgraph "Transformer Backbone"
        D1[Multi-Head Attention<br/>with RoPE]
        D2[Layer Normalization]
        D3[MLP Layers]
        D4[5 Transformer Layers]
    end
    
    subgraph "Output Decoders"
        E1[Spike Decoder<br/>Poisson NLL Loss]
        E2[Dynamic Decoder<br/>MSE Loss]
        E3[Static Decoder<br/>CrossEntropy Loss]
        E4[Session Stitching<br/>Variable outputs]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B2
    A4 --> B2
    
    B1 --> B4
    B2 --> B3
    B3 --> C1
    B4 --> C2
    
    C1 --> D1
    C2 --> D1
    C3 --> D2
    C4 --> D3
    C5 --> D4
    
    D1 --> E1
    D2 --> E2
    D3 --> E3
    D4 --> E4
```

### Detailed Architecture Diagram

```
🧠 NEURAL DATA                    📊 SESSION EMBEDDINGS              🎭 MIXED MASKING                   🔍 MULTI-HEAD ATTENTION            ⚡ SPIKE PREDICTION
• Spike trains (20ms bins)    →   • Handle 100-800 neurons       →   • Random scheme selection      →   • 8 attention heads            →   • Poisson distribution
• LFP features (optional)         • Session-specific parameters      • Batch-wise variation             • Flash attention optimized        • Count data modeling
• Variable neurons per session                                                                                                               
                                       ↓                                     ↓                                     ↓                              ↓
🐭 BEHAVIORAL DATA                🔄 MODALITY EMBEDDINGS              → ENCODING: Mask behavior          📏 LAYER NORMALIZATION             🎯 BEHAVIOR PREDICTION
• Static: choice, block       →   • 5 modality types             →   ← DECODING: Mask neural       →   • ScaleNorm option             →   • MSE for continuous
• Dynamic: wheel, whisker         • Cross-modal attention            ↻ SELF: Within modality           • Gradient stabilization           • CrossEntropy for discrete
• Continuous signals                                                 ↔ CROSS: Between modality                                             
                                       ↓                                     ↓                                     ↓                              ↓
                                  📍 POSITION EMBEDDINGS              TEMPORAL MASKING                   � MLP LAYERS                       🧩 SESSION STITCHING
                                  • Temporal sequences                • Consecutive time bins            • GELU activation                   • Variable output dimensions
                                  • RoPE encoding                     • 30% masking ratio               • 512 intermediate dim              • Session-specific decoders
                                                                                                              ↓
                                                                                                         📚 5-LAYER STACK
                                                                                                         • Residual connections
                                                                                                         • 256 hidden dimensions
```

**Notion-Compatible Flow Chart:**

```
INPUT LAYER:
├── 🧠 Neural Data (Spikes + LFP)
└── 🐭 Behavioral Data (Static + Dynamic)
    ↓
EMBEDDING LAYER:
├── 📊 Session Embeddings (Handle variable neurons)
├── 🔄 Modality Embeddings (5 types)
└── 📍 Position Embeddings (RoPE)
    ↓
MASKING STRATEGY:
├── 🎭 Mixed Masking (Random selection)
├── → Encoding (Mask behavior)
├── ← Decoding (Mask neural)
├── ↻ Self-Modality (Within modality)
└── ↔ Cross-Modality (Between modalities)
    ↓
TRANSFORMER CORE:
├── 🔍 Multi-Head Attention (8 heads, Flash optimized)
├── 📏 Layer Normalization (ScaleNorm option)
├── 🔢 MLP Layers (GELU, 512 dim)
└── 📚 5-Layer Stack (Residual, 256 hidden)
    ↓
OUTPUT LAYER:
├── ⚡ Spike Prediction (Poisson NLL)
├── 🎯 Behavior Prediction (MSE/CrossEntropy)
└── 🧩 Session Stitching (Variable dimensions)
```

---

## Data Flow Pipeline

### Complete Data Processing Pipeline

```mermaid
flowchart TD
    subgraph "Raw Data Sources"
        IBL["🗄️ IBL Database<br/>• 84 experimental sessions<br/>• Standardized mouse behavior<br/>• Multi-region recordings"]
        ONE["🔌 ONE API<br/>• Data access interface<br/>• Automatic caching<br/>• Session metadata"]
    end
    
    subgraph "Data Preprocessing"
        PrepData["📋 prepare_data.py<br/>• Download neural & behavioral data<br/>• Spike binning (20ms)<br/>• Behavioral alignment<br/>• Quality filtering"]
        
        SpikeProc["⚡ Spike Processing<br/>• Bin to 20ms windows<br/>• Filter responsive neurons<br/>• Sparse matrix storage"]
        
        BehProc["🎯 Behavior Processing<br/>• Wheel speed extraction<br/>• Whisker motion energy<br/>• Choice/block categorization"]
        
        LFPProc["🌊 LFP Processing (Optional)<br/>• Signal filtering & referencing<br/>• Power spectral density<br/>• Frequency band features"]
    end
    
    subgraph "Dataset Creation"
        CreateDS["📦 create_dataset.py<br/>• Convert to HuggingFace format<br/>• Train/val/test splits (70/10/20)<br/>• Memory-efficient storage"]
        
        DataSplit["📊 Data Partitioning<br/>• Session-wise splits<br/>• Trial-wise splits<br/>• Cross-session validation"]
        
        NPYSave["💾 NumPy Storage<br/>• Efficient .npy format<br/>• Batch-friendly structure<br/>• Fast loading during training"]
    end
    
    subgraph "Training Pipeline"
        DataLoader["📥 Data Loading<br/>• Length-grouped batching<br/>• Session-aware sampling<br/>• Multimodal alignment"]
        
        ModelTrain["🎓 Model Training<br/>• Mixed masking strategy<br/>• Multimodal fusion<br/>• Cross-session generalization"]
        
        Eval["📈 Evaluation<br/>• Encoding performance<br/>• Decoding accuracy<br/>• Cross-modal prediction"]
    end
    
    IBL --> ONE
    ONE --> PrepData
    PrepData --> SpikeProc
    PrepData --> BehProc
    PrepData --> LFPProc
    
    SpikeProc --> CreateDS
    BehProc --> CreateDS
    LFPProc --> CreateDS
    
    CreateDS --> DataSplit
    DataSplit --> NPYSave
    
    NPYSave --> DataLoader
    DataLoader --> ModelTrain
    ModelTrain --> Eval
    
    style IBL fill:#e1f5fe
    style PrepData fill:#f3e5f5
    style CreateDS fill:#e8f5e8
    style ModelTrain fill:#fff3e0
```

### Data Transformation Flow

```mermaid
graph LR
    subgraph "Raw Neural Data"
        A1["🔬 Raw Spikes<br/>• Continuous timestamps<br/>• Variable trial lengths<br/>• Multiple brain regions"]
        A2["📡 Raw LFP<br/>• 2500 Hz sampling<br/>• Multi-channel recordings<br/>• Continuous voltage"]
    end
    
    subgraph "Binned Neural Data"
        B1["📊 Binned Spikes<br/>• 20ms time bins<br/>• Poisson count data<br/>• (trials × time × neurons)"]
        B2["🌊 LFP Features<br/>• Power spectral density<br/>• 5 frequency bands<br/>• (trials × time × features)"]
    end
    
    subgraph "Behavioral Data"
        C1["🎯 Trial Events<br/>• Stimulus onset<br/>• Choice decisions<br/>• Reward delivery"]
        C2["📐 Continuous Signals<br/>• Wheel position<br/>• Whisker motion<br/>• Eye movements"]
    end
    
    subgraph "Aligned Multimodal Data"
        D1["🔄 Temporal Alignment<br/>• Common time base<br/>• Stimulus-locked<br/>• 100 time bins per trial"]
        D2["📋 Trial Structure<br/>• -500ms to +1500ms<br/>• Around stimulus onset<br/>• Standardized format"]
    end
    
    subgraph "Model-Ready Data"
        E1["🎭 Masked Inputs<br/>• Dynamic masking<br/>• Task-specific patterns<br/>• Training objectives"]
        E2["🎯 Prediction Targets<br/>• Modality-specific<br/>• Loss-function ready<br/>• Cross-modal tasks"]
    end
    
    A1 --> B1
    A2 --> B2
    C1 --> D1
    C2 --> D2
    
    B1 --> D1
    B2 --> D1
    D1 --> E1
    D2 --> E2
    
    style A1 fill:#ffebee
    style B1 fill:#e8f5e8
    style D1 fill:#e3f2fd
    style E1 fill:#fff8e1
```

---

## Multi-Task Masking Strategy

### Masking Scheme Overview

```mermaid
graph TD
    subgraph "Mixed Masking Strategy"
        A["🎲 Random Scheme Selection<br/>Per batch, randomly choose masking strategy"]
        
        B1["🧠→🎯 Encoding<br/>Mask: Behavioral data<br/>Predict: From neural signals<br/>Task: Neural → Behavior"]
        
        B2["🎯→🧠 Decoding<br/>Mask: Neural data<br/>Predict: From behavioral signals<br/>Task: Behavior → Neural"]
        
        B3["🔄 Self-Modality<br/>Mask: Within same modality<br/>Predict: Missing parts<br/>Task: Intra-modal completion"]
        
        B4["↔️ Cross-Modality<br/>Mask: Across modalities<br/>Predict: From other modalities<br/>Task: Inter-modal translation"]
        
        B5["🎭 Random Token<br/>Mask: Random temporal positions<br/>Predict: Masked time bins<br/>Task: Temporal prediction"]
    end
    
    subgraph "Masking Implementation"
        C1["⚙️ Temporal Masking<br/>• Consecutive time bins<br/>• Expandable windows<br/>• 30% masking ratio"]
        
        C2["🧩 Modality-Specific<br/>• Neural vs Behavioral<br/>• Static vs Dynamic<br/>• Session-aware"]
        
        C3["🎯 Target Generation<br/>• Ground truth alignment<br/>• Loss computation<br/>• Performance metrics"]
    end
    
    A --> B1
    A --> B2
    A --> B3
    A --> B4
    A --> B5
    
    B1 --> C1
    B2 --> C1
    B3 --> C2
    B4 --> C2
    B5 --> C3
    
    style A fill:#e1f5fe
    style B1 fill:#e8f5e8
    style B2 fill:#fff3e0
    style B3 fill:#f3e5f5
    style B4 fill:#fce4ec
    style B5 fill:#e0f2f1
```

### Task-Specific Masking Patterns

```mermaid
graph LR
    subgraph "Input Data Timeline"
        T1["T1"] --> T2["T2"] --> T3["T3"] --> T4["T4"] --> T5["T5"]
    end
    
    subgraph "Encoding Task"
        E1["🧠 Neural"] --> E2["🧠 Neural"] --> E3["❌ Masked"] --> E4["❌ Masked"] --> E5["🧠 Neural"]
        E11["❌ Masked"] --> E12["❌ Masked"] --> E13["🎯 Behavior"] --> E14["🎯 Behavior"] --> E15["❌ Masked"]
    end
    
    subgraph "Decoding Task"
        D1["❌ Masked"] --> D2["❌ Masked"] --> D3["🧠 Neural"] --> D4["🧠 Neural"] --> D5["❌ Masked"]
        D11["🎯 Behavior"] --> D12["🎯 Behavior"] --> D13["❌ Masked"] --> D14["❌ Masked"] --> D15["🎯 Behavior"]
    end
    
    subgraph "Self-Modality Task"
        S1["🧠 Neural"] --> S2["❌ Masked"] --> S3["❌ Masked"] --> S4["🧠 Neural"] --> S5["🧠 Neural"]
        S11["🎯 Behavior"] --> S12["🎯 Behavior"] --> S13["❌ Masked"] --> S14["❌ Masked"] --> S15["🎯 Behavior"]
    end
    
    T1 -.-> E1
    T2 -.-> E2
    T3 -.-> E3
    T4 -.-> E4
    T5 -.-> E5
    
    style E3 fill:#ffcdd2
    style E4 fill:#ffcdd2
    style E11 fill:#ffcdd2
    style E12 fill:#ffcdd2
    style E15 fill:#ffcdd2
```

---

## Session Stitching

### Variable Neuron Count Handling

```mermaid
graph TB
    subgraph "Problem: Variable Neuron Counts"
        A1["Session A<br/>150 neurons"]
        A2["Session B<br/>300 neurons"] 
        A3["Session C<br/>500 neurons"]
        A4["Session D<br/>120 neurons"]
    end
    
    subgraph "Solution: Stitching Architecture"
        B1["🧩 Stitch Encoder<br/>• Session-specific input layers<br/>• Normalize to hidden dim<br/>• Learnable transformations"]
        
        B2["🔄 Shared Transformer<br/>• Fixed hidden dimension<br/>• Cross-session learning<br/>• Common representations"]
        
        B3["🎯 Stitch Decoder<br/>• Session-specific output layers<br/>• Variable output dimensions<br/>• Flexible predictions"]
    end
    
    subgraph "Implementation Details"
        C1["📊 Session Mapping<br/>• EID → neuron count<br/>• Dynamic layer creation<br/>• Parameter dictionaries"]
        
        C2["⚙️ Forward Pass<br/>• Batch session grouping<br/>• Parallel processing<br/>• Efficient computation"]
        
        C3["🎓 Training Strategy<br/>• Session-aware batching<br/>• Gradient accumulation<br/>• Cross-session generalization"]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    
    B1 --> C1
    B2 --> C2
    B3 --> C3
    
    style B1 fill:#e3f2fd
    style B2 fill:#e8f5e8
    style B3 fill:#fff3e0
```

### Stitching Implementation Flow

```mermaid
sequenceDiagram
    participant Input as Input Data
    participant Encoder as Stitch Encoder
    participant Transformer as Shared Transformer
    participant Decoder as Stitch Decoder
    participant Output as Variable Outputs
    
    Note over Input: Sessions with different<br/>neuron counts (100-800)
    
    Input->>Encoder: Batch with session IDs
    Note over Encoder: Session-specific<br/>linear layers
    
    Encoder->>Encoder: Group by session EID
    Encoder->>Encoder: Apply session-specific<br/>transformation
    Encoder->>Transformer: Fixed 256-dim vectors
    
    Note over Transformer: Shared parameters<br/>across all sessions
    
    Transformer->>Transformer: Multi-head attention
    Transformer->>Transformer: Layer normalization
    Transformer->>Transformer: MLP processing
    Transformer->>Decoder: Contextualized features
    
    Decoder->>Decoder: Group by session EID
    Decoder->>Decoder: Apply session-specific<br/>output layers
    Decoder->>Output: Variable-sized predictions
    
    Note over Output: Session-appropriate<br/>output dimensions
```

---

## Training Modes

### Three Primary Training Configurations

```mermaid
graph TD
    subgraph "Training Mode Selection"
        A["🎛️ Model Mode Selection"]
        
        A --> B1["🔄 Multimodal (mm)<br/>• All modalities active<br/>• Mixed masking strategy<br/>• Joint encoding/decoding"]
        
        A --> B2["🧠→🎯 Encoding Only<br/>• Neural → Behavioral<br/>• Behavioral data targets<br/>• Decoding neural activity"]
        
        A --> B3["🎯→🧠 Decoding Only<br/>• Behavioral → Neural<br/>• Neural data targets<br/>• Encoding behavior to spikes"]
    end
    
    subgraph "Session Scaling"
        C1["👤 Single Session<br/>• One experimental session<br/>• Session-specific learning<br/>• Fine-grained analysis"]
        
        C2["👥 Multi-Session<br/>• 10-70 sessions<br/>• Cross-session learning<br/>• Generalization focus"]
        
        C3["🌐 Large Scale<br/>• 70+ sessions<br/>• Population-level patterns<br/>• Maximum generalization"]
    end
    
    subgraph "Training Infrastructure"
        D1["🖥️ Single GPU<br/>• Development/debugging<br/>• Small-scale experiments<br/>• Local training"]
        
        D2["🖥️🖥️ Multi-GPU<br/>• Distributed training<br/>• Large batch sizes<br/>• Faster convergence"]
        
        D3["🔍 Hyperparameter Search<br/>• Ray Tune integration<br/>• Automated optimization<br/>• Performance maximization"]
    end
    
    B1 --> C1
    B1 --> C2
    B1 --> C3
    
    C1 --> D1
    C2 --> D2
    C3 --> D3
    
    style B1 fill:#e3f2fd
    style B2 fill:#e8f5e8
    style B3 fill:#fff3e0
```

### Evaluation Framework

```mermaid
graph LR
    subgraph "Model Evaluation"
        A1["📊 Cross-Session Testing<br/>• Train on 74 sessions<br/>• Test on 10 held-out<br/>• Generalization assessment"]
        
        A2["🔄 Cross-Modal Tasks<br/>• Neural→Behavioral accuracy<br/>• Behavioral→Neural fidelity<br/>• Multimodal coherence"]
        
        A3["⏱️ Temporal Prediction<br/>• Future state prediction<br/>• Sequence modeling<br/>• Temporal dynamics"]
    end
    
    subgraph "Performance Metrics"
        B1["📈 Encoding Metrics<br/>• R² correlation<br/>• Prediction accuracy<br/>• Behavioral decoding"]
        
        B2["🎯 Decoding Metrics<br/>• Spike prediction error<br/>• Poisson likelihood<br/>• Neural fidelity"]
        
        B3["🔗 Multimodal Metrics<br/>• Cross-modal consistency<br/>• Joint representation quality<br/>• Information transfer"]
    end
    
    A1 --> B1
    A2 --> B2
    A3 --> B3
    
    style A1 fill:#e8f5e8
    style B1 fill:#fff3e0
```

---

## Data Specifications

### Data Dimensions and Formats

```mermaid
graph TB
    subgraph "Temporal Structure"
        T1["⏱️ Trial Duration: 2 seconds<br/>• -500ms to +1500ms<br/>• Stimulus-aligned<br/>• Standardized timing"]
        
        T2["📊 Time Binning: 20ms<br/>• 100 bins per trial<br/>• 50 Hz effective sampling<br/>• Temporal resolution"]
        
        T3["🔄 Trial Organization<br/>• 70% training trials<br/>• 10% validation trials<br/>• 20% testing trials"]
    end
    
    subgraph "Neural Data Specs"
        N1["⚡ Spike Data<br/>• Poisson count data<br/>• 100-800 neurons/session<br/>• Sparse matrix storage"]
        
        N2["🌊 LFP Data (Optional)<br/>• 5 frequency bands<br/>• Power spectral density<br/>• 300ms sliding windows"]
        
        N3["🧠 Brain Regions<br/>• Multi-area recordings<br/>• Region-specific analysis<br/>• Anatomical organization"]
    end
    
    subgraph "Behavioral Data Specs"
        B1["🎯 Static Variables<br/>• Choice: Binary (-1/1)<br/>• Block: Categorical (0.2/0.5/0.8)<br/>• Trial-level constants"]
        
        B2["📐 Dynamic Variables<br/>• Wheel speed: Continuous<br/>• Whisker motion: Continuous<br/>• Time-varying signals"]
        
        B3["📊 Data Quality<br/>• Missing value handling<br/>• Outlier detection<br/>• Signal preprocessing"]
    end
    
    T1 --> N1
    T2 --> N2
    T3 --> N3
    
    N1 --> B1
    N2 --> B2
    N3 --> B3
    
    style T1 fill:#e3f2fd
    style N1 fill:#e8f5e8
    style B1 fill:#fff3e0
```

### Memory and Storage Optimization

```mermaid
graph LR
    subgraph "Data Storage Strategy"
        A1["💾 Raw Storage<br/>• HuggingFace datasets<br/>• Compressed format<br/>• Metadata preservation"]
        
        A2["⚡ Training Storage<br/>• NumPy arrays<br/>• Memory-mapped files<br/>• Fast loading"]
        
        A3["🗜️ Compression<br/>• Sparse matrices<br/>• Efficient encoding<br/>• Reduced footprint"]
    end
    
    subgraph "Memory Management"
        B1["📥 Batch Loading<br/>• On-demand loading<br/>• Memory-efficient<br/>• Cache optimization"]
        
        B2["🔄 Data Streaming<br/>• Large dataset handling<br/>• Progressive loading<br/>• Resource management"]
        
        B3["⚙️ Processing Pipeline<br/>• Parallel processing<br/>• Batch optimization<br/>• GPU memory efficiency"]
    end
    
    A1 --> B1
    A2 --> B2
    A3 --> B3
    
    style A1 fill:#e1f5fe
    style B1 fill:#f3e5f5
```

---

## Key Innovations

### Revolutionary Contributions

```mermaid
mindmap
  root((NEDS Innovations))
    Multi-Task Masking
      Mixed Strategy
      Dynamic Selection
      Cross-Modal Learning
      Simultaneous Tasks
    Session Stitching
      Variable Neurons
      Cross-Session Learning
      Scalable Architecture
      Unified Framework
    Multimodal Fusion
      Neural + Behavioral
      LFP Integration
      Temporal Alignment
      Joint Representation
    Transformer Design
      RoPE Encoding
      Flash Attention
      ScaleNorm Option
      Efficient Processing
    Training Framework
      Ray Tune Integration
      Multi-GPU Support
      Cross-Session Validation
      Hyperparameter Search
```

### Technical Breakthroughs

```mermaid
graph TD
    subgraph "Architectural Innovations"
        A1["🧩 Session Stitching<br/>First framework to handle<br/>variable neuron counts<br/>across recording sessions"]
        
        A2["🎭 Multi-Task Masking<br/>Single model performs<br/>encoding AND decoding<br/>simultaneously"]
        
        A3["🔄 Multimodal Fusion<br/>Unified processing of<br/>neural and behavioral<br/>data streams"]
    end
    
    subgraph "Performance Advances"
        B1["📊 Cross-Session Generalization<br/>Models trained on some sessions<br/>generalize to unseen sessions<br/>without retraining"]
        
        B2["⚡ Efficient Processing<br/>Sparse matrix optimization<br/>Flash attention integration<br/>Memory-efficient training"]
        
        B3["🎯 Joint Learning<br/>Simultaneous optimization<br/>of multiple objectives<br/>Improved performance"]
    end
    
    subgraph "Scientific Impact"
        C1["🧠 Neuroscience Applications<br/>Brain-computer interfaces<br/>Neural prosthetics<br/>Cognitive modeling"]
        
        C2["🔬 Research Tools<br/>Standardized benchmarks<br/>Open-source framework<br/>Reproducible results"]
        
        C3["🚀 Future Directions<br/>Foundation models for neuroscience<br/>Multi-species adaptation<br/>Real-time applications"]
    end
    
    A1 --> B1 --> C1
    A2 --> B2 --> C2
    A3 --> B3 --> C3
    
    style A1 fill:#e3f2fd
    style B1 fill:#e8f5e8
    style C1 fill:#fff3e0
```

---

## Summary

NEDS represents a paradigm shift in computational neuroscience by:

1. **🔄 Unifying Encoding/Decoding**: Single model handles both neural→behavioral and behavioral→neural tasks
2. **🧩 Scalable Architecture**: Handles variable neuron counts across different recording sessions  
3. **🎭 Intelligent Masking**: Dynamic masking strategy enables multi-task learning
4. **⚡ Efficient Processing**: Optimized for large-scale neural data with sparse representations
5. **🌐 Cross-Session Learning**: Generalizes across different experimental sessions and subjects

This comprehensive framework enables unprecedented analysis of neural-behavioral relationships at scale, opening new possibilities for brain-computer interfaces, neural prosthetics, and fundamental neuroscience research.

---

*For detailed implementation examples and usage instructions, see the main [README.md](README.md)*
