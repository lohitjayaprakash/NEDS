# NEDS: Neural Encoding and Decoding at Scale - Technical Guide

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yzhang511/NEDS.git
cd NEDS

# Create conda environment
conda env create -f env.yaml
conda activate neds

# Install package
pip install -e .
```

### Basic Usage
```bash
# Download single session data
sbatch script/prepare_data.sh 1 EID_HERE

# Create training dataset
sbatch script/create_dataset.sh 1 EID_HERE

# Train multimodal model
sbatch script/train.sh 1 EID_HERE train mm 0 0.1 False random

# Evaluate model
sbatch script/eval.sh 1 EID_HERE train mm 0.1 random False
```

---

## Project Structure

```
NEDS/
├── 📁 src/                     # Source code
│   ├── 📄 prepare_data.py      # Data download & preprocessing
│   ├── 📄 create_dataset.py    # Dataset creation & formatting
│   ├── 📄 train.py            # Model training
│   ├── 📄 eval.py             # Model evaluation
│   ├── 📄 finetune.py         # Fine-tuning pre-trained models
│   ├── 📁 multi_modal/        # Core model architecture
│   │   ├── 📄 mm.py           # Main MultiModal class
│   │   ├── 📄 encoder_embeddings.py  # Embedding layers
│   │   └── 📄 mm_utils.py     # Utility functions
│   ├── 📁 models/             # Model components
│   │   ├── 📄 masker.py       # Masking strategies
│   │   ├── 📄 stitcher.py     # Session stitching
│   │   └── 📄 model_output.py # Output handling
│   ├── 📁 loader/             # Data loading
│   ├── 📁 trainer/            # Training logic
│   ├── 📁 utils/              # Utilities
│   └── 📁 configs/            # Configuration files
├── 📁 script/                  # Batch job scripts
├── 📁 data/                    # Session IDs
│   ├── 📄 train_eids.txt      # 74 training sessions
│   ├── 📄 test_eids.txt       # 10 test sessions
│   └── 📄 eids.txt            # All 84 sessions
├── 📄 env.yaml                # Conda environment
├── 📄 README.md               # Basic documentation
└── 📄 ARCHITECTURE.md         # Detailed architecture docs
```

---

## Data Pipeline

### 1. Data Download & Preprocessing

#### Single Session
```bash
# Download and preprocess one experimental session
python src/prepare_data.py --n_sessions 1 \
                          --eid "754b74d5-7a06-4004-ae0c-72a10b6ed2e6" \
                          --base_path ./data/ \
                          --use_lfp  # Optional: include LFP data
```

#### Multiple Sessions
```bash
# Download first 10 sessions from training set
python src/prepare_data.py --n_sessions 10 \
                          --base_path ./data/
```

**What this does:**
- Downloads neural spike data from IBL database
- Downloads behavioral data (wheel, whisker, choice, block)
- Optionally processes LFP data for frequency features
- Bins spike data to 20ms intervals
- Aligns all data to stimulus onset
- Filters low-activity neurons
- Saves as HuggingFace datasets

### 2. Dataset Creation

```bash
# Create training-ready datasets
python src/create_dataset.py --eid "your_eid_here" \
                             --num_sessions 1 \
                             --model_mode mm \
                             --data_path ./data/ \
                             --base_path ./
```

**Output Format:**
```
data/
├── ibl_mm_1/              # Single session dataset
│   ├── train/             # Training trials (70%)
│   ├── val/               # Validation trials (10%)
│   └── test/              # Test trials (20%)
└── ibl_mm_10/             # Multi-session dataset
    ├── train/
    ├── val/
    └── test/
```

---

## Model Training

### Training Modes

#### 1. Multimodal Training (Recommended)
```bash
# Train model on all modalities with mixed masking
python src/train.py --eid "your_eid" \
                   --num_sessions 10 \
                   --model_mode mm \
                   --mixed_training \
                   --mask_ratio 0.3 \
                   --base_path ./ \
                   --data_path ./data/
```

#### 2. Encoding-Only Training
```bash
# Train neural → behavioral prediction only
python src/train.py --eid "your_eid" \
                   --model_mode encoding \
                   --enc_task_var "wheel,whisker,choice,block"
```

#### 3. Decoding-Only Training  
```bash
# Train behavioral → neural prediction only
python src/train.py --eid "your_eid" \
                   --model_mode decoding \
                   --enc_task_var all
```

### Scaling Configurations

The model automatically selects architecture based on session count:

| Sessions | Config File | Description |
|----------|-------------|-------------|
| 1 | `mm_single_session.yaml` | Single session optimization |
| 2-70 | `mm_medium_size.yaml` | Multi-session learning |
| 70+ | `mm_large_size.yaml` | Large-scale training |

### Multi-GPU Training

```bash
# Train on multiple GPUs for large datasets
sbatch script/train_multi_gpu.sh 70 none mm 0 0.1 all
```

### Hyperparameter Search

```bash
# Automated hyperparameter optimization with Ray Tune
python src/train.py --search \
                   --num_tune_sample 30 \
                   --eid "your_eid"
```

**Tuned Parameters:**
- Learning rate: [1e-5, 1e-3]
- Weight decay: [1e-6, 1e-3]  
- Hidden size: [128, 256, 512]
- Number of layers: [3, 5, 8]
- Masking ratio: [0.1, 0.5]

---

## Model Configuration

### Core Architecture Settings

```yaml
# src/configs/multi_modal/mm.yaml
encoder:
  transformer:
    n_layers: 5           # Transformer depth
    hidden_size: 256      # Hidden dimensions
    n_heads: 8           # Attention heads
    dropout: 0.4         # Dropout rate
    use_rope: true       # Rotary position encoding
    
  embedder:
    max_F: 100           # Max time steps
    n_modality: 5        # Number of data modalities
    pos: true            # Use positional embeddings
    
masker:
  mode: temporal         # Masking strategy
  ratio: 0.3            # Fraction to mask
  expand_prob: 0.0      # Mask expansion probability
```

### Masking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `temporal` | Mask consecutive time bins | General training |
| `random` | Random token masking | Robustness training |
| `causal` | Forward-only masking | Real-time applications |
| `neuron` | Mask entire neurons | Cross-neuron learning |
| `inter-region` | Mask brain regions | Cross-region analysis |

---

## Data Specifications

### Neural Data

```python
# Spike data format
spike_data.shape  # (n_trials, n_timepoints, n_neurons)
# n_trials: ~200-1000 per session
# n_timepoints: 100 (20ms bins × 2 seconds)  
# n_neurons: 100-800 per session

# LFP data format (optional)
lfp_data.shape    # (n_trials, n_timepoints, n_features)  
# n_features: 5 frequency bands × n_channels
```

### Behavioral Data

```python
# Static variables (trial-level)
choice_data.shape   # (n_trials,) - Binary: -1 or 1
block_data.shape    # (n_trials,) - Categorical: 0.2, 0.5, 0.8

# Dynamic variables (time-varying)
wheel_data.shape    # (n_trials, n_timepoints) - Continuous
whisker_data.shape  # (n_trials, n_timepoints) - Continuous
```

### Session Information

```python
# Each session contains:
session_info = {
    'eid': 'unique_session_id',
    'n_neurons': 150,  # Variable across sessions
    'n_trials': 500,
    'brain_regions': ['VISp', 'CA1', 'DG', ...],
    'lab': 'laboratory_name',
    'subject': 'mouse_id'
}
```

---

## Evaluation & Analysis

### Cross-Session Evaluation

```bash
# Train on 74 sessions, test on 10 held-out sessions
python src/eval.py --num_sessions 74 \
                  --eid "test_session_eid" \
                  --finetune \  # Use pre-trained model
                  --model_mode mm
```

### Performance Metrics

#### Encoding Tasks (Neural → Behavioral)
- **R² Correlation**: Behavioral prediction accuracy
- **Classification Accuracy**: Choice/block prediction
- **MSE**: Continuous variable prediction (wheel, whisker)

#### Decoding Tasks (Behavioral → Neural)  
- **Poisson Log-Likelihood**: Spike prediction quality
- **Pearson Correlation**: Neural activity correlation
- **Cross-Validated R²**: Generalization performance

#### Multimodal Tasks
- **Cross-Modal Consistency**: Inter-modality agreement
- **Information Transfer**: Mutual information metrics
- **Joint Representation Quality**: Shared embedding analysis

### Custom Evaluation

```python
# Example evaluation script
from src.eval import evaluate_model

results = evaluate_model(
    model_path="./model_best.pt",
    test_data="./data/test/",
    tasks=["encoding", "decoding", "cross_modal"],
    metrics=["r2", "accuracy", "likelihood"]
)
```

---

## Advanced Usage

### Fine-tuning Pre-trained Models

```bash
# Fine-tune a pre-trained 10-session model on new session
python src/finetune.py --num_sessions 10 \
                      --eid "new_session_eid" \
                      --model_mode mm \
                      --base_model_path "./pretrained_model.pt"
```

### Custom Masking Strategies

```python
# Define custom masking in masker.py
class CustomMasker(Masker):
    def forward(self, spikes, neuron_regions=None, mode="custom"):
        # Implement your masking logic
        mask = your_custom_masking_function(spikes)
        return spikes, mask
```

### Adding New Modalities

```python
# Extend the model for new data types
new_modality_config = {
    "eeg": "continuous_signal",  # Add EEG data
    "calcium": "neural_signal",  # Add calcium imaging
    "behavior_video": "image_sequence"  # Add video data
}
```

---

## Performance Optimization

### Memory Management

```python
# For large datasets, use these settings:
config = {
    "batch_size": 16,  # Reduce if OOM
    "max_time_length": 100,  # Clip long sequences
    "max_space_length": 500,  # Limit neuron count
    "use_sparse_storage": True,  # Enable sparse matrices
}
```

### Training Acceleration

```bash
# Use multiple workers for data loading
--n_workers 4

# Enable mixed precision training
--use_amp

# Use compiled model (PyTorch 2.0+)
--compile_model
```

### Monitoring Training

```python
# Enable wandb logging
python src/train.py --wandb \
                   --project_name "neds_experiments" \
                   --run_name "multimodal_training"
```

---

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce batch size and sequence length
--batch_size 8
--max_time_length 50
--max_space_length 300
```

#### Data Loading Errors
```bash
# Check data paths and permissions
ls -la /path/to/data/
# Verify EID exists in dataset
grep "your_eid" data/train_eids.txt
```

#### Model Convergence Issues
```bash
# Try different learning rates
--learning_rate 1e-4
# Adjust masking ratio
--mask_ratio 0.2
# Enable gradient clipping
--grad_clip_norm 1.0
```

### Debug Mode

```bash
# Run with debugging enabled
python src/train.py --debug \
                   --log_level DEBUG \
                   --save_every 100
```

---

## Citation

```bibtex
@article{zhang2025neural,
  title={Neural Encoding and Decoding at Scale},
  author={Zhang, Yizi and Wang, Yanchen and Azabou, Mehdi and Andre, Alexandre and Wang, Zixuan and Lyu, Hanrui and Laboratory, The International Brain and Dyer, Eva and Paninski, Liam and Hurwitz, Cole},
  journal={arXiv preprint arXiv:2504.08201},
  year={2025}
}
```

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)  
5. Create Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yzhang511/NEDS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yzhang511/NEDS/discussions)
- **Documentation**: [Architecture Guide](ARCHITECTURE.md)

For detailed technical implementation, see [ARCHITECTURE.md](ARCHITECTURE.md).
