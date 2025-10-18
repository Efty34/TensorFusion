# TensorFusion: Multimodal Sentiment Analysis on CMU-MOSI Dataset

## Project Overview

This project implements **unimodal sentiment analysis** using the **Tensor Fusion Network (TFN)** architecture on the **CMU-MOSI (Multimodal Opinion Sentiment Intensity)** dataset. We systematically developed and optimized models for all three modalities (text, video, audio) across both binary and 5-class sentiment classification tasks.

### Key Objectives

- Understand and analyze the CMU-MOSI multimodal dataset structure
- Implement TFN-based unimodal baselines for each modality
- Achieve performance comparable to the original TFN paper benchmarks
- Explore optimization techniques for challenging multi-class classification

---

## Dataset: CMU-MOSI

### Overview

The **CMU-MOSI** (Carnegie Mellon University Multimodal Opinion Sentiment Intensity) dataset contains **2,199 opinion video clips** from **93 YouTube movie review videos**. Each clip is annotated with sentiment intensity scores ranging from **-3 (highly negative)** to **+3 (highly positive)**.

### Dataset Splits

| Split          | Samples | Percentage |
| -------------- | ------- | ---------- |
| **Train**      | 1,283   | 58.3%      |
| **Validation** | 214     | 9.7%       |
| **Test**       | 686     | 31.2%      |

### Modalities

#### 1. **Text Modality**

- **Feature Type**: GloVe word embeddings
- **Dimensions**: 300-dimensional vectors
- **Sequence Length**: 50 words
- **Shape**: `(samples, 50, 300)`
- **Description**: Pre-trained word embeddings capturing semantic relationships

#### 2. **Video/Vision Modality**

- **Feature Type**: Facet facial features
- **Dimensions**: 35 features
- **Sequence Length**: 50 frames
- **Shape**: `(samples, 50, 35)`
- **Features Include**:
  - Facial Action Units (AUs)
  - Facial landmarks
  - Head pose
  - Gaze direction
  - Basic emotions (anger, disgust, fear, joy, sadness, surprise)

#### 3. **Audio Modality**

- **Feature Type**: COVAREP acoustic features
- **Dimensions**: 74 features
- **Sequence Length**: 50 frames
- **Shape**: `(samples, 50, 74)`
- **Features Include**:
  - Pitch (F0) and voicing features
  - Glottal source parameters
  - Spectral features (MFCCs, spectral slope)
  - Prosodic features (energy, duration)

### Sentiment Class Distribution

#### 7-Class Original Distribution (Continuous → Discretized)

```
Sentiment Score    Train    Valid    Test     Total
─────────────────────────────────────────────────────
    -3             80       9        35       124  (5.6%)
    -2             178      35       99       312  (14.2%)
    -1             186      36       102      324  (14.7%)
     0             289      43       149      481  (21.9%)
    +1             236      39       133      408  (18.6%)
    +2             245      40       127      412  (18.7%)
    +3             69       12       41       122  (5.5%)
─────────────────────────────────────────────────────
Total              1283     214      686      2183
```

#### Binary Classification Distribution

**Threshold**: Positive (>0) vs Non-positive (≤0)

```
Split       Positive (1)    Non-Positive (0)    Ratio
──────────────────────────────────────────────────────
Train       550 (42.9%)     733 (57.1%)         0.43
Valid       91 (42.5%)      123 (57.5%)         0.42
Test        301 (43.9%)     385 (56.1%)         0.44
```

- **Class Imbalance**: Slight imbalance favoring negative sentiment
- **Strategy**: Weighted loss functions used during training

#### 5-Class Distribution

**Thresholds**: ≤-1.5, (-1.5,-0.5], (-0.5,0.5], (0.5,1.5], >1.5

```
Class   Label              Train    Valid   Test    Total
──────────────────────────────────────────────────────────
  0     Highly Negative    157      19      64      240  (11.0%)
  1     Negative           276      50      147     473  (21.7%)
  2     Neutral            391      62      180     633  (29.0%)
  3     Positive           301      52      171     524  (24.0%)
  4     Highly Positive    158      31      124     313  (14.3%)
```

- **Most Frequent**: Neutral (29.0%)
- **Least Frequent**: Highly Negative (11.0%)
- **Imbalance Ratio**: 2.6:1 (max/min)
- **Strategy**: Class weighting + label smoothing

---

## Methodology & Approach

### Phase 1: Data Exploration & Preprocessing

1. **Dataset Loading** (`0_Mosi_Dataset.ipynb`)

   - Loaded CMU-MOSI from MultiBench framework
   - Analyzed data structure and dimensions for all modalities
   - Visualized sentiment distribution across 7 original classes

2. **Data Validation**

   - Checked for NaN and Inf values in all modalities
   - Implemented cleaning function: `np.nan_to_num()`
   - Validated data integrity across train/valid/test splits

3. **Modality Export**
   - Exported each modality separately for efficient loading
   - Created `.pkl` files:
     - `mosi_text.pkl` (300-dim GloVe)
     - `mosi_vision.pkl` (35-dim Facet)
     - `mosi_audio.pkl` (74-dim COVAREP)

### Phase 2: Label Conversion

#### Binary Labels

```python
def convert_to_binary(labels):
    return (labels > 0).astype(np.float32).flatten()
```

- Positive: sentiment > 0 → class 1
- Non-positive: sentiment ≤ 0 → class 0

#### 5-Class Labels

```python
def convert_to_5class(labels):
    labels = labels.flatten()
    class_labels = np.zeros_like(labels, dtype=np.int64)
    class_labels[labels <= -1.5] = 0          # Highly Negative
    class_labels[(labels > -1.5) & (labels <= -0.5)] = 1   # Negative
    class_labels[(labels > -0.5) & (labels <= 0.5)] = 2    # Neutral
    class_labels[(labels > 0.5) & (labels <= 1.5)] = 3     # Positive
    class_labels[labels > 1.5] = 4            # Highly Positive
    return class_labels
```

### Phase 3: Model Architecture

All models follow the **Tensor Fusion Network (TFN)** architecture from:

> Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L. P. (2017). Tensor fusion network for multimodal sentiment analysis. _EMNLP 2017_.

#### Core TFN Components

1. **Temporal Encoder**: LSTM/GRU for sequence modeling
2. **Embedding Subnetwork (U_m)**: Modality-specific feature extraction
3. **Sentiment Inference Subnetwork (U_s)**: Classification head

---

## Implementation Details

### 1. Text Unimodal Models

#### Binary Classification (`1_text_unimodal.ipynb`)

**Architecture:**

```
Input: (batch, 50, 300) GloVe embeddings
  ↓
LSTM: (300 → 128) + LayerNorm
  ↓
Text Embedding Network (U_t):
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.2)
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Classifier (U_s):
  Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
  Linear(64 → 1) + Sigmoid
  ↓
Output: Binary prediction (0 or 1)
```

**Training Configuration:**

- **Loss**: BCELoss (Binary Cross-Entropy)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau (patience=3)
- **Batch Size**: 64
- **Epochs**: 100 (early stopping patience=15)
- **Regularization**: Dropout=0.2, LayerNorm, BatchNorm
- **Initialization**: Xavier Uniform

**Results:**
| Metric | Our Model | TFN Paper | Difference |
|--------|-----------|-----------|------------|
| Accuracy | ~74% | 74.0% | ±0% |
| F1 Score | ~75% | 75.0% | ±0% |

✅ **Status**: Achieved paper benchmarks

---

#### 5-Class Classification (`2_text_5class.ipynb`)

**Architecture:**

```
Input: (batch, 50, 300) GloVe embeddings
  ↓
LSTM: (300 → 128)
  ↓
Text Embedding Network:
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.3)
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Classifier:
  Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
  Linear(64 → 5)
  ↓
Output: 5-class logits
```

**Training Configuration:**

- **Loss**: CrossEntropyLoss (no class weights)
- **Optimizer**: Adam (lr=5e-4, weight_decay=0.001)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 64
- **Epochs**: 100
- **Regularization**: Dropout=0.3, Gradient Clipping=1.0

**Results:**
| Metric | Our Model | TFN Paper | Difference |
|--------|-----------|-----------|------------|
| Accuracy | ~35-38% | 38.5% | -0% to -3.5% |

✅ **Status**: Close to paper benchmark

---

### 2. Video/Vision Unimodal Models

#### Binary Classification (`4_video_unimodal.ipynb`)

**Architecture:**

```
Input: (batch, 50, 35) Facet features
  ↓
LSTM: (35 → 128) + LayerNorm
  ↓
Video Embedding Network (U_v):
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.2)
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Classifier (U_s):
  Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
  Linear(64 → 1) + Sigmoid
  ↓
Output: Binary prediction
```

**Training Configuration:**

- **Loss**: BCELoss
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 64
- **Epochs**: 100

**Results:**
| Metric | Our Model | TFN Paper | Difference |
|--------|-----------|-----------|------------|
| Accuracy | ~73% | 73.0% | ±0% |
| F1 Score | ~73% | 73.0% | ±0% |

✅ **Status**: Achieved paper benchmarks

---

#### 5-Class Classification (`5_video_5class.ipynb`)

**Architecture (Enhanced):**

```
Input: (batch, 50, 35) Facet features
  ↓
Bidirectional GRU (2 layers): (35 → 128) → 256 features
  ↓
Attention Mechanism:
  Query: Linear(256 → 128) + Tanh
  Score: Linear(128 → 1) + Softmax
  Weighted Sum → Attended features (256)
  ↓
Video Embedding Network:
  Linear(256 → 256) + BatchNorm + ReLU + Dropout(0.25)
  Linear(256 → 256) + BatchNorm + ReLU + Dropout(0.25)
  ↓
Classifier:
  Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.25)
  Linear(128 → 5)
  ↓
Output: 5-class logits
```

**Advanced Training Configuration:**

- **Loss**: CrossEntropyLoss (weighted + label_smoothing=0.1)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=15, T_mult=2)
- **Batch Size**: 16 (reduced for better generalization)
- **Epochs**: 200 (patience=25)
- **Data Augmentation**: Mixup (alpha=0.2)
- **Regularization**: Dropout=0.25, Gradient Clipping=0.5
- **Initialization**: Kaiming Normal
- **Random Seed**: 123 (changed from 42)

**Optimization Iterations:**

1. Baseline LSTM → Bidirectional LSTM
2. Added Attention mechanism
3. Increased model capacity (128→256)
4. Applied Mixup augmentation
5. Added label smoothing
6. Switched to CosineAnnealing scheduler
7. Reduced batch size (64→32→16)
8. Changed initialization to Kaiming

**Results:**
| Metric | Our Model | TFN Paper | Difference |
|--------|-----------|-----------|------------|
| Accuracy | ~19-20% | 30.4% | -10% to -11% |

⚠️ **Status**: Below paper benchmark (challenging with limited visual features)

---

### 3. Audio Unimodal Models

#### Binary Classification (`6_audio_unimodal.ipynb`)

**Architecture:**

```
Input: (batch, 50, 74) COVAREP features
  ↓
LSTM: (74 → 128) + LayerNorm
  ↓
Audio Embedding Network (U_a):
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.2)
  Linear(128 → 128) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Classifier (U_s):
  Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
  Linear(64 → 1) + Sigmoid
  ↓
Output: Binary prediction
```

**Training Configuration:**

- **Loss**: BCELoss
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32 (reduced for stability)
- **Epochs**: 100
- **Device**: CPU (to avoid CUDA errors)
- **Special**: NaN/Inf validation and cleaning

**Results:**
| Metric | Our Model | TFN Paper | Difference |
|--------|-----------|-----------|------------|
| Accuracy | ~65% | 65.0% | ±0% |
| F1 Score | ~64% | 64.0% | ±0% |

✅ **Status**: Achieved paper benchmarks

---

#### 5-Class Classification (`7_audio_5class.ipynb`)

**Architecture (Enhanced):**

```
Input: (batch, 50, 74) COVAREP features
  ↓
Bidirectional GRU (2 layers): (74 → 192) → 384 features
  ↓
Attention Mechanism:
  Query: Linear(384 → 128) + Tanh
  Score: Linear(128 → 1) + Softmax
  Weighted Sum → Attended features (384)
  ↓
Audio Embedding Network:
  Linear(384 → 256) + BatchNorm + ReLU + Dropout(0.25)
  Linear(256 → 192) + BatchNorm + ReLU + Dropout(0.25)
  ↓
Classifier:
  Linear(192 → 256) + BatchNorm + ReLU + Dropout(0.25)
  Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.125)
  Linear(128 → 5)
  ↓
Output: 5-class logits
```

**Enhanced Training Configuration:**

- **Loss**: CrossEntropyLoss (weighted + label_smoothing=0.15)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.005)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=15, T_mult=2)
- **Batch Size**: 32
- **Epochs**: 200 (patience=25)
- **Regularization**: Dropout=0.25, Gradient Clipping=0.5
- **Initialization**: Kaiming Normal
- **Device**: CPU

**Results:**
| Metric | Our Model | TFN Paper | Difference |
|--------|-----------|-----------|------------|
| Accuracy | ~17-20% | 27.5% | -7% to -10% |

⚠️ **Status**: Below paper benchmark (working on improvements)

---

## Optimization Techniques Applied

### Regularization Techniques

1. **Dropout**: 0.2-0.3 (varied by complexity)
2. **Batch Normalization**: After each linear layer
3. **Layer Normalization**: After LSTM/GRU outputs
4. **L2 Weight Decay**: 0.001-0.01
5. **Gradient Clipping**: max_norm=0.5-1.0
6. **Label Smoothing**: 0.1-0.15 (for 5-class)

### Architecture Enhancements

1. **Bidirectional RNNs**: Better temporal modeling
2. **Attention Mechanisms**: Adaptive feature aggregation
3. **Deeper Networks**: Increased capacity for complex patterns
4. **GRU vs LSTM**: Faster training, fewer parameters

### Training Strategies

1. **Learning Rate Scheduling**:
   - ReduceLROnPlateau (binary tasks)
   - CosineAnnealingWarmRestarts (5-class tasks)
2. **Early Stopping**: patience=15-25 epochs
3. **Class Weighting**: Handle imbalanced datasets
4. **Data Augmentation**: Mixup (alpha=0.2) for video
5. **Batch Size Tuning**: 16-64 based on task difficulty

### Initialization Methods

1. **Xavier Uniform**: Binary classification
2. **Kaiming Normal**: Enhanced 5-class models
3. **Orthogonal**: RNN hidden-to-hidden weights

### Debugging & Stability

1. **NaN/Inf Detection**: Batch-level validation
2. **Gradient Monitoring**: Clipping to prevent explosions
3. **CPU Fallback**: Avoid CUDA issues
4. **Random Seeds**: 42, 123 for reproducibility

---

## Project Structure

```
TensorFusion/
│
├── README.md                      # This comprehensive documentation
│
├── 0_Mosi_Dataset.ipynb          # Data exploration & export
│   ├── Load CMU-MOSI dataset
│   ├── Analyze 3 modalities
│   ├── Visualize sentiment distribution
│   └── Export modality-specific .pkl files
│
├── 1_text_unimodal.ipynb         # Text Binary Classification
│   ├── TFN Architecture: LSTM → Embedding → Classifier
│   ├── Target: 74% accuracy, 75% F1
│   └── Checkpoint: best_text_unimodal.pt
│
├── 2_text_5class.ipynb           # Text 5-Class Classification
│   ├── TFN Architecture with CrossEntropyLoss
│   ├── Target: 38.5% accuracy
│   └── Checkpoint: best_text_5class.pt
│
├── 4_video_unimodal.ipynb        # Video Binary Classification
│   ├── TFN Architecture for Facet features
│   ├── Target: 73% accuracy, 73% F1
│   └── Checkpoint: best_video_unimodal.pt
│
├── 5_video_5class.ipynb          # Video 5-Class Classification
│   ├── Enhanced: BiGRU + Attention + Mixup
│   ├── Target: 30.4% accuracy
│   └── Checkpoint: best_video_5class.pt
│
├── 6_audio_unimodal.ipynb        # Audio Binary Classification
│   ├── TFN Architecture for COVAREP features
│   ├── Target: 65% accuracy, 64% F1
│   └── Checkpoint: best_audio_unimodal.pt
│
├── 7_audio_5class.ipynb          # Audio 5-Class Classification
│   ├── Enhanced: BiGRU + Attention
│   ├── Target: 27.5% accuracy
│   └── Checkpoint: best_audio_5class.pt
│
├── MultiBench/                    # Framework directory
│   ├── data/
│   │   └── exported_modalities/
│   │       ├── mosi_text.pkl     # (1283, 50, 300) + labels
│   │       ├── mosi_vision.pkl   # (1283, 50, 35) + labels
│   │       └── mosi_audio.pkl    # (1283, 50, 74) + labels
│   │
│   ├── best_text_unimodal.pt     # Text binary checkpoint
│   ├── best_text_5class.pt       # Text 5-class checkpoint
│   ├── best_video_unimodal.pt    # Video binary checkpoint
│   ├── best_video_5class.pt      # Video 5-class checkpoint
│   ├── best_audio_unimodal.pt    # Audio binary checkpoint
│   └── best_audio_5class.pt      # Audio 5-class checkpoint
│
└── requirements.txt               # Python dependencies
```

---

## Results Summary

### Binary Classification Performance

| Modality  | Accuracy | F1 Score | Paper Acc | Paper F1 | Status      |
| --------- | -------- | -------- | --------- | -------- | ----------- |
| **Text**  | ~74%     | ~75%     | 74.0%     | 75.0%    | ✅ Achieved |
| **Video** | ~73%     | ~73%     | 73.0%     | 73.0%    | ✅ Achieved |
| **Audio** | ~65%     | ~64%     | 65.0%     | 64.0%    | ✅ Achieved |

### 5-Class Classification Performance

| Modality  | Accuracy | Paper Acc | Difference   | Status   |
| --------- | -------- | --------- | ------------ | -------- |
| **Text**  | ~35-38%  | 38.5%     | -0% to -3.5% | ✅ Close |
| **Video** | ~19-20%  | 30.4%     | -10% to -11% | ⚠️ Below |
| **Audio** | ~17-20%  | 27.5%     | -7% to -10%  | ⚠️ Below |

### Key Observations

1. **Binary Classification**: Successfully achieved paper benchmarks across all modalities
2. **Text 5-Class**: Closest to paper performance (most expressive features)
3. **Video 5-Class**: Most challenging (limited facial features, high class overlap)
4. **Audio 5-Class**: Moderate performance (acoustic features less discriminative for fine-grained sentiment)
5. **Class Imbalance**: 5-class tasks require sophisticated handling (weighting, smoothing)

---

## Technical Stack

### Core Libraries

- **PyTorch 1.x**: Deep learning framework
- **NumPy**: Numerical computations
- **Scikit-learn**: Metrics and evaluation
- **Matplotlib & Seaborn**: Visualization
- **Pickle**: Data serialization

### Key Dependencies

```
torch>=1.8.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

### Hardware Requirements

- **CPU**: Multi-core processor (used for stability)
- **RAM**: 8GB+ recommended
- **Storage**: ~500MB for dataset and checkpoints
- **GPU**: Optional (CUDA-compatible, but CPU mode implemented for stability)

---

## Challenges & Solutions

### Challenge 1: CUDA Memory Errors

**Problem**: `[WinError 1455] The paging file is too small` when training audio models on GPU

**Solution**:

- Switched to CPU training (`device='cpu'`)
- Reduced batch size (64→32)
- Implemented NaN/Inf validation before GPU operations
- Added gradient clipping (max_norm=0.5-1.0)

### Challenge 2: 5-Class Performance Gap

**Problem**: Significant gap between our results and paper benchmarks for 5-class tasks

**Solutions Attempted**:

- Enhanced architecture: LSTM→GRU, unidirectional→bidirectional
- Added attention mechanisms for better feature selection
- Applied data augmentation (Mixup for video)
- Increased model capacity (128→192→256 hidden units)
- Label smoothing (0.1-0.15)
- Class weighting based on frequency
- Different schedulers (ReduceLROnPlateau→CosineAnnealing)
- Multiple random seeds (42, 123)
- Longer training (100→200 epochs)

### Challenge 3: Class Imbalance

**Problem**: Neutral class dominates (29%), Highly Negative rare (11%)

**Solution**:

```python
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

### Challenge 4: Overfitting in Small Dataset

**Problem**: 1,283 training samples insufficient for deep models

**Solutions**:

- Multiple regularization layers (Dropout + BatchNorm + LayerNorm)
- Early stopping (patience=15-25)
- Weight decay (0.005-0.01)
- Smaller batch sizes for 5-class (16-32)
- Label smoothing

---

## Future Work

### Immediate Next Steps

1. **Multimodal Fusion**: Combine text + video + audio
   - Early Fusion (concatenate features)
   - Late Fusion (ensemble predictions)
   - Tensor Fusion (TFN's core contribution)
2. **Hyperparameter Optimization**:

   - Grid search for learning rates
   - Bayesian optimization for architecture search
   - AutoML for automated tuning

3. **Advanced Augmentation**:
   - Text: Back-translation, synonym replacement
   - Video: Temporal jittering, frame dropout
   - Audio: SpecAugment, time masking

### Long-term Goals

1. **Attention Fusion Networks**: Learn modality importance
2. **Transformer-based Models**: Replace LSTM/GRU with Transformers
3. **Pre-trained Embeddings**: BERT for text, ResNet for video
4. **Cross-modal Learning**: Transfer knowledge between modalities
5. **Explainability**: Visualize attention weights and feature importance

---

## References

### Primary Paper

```
Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L. P. (2017).
Tensor Fusion Network for Multimodal Sentiment Analysis.
In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017).
```

### Dataset

```
Zadeh, A., Zellers, R., Pincus, E., & Morency, L. P. (2016).
MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos.
arXiv preprint arXiv:1606.06259.
```

### Framework

```
MultiBench: Multiscale Benchmarks for Multimodal Representation Learning
https://github.com/pliang279/MultiBench
```

---

## Usage Instructions

### 1. Environment Setup

```bash
# Install dependencies
pip install torch numpy scikit-learn matplotlib seaborn

# Navigate to project directory
cd h:\4-1_Labs\ML\Codes\TensorFusion
```

### 2. Data Preparation

```bash
# Run data exploration notebook
jupyter notebook 0_Mosi_Dataset.ipynb

# Execute all cells to:
# - Load CMU-MOSI dataset
# - Visualize distributions
# - Export modality-specific files
```

### 3. Training Models

#### Binary Classification

```bash
# Text binary (74% accuracy target)
jupyter notebook 1_text_unimodal.ipynb

# Video binary (73% accuracy target)
jupyter notebook 4_video_unimodal.ipynb

# Audio binary (65% accuracy target)
jupyter notebook 6_audio_unimodal.ipynb
```

#### 5-Class Classification

```bash
# Text 5-class (38.5% accuracy target)
jupyter notebook 2_text_5class.ipynb

# Video 5-class (30.4% accuracy target)
jupyter notebook 5_video_5class.ipynb

# Audio 5-class (27.5% accuracy target)
jupyter notebook 7_audio_5class.ipynb
```

### 4. Checkpoint Loading

```python
import torch

# Load trained model
checkpoint = torch.load('MultiBench/best_text_unimodal.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
```

---

## Project Timeline

### Week 1: Data Understanding

- Loaded CMU-MOSI dataset
- Analyzed 3 modalities and 2,199 samples
- Visualized sentiment distributions
- Exported modality-specific files

### Week 2: Text Modality

- Implemented text binary classification (✅ 74% accuracy)
- Implemented text 5-class classification (✅ 38% accuracy)
- Established baseline TFN architecture

### Week 3: Video Modality

- Implemented video binary classification (✅ 73% accuracy)
- Implemented video 5-class classification (⚠️ 20% vs 30.4% target)
- Applied multiple optimization strategies

### Week 4: Audio Modality

- Resolved CUDA/memory issues (switched to CPU)
- Implemented audio binary classification (✅ 65% accuracy)
- Implemented audio 5-class classification (⚠️ 17-20% vs 27.5% target)
- Enhanced architecture with BiGRU + Attention

---

## Acknowledgments

- **CMU Multimodal SDK**: For providing the MOSI dataset
- **MultiBench Framework**: For dataset loading utilities
- **TFN Authors**: For the foundational architecture
- **PyTorch Community**: For excellent documentation and support

---

## License

This project is for educational purposes as part of a Machine Learning lab course.

---

## Contact & Contribution

For questions, improvements, or collaboration:

- Review the Jupyter notebooks for detailed implementation
- Check checkpoint files for trained model weights
- Examine training curves for performance insights

**Last Updated**: October 18, 2025

---

## Appendix

### A. Model Parameter Counts

| Model         | Parameters | Batch Size | Training Time (CPU) |
| ------------- | ---------- | ---------- | ------------------- |
| Text Binary   | ~170K      | 64         | ~5 min/epoch        |
| Text 5-Class  | ~170K      | 64         | ~5 min/epoch        |
| Video Binary  | ~150K      | 64         | ~3 min/epoch        |
| Video 5-Class | ~450K      | 16         | ~8 min/epoch        |
| Audio Binary  | ~180K      | 32         | ~4 min/epoch        |
| Audio 5-Class | ~520K      | 32         | ~10 min/epoch       |

### B. Hyperparameter Summary

| Task          | LR   | Weight Decay | Dropout | Batch Size | Scheduler       |
| ------------- | ---- | ------------ | ------- | ---------- | --------------- |
| Text Binary   | 1e-3 | 0.01         | 0.2     | 64         | ReduceLR        |
| Text 5-Class  | 5e-4 | 0.001        | 0.3     | 64         | ReduceLR        |
| Video Binary  | 1e-3 | 0.01         | 0.2     | 64         | ReduceLR        |
| Video 5-Class | 5e-4 | 0.01         | 0.25    | 16         | CosineAnnealing |
| Audio Binary  | 1e-3 | 0.01         | 0.2     | 32         | ReduceLR        |
| Audio 5-Class | 3e-4 | 0.005        | 0.25    | 32         | CosineAnnealing |

### C. Loss Functions

- **Binary**: `BCELoss()` - Binary Cross-Entropy
- **5-Class**: `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1-0.15)`

### D. Evaluation Metrics

- **Binary**: Accuracy, F1 Score, Precision, Recall, Confusion Matrix
- **5-Class**: Accuracy, Weighted F1, Per-Class Accuracy, Classification Report

---

