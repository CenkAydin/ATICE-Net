# ATICE-Net: Advanced Copy-Move Forgery Detection Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

ATICE-Net is a state-of-the-art deep learning framework for copy-move forgery detection, inspired by and improving upon LHCM-Net. This project provides a complete, modular implementation with advanced features for robust forgery detection.

## 🚀 Features

### Architecture Components

- **Lightweight CNN-based Encoder**: FasterNet-inspired efficient feature extraction
- **Multi-Resolution Feature Fusion**: Attention-based fusion of multi-scale features
- **Similarity Comparison Module**: LCSCC-style with dual-scale fusion and learnable positional offsets
- **Edge-Aware Attention**: Enhanced boundary detection for better segmentation
- **CRF Post-Processing**: Optional Conditional Random Field refinement
- **Adversarial Training**: Discriminator-based regularization for improved realism

### Training Features

- **Consistency Loss**: Multi-view consistency during training
- **Multi-Scale Supervision**: Deep supervision at multiple decoder levels
- **Comprehensive Metrics**: Accuracy, F1-score, IoU, Precision, Recall
- **TensorBoard Integration**: Real-time training visualization
- **Checkpoint Management**: Automatic best model saving and resume capability

### Evaluation & Visualization

- **Batch Evaluation**: Efficient testing on large datasets
- **Single Image Testing**: Quick inference on individual images
- **Comparison Grids**: Visual comparison of multiple predictions
- **Detailed Metrics**: Comprehensive evaluation reports
- **Result Visualization**: Original + prediction + ground truth overlays

## 📁 Project Structure

```
atice_net/
├── models/                 # Neural network architecture
│   ├── encoder.py         # Lightweight CNN encoder
│   ├── decoder.py         # Multi-scale decoder
│   ├── fusion.py          # Multi-resolution feature fusion
│   ├── similarity.py      # Similarity comparison module
│   ├── edge_attention.py  # Edge-aware attention
│   └── atice_net.py       # Main ATICE-Net model
├── losses/                # Loss functions
│   ├── consistency_loss.py # Multi-view consistency loss
│   ├── adversarial_loss.py # Adversarial training losses
│   └── total_loss.py      # Combined loss function
├── dataloaders/           # Data loading and preprocessing
│   └── casia_loader.py    # CASIA 2.0 dataset loader
├── utils/                 # Utility functions
│   ├── metrics.py         # Evaluation metrics
│   ├── visualization.py   # Result visualization
│   └── crf.py            # CRF post-processing
├── train.py              # Training script
├── test.py               # Testing script
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/CenkAydin/ATICE-Net.git
cd atice-net
```

2. **Create virtual environment**

```bash
python -m venv atice_env
source atice_env/bin/activate  # On Windows: atice_env\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify installation**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
```

## 📊 Dataset Preparation

### CASIA 2.0 Dataset

1. **Download CASIA 2.0**

   - Download from [CASIA website](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
   - Extract the dataset to your local directory

2. **Organize dataset structure**

```
data/CASIA2.0/
├── Au/          # Authentic images
│   ├── Au_ani_00001.jpg
│   ├── Au_ani_00002.jpg
│   └── ...
├── Tp/          # Tampered images
│   ├── Tp_D_NRN_S_N_ani00001_ani00002_001.jpg
│   ├── Tp_D_NRN_S_N_ani00001_ani00002_002.jpg
│   └── ...
└── Gt/          # Ground truth masks
    ├── Tp_D_NRN_S_N_ani00001_ani00002_001_gt.png
    ├── Tp_D_NRN_S_N_ani00001_ani00002_002_gt.png
    └── ...
```

3. **Update configuration**
   Edit `config.yaml` and set the correct data path:

```yaml
paths:
  data_dir: "./data/CASIA2.0"
```

## 🚀 Training

### Basic Training

```bash
python train.py --config config.yaml
```

### Resume Training

```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

### Training with Custom Seed

```bash
python train.py --config config.yaml --seed 123
```

### Training Configuration

Key parameters in `config.yaml`:

```yaml
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

model:
  encoder_channels: [64, 128, 256, 512]
  decoder_channels: [256, 128, 64, 32]
  similarity_dim: 128
  use_crf: true
  use_adversarial: true
```

### Training Outputs

- **Checkpoints**: Saved in `checkpoints/` directory
- **Logs**: TensorBoard logs in `logs/` directory
- **Results**: Training curves in `results/` directory

### Monitor Training

```bash
tensorboard --logdir logs
```

## 🧪 Testing & Evaluation

### Test on Full Dataset

```bash
python test.py --checkpoint checkpoints/best_model.pth --config config.yaml
```

### Test Single Image

```bash
python test.py --checkpoint checkpoints/best_model.pth --single_image path/to/image.jpg
```

### Generate Comparison Grid

```bash
python test.py --checkpoint checkpoints/best_model.pth --comparison_grid
```

### Evaluation Metrics

The model outputs comprehensive metrics:

- **Accuracy**: Overall pixel-wise accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union

## 📈 Example Results

### Training Progress

```
Epoch 1/100
Train Loss: 0.8234, Val Loss: 0.7891
Train F1: 0.7234, Val F1: 0.7456
Train IoU: 0.6234, Val IoU: 0.6456
```

### Evaluation Summary

```
==================================================
EVALUATION SUMMARY
==================================================
ACCURACY    : 0.9234 ± 0.0123
PRECISION   : 0.8956 ± 0.0234
RECALL      : 0.9123 ± 0.0189
F1          : 0.9034 ± 0.0156
IOU         : 0.8234 ± 0.0234
==================================================
```

### Visualization Examples

The model generates comprehensive visualizations:

- Original image
- Predicted forgery mask
- Overlay visualization
- Ground truth comparison (if available)

## 🔧 Configuration

### Model Architecture

```yaml
model:
  name: "ATICE-Net"
  encoder_channels: [64, 128, 256, 512] # Encoder feature dimensions
  decoder_channels: [256, 128, 64, 32] # Decoder feature dimensions
  similarity_dim: 128 # Similarity comparison dimension
  edge_attention_dim: 64 # Edge attention dimension
  use_crf: true # Enable CRF post-processing
  use_adversarial: true # Enable adversarial training
```

### Training Parameters

```yaml
training:
  batch_size: 8 # Training batch size
  num_epochs: 100 # Number of training epochs
  learning_rate: 0.001 # Learning rate
  weight_decay: 0.0001 # Weight decay
  scheduler_step_size: 30 # LR scheduler step size
  scheduler_gamma: 0.5 # LR scheduler gamma
```

### Loss Weights

```yaml
loss_weights:
  bce: 1.0 # Binary cross-entropy weight
  dice: 0.5 # Dice loss weight
  consistency: 0.3 # Consistency loss weight
  adversarial: 0.1 # Adversarial loss weight
  edge: 0.2 # Edge loss weight
```

## 🎯 Performance

### Model Statistics

- **Parameters**: ~3M trainable parameters
- **Memory Usage**: ~4GB GPU memory (batch size 8)
- **Training Time**: ~2-3 hours on RTX 3080 (100 epochs)
- **Inference Time**: ~50ms per image (256x256)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## 🙏 Acknowledgments

- **CASIA 2.0 Dataset**: Provided by the Institute of Automation, Chinese Academy of Sciences
- **LHCM-Net**: Original architecture that inspired this work
- **PyTorch Community**: For the excellent deep learning framework
- **Open Source Community**: For various tools and libraries used in this project

## 📞 Contact

- **Author**: [Mustafa Cenk Aydın]
- **Email**: [mustafacenk28@gmail.com.com]
- **GitHub**: [@CenkAydin](https://github.com/CenkAydin)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{atice_net_2025,
  title={ATICE-Net: Advanced Copy-Move Forgery Detection Network},
  author={Mustafa Cenk Aydın},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.
