# Quick Start Guide

This guide will help you get started with the Sparse 3D Shape Classifier.

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (for training and inference with MinkowskiEngine)
- At least 4GB of GPU memory recommended

## Installation

### 1. Install PyTorch

First, install PyTorch with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your configuration, or use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install MinkowskiEngine

MinkowskiEngine requires CUDA and needs to be compiled:

```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
```

If you encounter issues, see the [MinkowskiEngine installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation).

### 3. Install Other Dependencies

```bash
pip install numpy scikit-learn
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Demo (No GPU Required)

Run the demo to see shape generation:

```bash
python demo.py
```

Run basic tests:

```bash
python test_basic.py
```

### Training

Train with default settings (50 epochs, 100 samples/class):

```bash
python train.py
```

Train with custom settings:

```bash
python train.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --train-samples 200 \
    --val-samples 50
```

The model will be saved to `checkpoints/best_model.pth`.

### Inference

Evaluate on test samples:

```bash
python inference.py --mode evaluate --num-samples 20
```

Predict a single shape:

```bash
python inference.py --mode single --shape sphere
```

Try different shapes:

```bash
python inference.py --mode single --shape cube
python inference.py --mode single --shape cylinder
python inference.py --mode single --shape cone
python inference.py --mode single --shape pyramid
```

## Expected Results

With default training parameters (50 epochs):
- Training should complete in 5-10 minutes on a modern GPU
- Training accuracy: >95%
- Validation accuracy: >90%
- All 5 shape classes should be well-separated

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --batch-size 4
```

### MinkowskiEngine Installation Issues

1. Ensure CUDA is properly installed: `nvcc --version`
2. Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Install MinkowskiEngine matching your CUDA version
4. See [official documentation](https://github.com/NVIDIA/MinkowskiEngine#installation) for details

### CPU-Only Mode

The model and dataset modules require MinkowskiEngine, which needs CUDA. However, you can:
- Run the demo: `python demo.py` (no GPU needed)
- Test shape generation: `python test_basic.py` (no GPU needed)

For full training/inference, a CUDA GPU is required.

## Next Steps

1. Experiment with different network architectures in `model.py`
2. Try different voxel sizes with `--voxel-size`
3. Adjust the number of points per cloud with `--num-points`
4. Add more shape types by extending `generate_shapes.py`
5. Implement your own data augmentation strategies in `dataset.py`

## File Overview

- `generate_shapes.py` - Point cloud generation for each shape type
- `model.py` - Sparse 3D CNN architecture
- `dataset.py` - PyTorch dataset and data loading
- `train.py` - Training loop and logic
- `inference.py` - Model evaluation and prediction
- `demo.py` - Quick demonstration (no GPU needed)
- `test_basic.py` - Basic functionality tests (no GPU needed)

## Support

For issues or questions, please open an issue on GitHub.
