# SparseShapeClassifier

A 3D shape classifier that uses sparse 3D convolutions to classify point clouds into 5 geometric shapes: **sphere**, **cube**, **cylinder**, **cone**, and **pyramid**.

## Features

- **Sparse 3D Convolutions**: Uses MinkowskiEngine for efficient sparse convolution operations on 3D point clouds
- **5 Shape Classes**: Classifies sphere, cube, cylinder, cone, and pyramid
- **Synthetic Data Generation**: Generates realistic point clouds for each shape type
- **Data Augmentation**: Includes rotation, scaling, and jitter augmentation
- **Easy Training & Inference**: Simple command-line interface for training and evaluation

## Architecture

The model uses a sparse 3D convolutional neural network with the following structure:
- 4 sparse convolutional blocks with batch normalization and ReLU activation
- Progressive downsampling with strides
- Global pooling to aggregate spatial features
- Fully connected layers for classification

## Installation

### Requirements

- Python 3.7+
- PyTorch 2.0+
- MinkowskiEngine 0.5.4+
- NumPy
- scikit-learn

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Chainplain/SparseShapCalssifier.git
cd SparseShapCalssifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: MinkowskiEngine requires a CUDA-enabled GPU and may need manual installation. See [MinkowskiEngine installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation) for details.

## Usage

### Training

Train the model with default parameters:
```bash
python train.py
```

Customize training parameters:
```bash
python train.py --epochs 100 --batch-size 16 --lr 0.001 --train-samples 200 --device cuda
```

Training parameters:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.001)
- `--train-samples`: Training samples per class (default: 100)
- `--val-samples`: Validation samples per class (default: 20)
- `--num-points`: Points per point cloud (default: 1000)
- `--voxel-size`: Voxel size for quantization (default: 0.05)
- `--checkpoint-dir`: Directory to save checkpoints (default: 'checkpoints')
- `--device`: Device to use, 'cuda' or 'cpu' (default: 'cuda')

The trained model will be saved to `checkpoints/best_model.pth`.

### Inference

Evaluate the model on test samples:
```bash
python inference.py --mode evaluate --num-samples 20
```

Predict a single shape:
```bash
python inference.py --mode single --shape sphere
```

Inference parameters:
- `--checkpoint`: Path to model checkpoint (default: 'checkpoints/best_model.pth')
- `--mode`: Inference mode, 'evaluate' or 'single' (default: 'evaluate')
- `--shape`: Shape type for single prediction (sphere, cube, cylinder, cone, pyramid)
- `--num-samples`: Number of test samples per class for evaluation (default: 10)
- `--num-points`: Points per point cloud (default: 1000)
- `--device`: Device to use (default: 'cuda')

### Generate Shapes

Test shape generation:
```bash
python generate_shapes.py
```

This will create visualizations of all 5 shape types.

## Project Structure

```
SparseShapCalssifier/
├── generate_shapes.py    # Point cloud generation for each shape
├── model.py             # Sparse 3D CNN model definition
├── dataset.py           # Dataset and data loader implementation
├── train.py             # Training script
├── inference.py         # Inference and evaluation script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Model Details

### Shape Generation

Each shape is generated as a point cloud with configurable number of points:
- **Sphere**: Points distributed on sphere surface using Fibonacci sphere algorithm
- **Cube**: Points sampled uniformly on 6 faces
- **Cylinder**: Points on cylindrical surface and circular caps
- **Cone**: Points on conical surface and circular base
- **Pyramid**: Points on square base and 4 triangular faces

### Data Augmentation

Training includes:
- Random rotation around Y-axis
- Random scaling (0.8-1.2x)
- Random jitter (Gaussian noise)

### Network Architecture

```
Input: Sparse Point Cloud
  ↓
Conv3D (32) + BN + ReLU
  ↓
Conv3D (64, stride=2) + BN + ReLU
  ↓
Conv3D (128, stride=2) + BN + ReLU
  ↓
Conv3D (256, stride=2) + BN + ReLU
  ↓
Global Pooling
  ↓
FC (128) + ReLU + Dropout(0.5)
  ↓
FC (5 classes)
```

## Performance

With default parameters (50 epochs, 100 samples/class), the model typically achieves:
- Training accuracy: >95%
- Validation accuracy: >90%

The model is lightweight and can be trained on a single GPU in a few minutes.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.