# CNN Project: MNIST and CIFAR-10 Classification

This project demonstrates Convolutional Neural Network (CNN) implementations for image classification on two popular datasets: CIFAR-10 (color images of objects).

## Project Structure

```
CNN project/
├── CIFAR-10/
│   ├── main.ipynb            # CIFAR-10 classification notebook
│   ├── cifar_net.pth         # Saved model weights (if available)
│   ├── data/
│   │   └── cifar-10-batches-py/  # CIFAR-10 dataset files
│   └── results/
│       ├── model.pth         # Trained model weights
│       └── optimizer.pth     # Optimizer state
└── README.md                 # This file
```

## Datasets

### MNIST
- 70,000 grayscale images of handwritten digits (0-9)
- 60,000 training images, 10,000 test images
- Image size: 28x28 pixels

### CIFAR-10
- 60,000 color images in 10 classes
- 50,000 training images, 10,000 test images
- Image size: 32x32 pixels
- Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## CNN Implementations

### Basic CNN (Net Class)
A simple convolutional neural network with:
- 2 convolutional layers (3→6→16 channels)
- Max pooling after each conv layer
- 3 fully connected layers (400→120→84→10)

### BetterCNN Class
An improved convolutional neural network with:
- 3 convolutional blocks with increasing channels (64→128→256)
- Batch Normalization after each convolution
- Dropout regularization (Dropout2d in conv blocks, Dropout in classifier)
- Adaptive Average Pooling for size-agnostic classification
- Deeper architecture with more parameters

## Why BetterCNN is Better than Basic CNN

The BetterCNN achieves significantly higher accuracy (81%) compared to the basic CNN (61%) due to several architectural improvements:

### 1. **Deeper Architecture**
- **Basic CNN**: 2 conv layers
- **BetterCNN**: 6 conv layers organized in 3 blocks
- **Impact**: Deeper networks can learn hierarchical features, from simple edges to complex object parts

### 2. **Batch Normalization**
- **Basic CNN**: No batch normalization
- **BetterCNN**: BatchNorm2d after each conv layer
- **Impact**: 
  - Stabilizes training by normalizing activations
  - Allows higher learning rates
  - Reduces internal covariate shift
  - Acts as regularization

### 3. **Dropout Regularization**
- **Basic CNN**: No dropout
- **BetterCNN**: Dropout2d (25%) in conv blocks + Dropout (50%) in classifier
- **Impact**:
  - Prevents overfitting by randomly dropping features/channels
  - Improves generalization to unseen data

### 4. **More Feature Maps**
- **Basic CNN**: 6 → 16 channels
- **BetterCNN**: 64 → 128 → 256 channels
- **Impact**: More channels allow learning richer representations

### 5. **Adaptive Pooling**
- **Basic CNN**: Manual flattening after fixed-size pooling
- **BetterCNN**: AdaptiveAvgPool2d for size-agnostic processing
- **Impact**: Works with different input sizes without recalculating dimensions

### 6. **Better Training Practices**
- Larger batch size (72 vs 4)
- Appropriate learning rate (0.05 vs 0.03)
- More training epochs

## Accuracy Comparison

| Model      | Test Accuracy |
|------------|---------------|
| Basic CNN  | 61%          |
| BetterCNN  | 81%          |

The 20% improvement demonstrates the effectiveness of modern CNN techniques like batch normalization, dropout, and deeper architectures.

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### CIFAR-10 Classification

1. Open `CIFAR-10/main.ipynb` in Jupyter
2. Run cells sequentially to:
   - Load and preprocess CIFAR-10 data
   - Define the BetterCNN model
   - Train the model
   - Evaluate on test set


Training progress is printed every 100 batches showing average loss.

## Model Evaluation

After training, the model achieves ~81% accuracy on the CIFAR-10 test set. The notebook includes:
- Individual image predictions
- Overall test set accuracy calculation
- Model saving/loading functionality

## Future Improvements

- Implement data augmentation
- Use learning rate scheduling
- Try different optimizers (Adam, AdamW)
- Add more regularization techniques
- Experiment with different architectures (ResNet, EfficientNet)