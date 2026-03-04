# ResNet CIFAR-10 Implementation

This repository contains a PyTorch implementation of a ResNet architecture optimized for CIFAR-10 images (32x32). It demonstrates the usage of identity and projection shortcuts within residual blocks.

---

## Architecture Overview

The network consists of the following main components:

### Input Layer

* Convolution: `3x3` kernel, stride `1`, padding `1`, output channels `16`
* Batch Normalization
* ReLU Activation

### Stacks (Residual Blocks)

1. **Stack 1** (16 channels, 32x32)

   * `n` residual blocks
   * `subsample=False` → identity shortcuts

2. **Stack 2** (32 channels, 16x16)

   * First block (`stack2a`) → `subsample=True` → projection shortcut
   * Remaining `n-1` blocks (`stack2b`) → `subsample=False` → identity shortcuts

3. **Stack 3** (64 channels, 8x8)

   * First block (`stack3a`) → `subsample=True` → projection shortcut
   * Remaining `n-1` blocks (`stack3b`) → `subsample=False` → identity shortcuts

### Output Layer

* Global Average Pooling (AdaptiveAvgPool2d to 1x1)
* Fully Connected Layer → 10 classes
* LogSoftmax Activation

### Weight Initialization

* Fully connected layers use **He (Kaiming) initialization**
* Bias set to zero

---

## Forward Pass

1. Input `x` passes through the initial convolution, batch normalization, and ReLU.
2. Iterates through Stack 1 blocks (identity shortcuts).
3. Passes through Stack 2:

   * First block uses projection shortcut if `subsample=True`
   * Remaining blocks use identity shortcuts
4. Passes through Stack 3:

   * First block uses projection shortcut if `subsample=True`
   * Remaining blocks use identity shortcuts
5. Global average pooling → flatten → fully connected layer → log softmax output

---

## Shortcut Connections

### Identity Shortcut

* Used within a stack where input and output dimensions match
* Adds input directly to the residual output

### Projection Shortcut (or Modified Identity, Option A)

* Used when moving to a new stack where channels or spatial dimensions change
* Downsamples the input and optionally zero-pads extra channels
* No extra learnable parameters in Option A
* Ensures input and residual output dimensions match for addition

**Rule:** If shape changes → projection shortcut; if shape remains → identity shortcut.

---

## Tensor Shape Flow

| Stack  | Channels | Spatial Size | Shortcut Type            |
| ------ | -------- | ------------ | ------------------------ |
| Stack1 | 16       | 32x32        | Identity                 |
| Stack2 | 32       | 16x16        | Projection / first block |
| Stack2 | 32       | 16x16        | Identity                 |
| Stack3 | 64       | 8x8          | Projection / first block |
| Stack3 | 64       | 8x8          | Identity                 |
| Output | 64 → 10  | 1x1          | N/A                      |

---

## Notes

* `ModuleList` is used to manually iterate over blocks, giving flexibility for shortcut handling
* This implementation is designed for **CIFAR-10 images** and can be adapted to other datasets with minor changes
* The number of residual blocks per stack is controlled by parameter `n`

---

## Network Depth and Choice of `n`

In the original CIFAR ResNet described in the paper by He et al., the total network depth follows the formula:

**Depth = 6n + 2**

This comes from:

* 1 initial convolution layer
* 3 stacks
* Each stack contains `n` residual blocks
* Each residual block has 2 convolution layers → `3 × n × 2 = 6n`
* 1 final fully connected layer

So the total depth becomes:

`1 (conv) + 6n (residual conv layers) + 1 (fc) = 6n + 2`

This is why the commonly reported CIFAR ResNet models use:

| n | Depth |
| - | ----- |
| 3 | 20    |
| 5 | 32    |
| 7 | 44    |
| 9 | 56    |

These values are chosen specifically so that the total depth matches the reported model sizes (ResNet-20, ResNet-32, ResNet-44, ResNet-56).

If different values such as `n = 2, 4, 6, 8` are used, the architecture remains valid, but the resulting depth will be different and will not correspond to the original paper's naming convention.

---

## References

* [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385) by Kaiming He et al.
* [github]:    https://github.com/a-martyn/resnet