# YOLO Object Detection Notebook — Full Explanation

---

## 📊 Results Summary (Read This First)

Before diving into the code, here's what your two final metrics mean and whether they're good:

### Mean Average Precision (mAP) @ IoU=0.5: **0.0009**

**What it means:**
mAP measures how well your model detects objects. It combines both *precision* (are your detections correct?) and *recall* (did you find all the objects?) across all 80 COCO classes. The `@ IoU=0.5` part means a detection only counts as correct if the predicted bounding box overlaps the ground truth box by at least 50%.

A score of **0.0009** means the model is barely detecting anything correctly — essentially 0.09% accuracy. This is **very poor**.

**Is this good?** ❌ No. For context:
- A **randomly initialized model** (untrained) would score around 0.
- A **well-trained YOLOv1** on COCO scores around **0.30–0.45** (30–45%).
- Production models like YOLOv8 score **0.50+**.

Your score of 0.0009 indicates the model has barely learned to detect anything meaningful. The bounding boxes it predicts almost never align well with actual objects.

---

### Average Test Loss: **15.8951**

**What it means:**
This is the YOLO multi-part loss averaged across all test batches. It combines localization loss (are the box coordinates right?), confidence loss (does it know when objects are present?), and class loss (is it predicting the right category?). Lower is better.

**Is this good?** ❌ No. A loss of ~15.9 is high. Notice from training:

| Epoch | Training Loss |
|-------|--------------|
| 1     | 251.55        |
| 5     | 64.87         |
| 10    | 28.97         |

The training loss dropped significantly from 251 to 29 over 10 epochs, which is good — the model *is* learning. But the test loss of ~15.9 seems lower than the final training loss of 28.97, which is a bit unusual. This likely happens because the test set is evaluated differently (no data augmentation, smaller subset), or because the model slightly overfits to the training distribution.

**Root cause of the bad metrics:** Only 10 epochs of training on the COCO validation set (which is used here as both train/test data — a methodological issue) is not enough. YOLO typically needs hundreds of epochs, a proper training split, learning rate scheduling, and data augmentation to achieve good results.

---

## 🔢 Cell-by-Cell Code Explanation

---

### Cell 1 — Dataset Download Instructions & Initial Load Attempt

```python
import os
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import random_split
```

This cell tries to load the COCO 2017 validation dataset from a local path (`data/coco/`). If the dataset files aren't found, it prints step-by-step download instructions and sets the dataset variables to `None`. If the files exist, it loads them using PyTorch's built-in `CocoDetection` class, applies two transforms to every image (resize to 448×448 pixels, convert to a PyTorch tensor), and then does an 80/20 train-test split. Since the dataset wasn't found at this path, `yolo_train_dataset` and `yolo_test_dataset` are set to `None` here.

---

### Cell 2 — (Empty Cell)

This cell is empty — likely a placeholder.

---

### Cell 3 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

This mounts the user's Google Drive into the Colab environment so that files stored in Drive (like the COCO dataset) can be accessed at `/content/drive/My Drive/`.

---

### Cell 4 — List Google Drive Contents

```python
!ls -F '/content/drive/My Drive'
```

A shell command that lists what's in the root of the user's Google Drive. The output shows that `val2017/` and `annotations/` are already there — confirming the COCO dataset was uploaded to Drive.

---

### Cell 5 — Import Libraries & Detect GPU

```python
import torch, torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Output: Using device: cuda
```

Imports all necessary libraries for deep learning (PyTorch), data loading, plotting, and numerical operations. It then checks whether a GPU is available and sets the `device` variable accordingly. The output confirms a CUDA GPU (T4) is being used — this is important because training on a GPU is dramatically faster than on CPU.

---

### Cell 6 — (Empty Cell)

Another placeholder/empty cell.

---

### Cell 7 — Load COCO Dataset from Google Drive (Successful Load)

This is a repeat of Cell 1 but with the corrected path pointing to Google Drive (`/content/drive/My Drive`). This time the annotation file is found, so it successfully:

1. Loads the full COCO val2017 dataset (5,000 images).
2. Applies the same transforms: resize to 448×448 and convert to tensor.
3. Confirms a sample image has shape `[3, 448, 448]` (3 color channels, 448×448 pixels).
4. Splits the 5,000 images into 4,000 for training and 1,000 for testing using `random_split`.

---

### Cell 8 — Dummy COCO Target Example

```python
COCO_target = [{'bbox': [10, 20, 30, 40], 'category_id': 1}]
```

Creates a small example of what a COCO annotation looks like — a list of objects, each with a bounding box in `[x, y, width, height]` format and a category ID. This is just for illustration/testing purposes.

---

### Cell 9 — DataLoaders & Batch Shape Check

```python
def collate_fn(batch): ...

train_dataloader = DataLoader(yolo_train_dataset, batch_size=16, ...)
test_dataloader = DataLoader(yolo_test_dataset, batch_size=16, ...)
```

Defines a custom `collate_fn` function to handle the fact that different images have different numbers of objects (you can't simply stack variable-length annotation lists into a tensor). It keeps images as a stacked tensor but keeps targets as a plain Python list.

Then creates two DataLoaders — one for training (with shuffling) and one for testing (no shuffling). Both use a batch size of 16. The output confirms:
- Batched images shape: `[16, 3, 448, 448]` — 16 images, 3 channels, 448×448.
- Number of target samples: 16.

---

### Cell 10 — YOLO Configuration & Class Names

```python
S = 7   # Grid: 7x7 cells
B = 2   # 2 bounding boxes per cell
C = 80  # 80 COCO classes

classes = ['person', 'bicycle', 'car', ...]
class_to_id = {name: i for i, name in enumerate(classes)}
```

Sets three fundamental YOLO hyperparameters. In YOLOv1, the image is divided into an S×S grid (7×7 = 49 cells). Each cell predicts B bounding boxes (2 here), and each box has 5 values (x, y, w, h, confidence). On top of that, each cell predicts probabilities for C classes (80 for COCO). So the model's final output is a `7 × 7 × (2×5 + 80) = 7 × 7 × 90` tensor. Also defines all 80 COCO class names and a dictionary mapping class names to integer IDs.

---

### Cell 11 — `convert_coco_to_yolo()` Function

```python
def convert_coco_to_yolo(target, S=7, B=2, C=80):
    yolo_target = torch.zeros(S, S, B*5 + C)
    for obj in target:
        ...
        yolo_target[i, j, 0] = x_center * S - j  # x offset within cell
        yolo_target[i, j, 1] = y_center * S - i  # y offset within cell
        yolo_target[i, j, 2] = w                  # width relative to image
        yolo_target[i, j, 3] = h                  # height relative to image
        yolo_target[i, j, 4] = 1                  # objectness = 1 (object exists)
        yolo_target[i, j, 5 + class_id] = 1       # one-hot class label
    return yolo_target
```

Converts raw COCO annotations (which are in `[x, y, width, height]` pixel format) into YOLO's target format — a `7×7×90` tensor filled with zeros. For each object in the image:

1. It figures out which of the 49 grid cells the object's center falls into.
2. It stores the box coordinates *relative to that cell*.
3. Sets the confidence score to 1 (object is present).
4. Sets a one-hot vector for the class.

Only the first bounding box slot per cell is filled (a limitation — if two objects fall in the same cell, only one is stored).

---

### Cell 12 — YOLO Model Architecture

```python
class Yolo(nn.Module):
    def __init__(self):
        self.Conv_block1 = ...  # 448 → 224 → 112
        self.Conv_block2 = ...  # 112 → 56
        self.Conv_block3 = ...  # 56 → 28
        self.Conv_block4 = ...  # 28 → 14
        self.Conv_block5 = ...  # 14 → 7
        self.Conv_block6 = ...  # 7 → 7
        self.prediction_head = nn.Sequential(
            nn.Linear(1024*7*7, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, S*S*(B*5 + C))  # output: 7*7*90
        )
```

Defines the YOLOv1 convolutional neural network. The architecture consists of 6 convolutional blocks that progressively reduce the spatial dimensions from 448×448 down to 7×7 while increasing the number of feature channels (from 3 → 64 → 192 → 512 → 1024). Each block uses convolutional layers, batch normalization (for training stability), LeakyReLU activations (a variant of ReLU that allows small negative values), and max pooling (for downsampling).

After the convolutional backbone, the feature maps are flattened and passed through a fully connected "prediction head" with dropout regularization, producing the final output tensor of shape `[batch_size, 7, 7, 90]`.

---

### Cell 13 — YOLO Loss Function

```python
def yolo_loss(pred, target, S=7, B=2, C=80, lambda_coord=5, lambda_noobj=0.5):
    ...
    loc_loss = lambda_coord * (loc_xy_loss + loc_wh_loss)   # box coordinates
    conf_loss = conf_obj_loss + conf_noobj_loss              # objectness confidence
    class_loss = ...                                         # class probabilities
    total_loss = loc_loss + conf_loss + class_loss
    return total_loss / batch_size
```

Implements the YOLOv1 multi-part loss function. It computes three separate losses:

1. **Localization loss** — How far off are the predicted box coordinates (x, y, w, h) from the ground truth? Width and height use square-root differences to penalize errors on small boxes more. Weighted by `lambda_coord=5` to emphasize accurate box prediction.

2. **Confidence loss** — Does the model correctly predict *whether* an object is in each cell? Split into two parts: cells that have objects (target confidence = 1) and cells that don't (target confidence = 0). The no-object cells are downweighted by `lambda_noobj=0.5` since most cells won't contain objects.

3. **Class loss** — For cells with objects, how accurate are the class probability predictions?

---

### Cell 14 — `compute_iou()` Function

```python
def compute_iou(box1, box2):
    # intersection area / union area
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0
```

Computes **Intersection over Union (IoU)** — the standard metric for measuring how much two bounding boxes overlap. It calculates the area of the intersection rectangle divided by the area of the union of both rectangles. A score of 1.0 means perfect overlap; 0.0 means no overlap at all. Used in both NMS and mAP evaluation.

---

### Cell 15 — `compute_ap()` Function

```python
def compute_ap(recalls, precisions):
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):  # 11 thresholds: 0.0, 0.1, ..., 1.0
        p = np.max(precisions[recalls >= t]) if any recalls >= t else 0
        ap += p / 11.0
    return ap
```

Computes **Average Precision (AP)** using the classic 11-point interpolation method (from the original PASCAL VOC challenge). It samples the precision-recall curve at 11 recall thresholds (0.0, 0.1, ..., 1.0), takes the maximum precision at or above each threshold, and averages them. This gives a single number summarizing model quality.

---

### Cell 16 — Training Loop

```python
model = Yolo().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_dataloader:
        predictions = model(images)
        loss = yolo_loss(predictions, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The main training loop. Creates a YOLO model and moves it to the GPU, then uses the Adam optimizer with a learning rate of 0.001. For each epoch (full pass through the training data):

1. Sets model to training mode.
2. For each batch: runs images through the model, computes the YOLO loss, clears old gradients, runs backpropagation, updates model weights.
3. Prints the average loss for that epoch.

Training loss dropped from 251.55 → 28.97 over 10 epochs, showing the model is definitely learning, but hasn't converged to good detection performance yet.

---

### Cell 17 — Save Model

```python
torch.save(model.state_dict(), 'yolo_model.pth')
```

Saves only the model's learned parameters (weights and biases) — called the "state dict" — to a file. This is the standard PyTorch way to save a model. Later you can reload these weights without retraining.

---

### Cell 18 — `cells_to_boxes()` Function

```python
def cells_to_boxes(model_output, S=7, B=2, C=80, img_size=448):
    for i in range(S):
        for j in range(S):
            for b in range(B):
                # Decode cell-relative predictions to image-absolute pixel coordinates
                x_center_abs = (j + sigmoid(x_offset)) / S
                y_center_abs = (i + sigmoid(y_offset)) / S
                # Convert center+size to corner format [x1,y1,x2,y2]
                x1 = (x_center_abs - w/2) * img_size
                ...
```

Converts the model's raw output tensor back into a list of human-readable bounding box predictions. For every one of the 49 grid cells and both bounding boxes per cell (98 boxes total per image), it decodes the raw numbers into absolute pixel coordinates `[x1, y1, x2, y2]`, computes an overall confidence score (objectness × class probability), and returns all detections as a list of dictionaries.

---

### Cell 19 — Non-Maximum Suppression (NMS)

```python
def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.01):
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    while boxes:
        best_box = boxes.pop(0)
        if best_box['confidence'] < conf_threshold: continue
        filtered_boxes.append(best_box)
        # Remove overlapping boxes of the same class
        boxes = [box for box in boxes if
                 box['class_id'] != best_box['class_id'] or
                 compute_iou(box['bbox'], best_box['bbox']) < iou_threshold]
```

Implements **Non-Maximum Suppression** — a post-processing step to remove duplicate detections. Since the model generates 98 box proposals per image (7×7×2), many will overlap and predict the same object. NMS works by repeatedly picking the highest-confidence box, keeping it, and discarding any remaining boxes of the same class that heavily overlap with it (IoU ≥ 0.5). Boxes below the confidence threshold (0.01) are discarded entirely.

---

### Cell 20 — `evaluate_yolo_model()` & mAP Calculation

```python
def evaluate_yolo_model(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.01):
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            predictions = model(images)
            # Decode predictions → apply NMS → collect detections
            # Collect ground truth boxes
    
    # Sort all detections by confidence (highest first)
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # For each detection: check if it matches any ground truth (IoU >= 0.5)
    # → True Positive if yes, False Positive if no
    
    # Compute precision and recall at each detection threshold
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
    recalls = cumulative_tp / num_ground_truths
    
    ap = compute_ap(recalls, precisions)
    print(f"mAP @ IoU=0.5: {ap:.4f}")  # → 0.0009
```

The full evaluation pipeline. Sets the model to evaluation mode (disables dropout, uses running batch norm stats), runs all test images through the model without computing gradients (saving memory), decodes predictions and applies NMS, then matches predictions against ground truth boxes. A detection is a True Positive only if it overlaps a ground truth box of the same class by at least 50% IoU, and that ground truth hasn't already been matched. From the running TP/FP counts, it builds a precision-recall curve and computes mAP.

**Output: mAP = 0.0009** ❌

---

### Cell 21 — Test Loss Calculation

```python
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, targets in test_dataloader:
        predictions = model(images)
        loss = yolo_loss(predictions, batch_targets)
        test_loss += loss.item()

average_test_loss = test_loss / num_test_batches
print(f'Average Test Loss: {average_test_loss:.4f}')  # → 15.8951
```

Runs the model on the test set in evaluation mode and computes the average YOLO loss across all test batches. Unlike the mAP evaluation, this directly uses the loss function (same one used during training) rather than decoding boxes. The loss of 15.8951 tells us numerically how far off the model's raw predictions are from the ground truth targets.

**Output: Average Test Loss = 15.8951** ❌

---

## 🛠 How to Improve These Results

1. **Train for more epochs** — 10 epochs is very few. Try 50–100+.
2. **Use the actual training set** — The COCO `train2017` split has 118,000 images vs. just 5,000 here.
3. **Add learning rate scheduling** — Use cosine annealing or ReduceLROnPlateau.
4. **Add data augmentation** — Random flips, crops, color jitter help generalization.
5. **Use pretrained backbone weights** — Initialize Conv_block1–5 from a model pretrained on ImageNet.
6. **Fix the target coordinate bug** — In `convert_coco_to_yolo`, the COCO bbox format is `[x, y, w, h]`, so `x_center` should be `bbox[0] + bbox[2]/2` — double-check this matches how the loss expects it.