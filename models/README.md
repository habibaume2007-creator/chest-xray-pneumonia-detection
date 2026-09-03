# Trained Model Weights

This folder contains the trained model weights developed for the **Chest X-Ray Pneumonia Classification** research project.

The project evaluates two deep learning approaches:

1. **ResNet50 Transfer Learning (Baseline Model)**
2. **PneumoNet-Lite Custom CNN (Proposed Lightweight Model)**

Due to GitHub file size limitations, large model files may be hosted externally. Documentation and access information for each model are provided below.

---

# Available Models

## 1. PneumoNet-Lite Custom CNN

### Status

✅ Available in this repository

### Model File

```
pneumonet_lite_final.pth
```

---

## Model Description

PneumoNet-Lite is a lightweight custom convolutional neural network designed for binary chest X-ray pneumonia classification.

The architecture was developed from scratch and evaluated against a ResNet50 transfer learning baseline to analyze the trade-off between classification performance and model complexity.

---

## Architecture Information

- **Model Type:** Custom Convolutional Neural Network (CNN)
- **Task:** Binary Image Classification
- **Classes:**
  - NORMAL
  - PNEUMONIA
- **Parameters:** 1,207,201
- **Training Strategy:** Supervised Learning
- **Loss Function:** Weighted Binary Cross Entropy Loss
- **Optimizer:** Adam

---

## Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 88.46% |
| Precision | 86.98% |
| Recall | 95.90% |
| F1-score | 91.22% |

---

# 2. ResNet50 Transfer Learning Baseline

### Status

⚠️ Hosted externally due to GitHub file size limitations

### Model File

```
resnet50_pneumonia_final.pth
```

---

## Model Description

ResNet50 was implemented as the baseline transfer learning approach for chest X-ray pneumonia classification.

The model uses a pretrained ImageNet backbone and was fine-tuned for binary image classification.

---

## Architecture Information

- **Model Type:** ResNet50
- **Approach:** Transfer Learning
- **Pretrained Backbone:** ImageNet
- **Task:** Binary Image Classification
- **Classes:**
  - NORMAL
  - PNEUMONIA

---

## Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 79.97% |
| Precision | [ADD VALUE] |
| Recall | 100% |
| F1-score | 86.19% |

---

## Model Download

The ResNet50 trained weights can be accessed from the following Google Drive link:

[Download ResNet50 Pneumonia Model Weights](https://drive.google.com/file/d/1UNMAHtpzio8hY7gmx9lbcPmFf6cGmu0p/view?usp=sharing)

---

# Model Loading Instructions

## Loading PneumoNet-Lite

```python
import torch

model.load_state_dict(
    torch.load(
        "pneumonet_lite_final.pth",
        map_location="cpu"
    )
)

model.eval()
```

---

## Loading ResNet50

After downloading the ResNet50 weights:

```python
model.load_state_dict(
    torch.load(
        "resnet50_pneumonia_final.pth",
        map_location="cpu"
    )
)

model.eval()
```

---

# Notes

- These models were developed for research and educational purposes.
- The models are not intended for clinical diagnosis or direct medical decision-making.
- Further validation using larger and diverse clinical datasets is required before real-world deployment.
