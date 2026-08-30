# Chest X-Ray Pneumonia Classification Using ResNet50 Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Classification](https://img.shields.io/badge/Task-Binary%20Image%20Classification-blue)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Image%20Analysis-green)
![Transfer Learning](https://img.shields.io/badge/Method-ResNet50%20Transfer%20Learning-orange)
![Research Project](https://img.shields.io/badge/Project-Research%20Implementation-purple)

---

# Project Overview

This repository presents a medical image classification approach for categorizing chest X-ray images into two classes:

- **NORMAL**
- **PNEUMONIA**

The objective of this research project was to develop and evaluate a binary image classification model capable of distinguishing pneumonia-positive and normal chest X-ray images using a ResNet50 transfer learning approach.

The project demonstrates a complete research workflow, including:

- Dataset exploration
- Image preprocessing
- Data augmentation
- Transfer learning-based classification
- Experimental comparison of training strategies
- Model evaluation
- Performance visualization
- Classification analysis

---

# Research Motivation

Chest X-ray imaging is one of the most commonly used diagnostic methods for respiratory diseases such as pneumonia. However, accurate interpretation of radiographic images requires expert knowledge and may become challenging due to workload, limited resources, and variations in image interpretation.

Recent advances in computer vision and deep learning techniques have enabled automated analysis of medical images by learning meaningful visual patterns from image datasets.

This research investigates the application of deep learning-based classification techniques for distinguishing pneumonia and normal cases from chest X-ray images.

---

# Research Objectives

The main objectives of this project were:

1. To develop a binary classification model for categorizing chest X-ray images into NORMAL and PNEUMONIA classes.

2. To implement ResNet50-based transfer learning for chest X-ray image classification.

3. To investigate the effect of different loss strategies on classification performance.

4. To evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix analysis.

5. To analyze model behaviour through experimental results and visualization techniques.

---

# Dataset Description

## Chest X-Ray Pneumonia Dataset

This project uses the publicly available Chest X-Ray Pneumonia dataset.

The dataset contains labeled chest X-ray images belonging to two categories:

| Class | Description |
|---|---|
| NORMAL | Chest X-ray images without pneumonia |
| PNEUMONIA | Chest X-ray images showing pneumonia |

Dataset organization:

```
Chest X-Ray Dataset

├── Train
│   ├── NORMAL
│   └── PNEUMONIA
│
├── Validation
│   ├── NORMAL
│   └── PNEUMONIA
│
└── Test
    ├── NORMAL
    └── PNEUMONIA
```

The dataset is not included in this repository due to size limitations.

---

# Methodology

The classification workflow followed these steps:

```
Chest X-Ray Images

        ↓

Image Preprocessing

        ↓

Data Augmentation

        ↓

ResNet50 Transfer Learning Classifier

        ↓

Model Training

        ↓

Classification Evaluation

        ↓

Experimental Analysis
```

---

# Image Preprocessing

Before model training, chest X-ray images were processed using:

- Image resizing
- Tensor conversion
- Pixel normalization
- Data augmentation techniques

Data augmentation was applied to improve model generalization and reduce overfitting.

The preprocessing pipeline was implemented using PyTorch and Torchvision.

---

# Classification Model Architecture

## ResNet50 Transfer Learning Classifier

This project uses ResNet50 as the classification architecture through transfer learning.

ResNet50 was selected because:

- It is a well-established convolutional neural network architecture.
- It provides effective feature extraction capabilities.
- Pre-trained weights improve learning efficiency.
- It has demonstrated strong performance in image classification tasks.

The classification architecture follows:

```
Input Chest X-Ray Image

        ↓

Pre-trained ResNet50 Feature Extractor

        ↓

Classification Layer

        ↓

NORMAL / PNEUMONIA Prediction
```

---

# Experimental Design

Two classification experiments were conducted to investigate the impact of different training strategies.

---

## Experiment 1: Baseline ResNet50 Classification Model

### Objective

To establish baseline classification performance using the standard training approach.

### Configuration

- ResNet50 transfer learning classifier
- Standard Cross Entropy Loss
- Adam optimizer
- Binary image classification framework

---

## Experiment 2: Weighted Loss ResNet50 Classification Model

### Objective

To investigate whether class-weighted loss improves classification performance by addressing class imbalance.

### Configuration

- ResNet50 transfer learning classifier
- Weighted Cross Entropy Loss
- Same evaluation methodology as baseline experiment

---

# Classification Evaluation

The trained models were evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify pneumonia cases |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Detailed classification performance analysis |

---

# Experimental Results

The repository contains the generated experimental artifacts:

```
results/

├── confusion_matrix.png

├── training_validation_loss.png

├── training_validation_f1.png

├── model_experiment_comparison.csv

└── experiment1_history.json
```

These artifacts provide:

- Confusion matrix analysis
- Training behaviour visualization
- F1-score progression
- Comparison between classification experiments
- Model training history

---

# Repository Structure

```
chest-xray-pneumonia-detection/

│
├── README.md
│
├── requirements.txt
│
├── notebooks/
│   └── chest_xray_pneumonia_detection.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_validation_loss.png
│   ├── training_validation_f1.png
│   ├── model_experiment_comparison.csv
│   └── experiment1_history.json
│
├── models/
│   └── README.md
│
└── dataset/
    └── README.md
```

---

# Installation Requirements

## Python Environment

Recommended:

```
Python 3.10+
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:

```
PyTorch
Torchvision
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Pillow
TQDM
Jupyter
```

---

# Running the Project

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd chest-xray-pneumonia-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
notebooks/chest_xray_pneumonia_detection.ipynb
```

The notebook contains:

- Dataset loading
- Exploratory analysis
- Image preprocessing
- Model development
- Classification experiments
- Evaluation
- Results visualization

---

# Model Files

The trained model weights are not directly uploaded because of GitHub file size limitations.

The models generated during this research include:

- Final ResNet50 classification model
- Best experimental checkpoint

Model documentation is available in:

```
models/README.md
```

---

# Limitations

This project has several limitations:

- The model was evaluated using a publicly available dataset.
- Dataset-specific characteristics may affect generalization.
- External clinical validation is required.
- The model should not be considered a replacement for professional medical diagnosis.

---

# Future Research Directions

Future improvements may include:

- Explainable AI techniques such as Grad-CAM.
- Evaluation using external clinical datasets.
- Comparison with advanced classification architectures.
- Vision Transformer-based classification approaches.
- Improved robustness through domain adaptation.

---

# Skills Demonstrated

This project demonstrates practical experience in:

- Medical image classification
- Computer vision
- Deep learning-based classification methods
- Transfer learning
- ResNet50 implementation
- PyTorch framework
- Experimental analysis
- Model evaluation
- Research documentation

---

# Author

**Um e Habiba**

Medical Image Classification Research Project

**Chest X-Ray Pneumonia Classification Using ResNet50 Transfer Learning**

---

# Citation

If you use this repository for academic purposes, please cite:

```
Um e Habiba.
Chest X-Ray Pneumonia Classification Using ResNet50 Transfer Learning.
GitHub Repository.
```
