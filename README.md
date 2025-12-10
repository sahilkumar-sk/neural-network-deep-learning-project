# Neural Network Deep Learning Project
### Image Classification using Transfer Learning on PASCAL VOC 2008

This project implements an end-to-end deep learning pipeline to classify images (e.g., dog vs. non-dog or dining-table vs. non-dining-table) using multiple state-of-the-art vision architectures.  
It demonstrates dataset handling, model fine-tuning, evaluation, visualization, and comparison of transformer-based and CNN-based models.

---

## Features

### 1. Dataset: PASCAL VOC 2008
- Automated dataset download & extraction
- Custom PyTorch `Dataset` class for structured loading and binary labeling 
- Image preprocessing (resize, tensor conversion, normalization)  
- Hyperparameters such as batch size, epochs, and learning rate defined at the top

### 2. Models Implemented
The project fine-tunes several pretrained deep learning architectures:

| Model Type | Architectures Used |
|------------|-------------------|
| **Transformers** | Vision Transformer (ViT), Swin Transformer
| **CNNs** | ResNet-50, DenseNet-201

Each model:
- Loads pretrained ImageNet weights  
- Replaces final classifier with a custom two-class output layer 
- Trains using GPU 

### 3. Training Pipeline
Includes:
- Train/validation split  
- Early stopping & learning-rate scheduling 
- Optimizers (Adam)  
- Mean Average Precision (mAP) evaluation  
- Logging and saving best model weights  

### 4. Evaluation
- Computes **mAP** for performance comparison 
- Generates **Top-10 predictions visualization**  
- Compares transformer vs. CNN performance  
- Saves best-performing model weights  

---

## Tech Stack

| Component | Tools |
|----------|-------|
| **Language** | Python |
| **Deep Learning** | PyTorch, Torchvision, timm |
| **Models** | ViT, Swin Transformer, ResNet-50, DenseNet-201 |
| **Dataset** | PASCAL VOC 2008 |
| **Utilities** | NumPy, Matplotlib, tqdm, scikit-learn |

---

## Project Structure

```

neural-network-deep-learning-project/
│
├── notebooks/                 # Jupyter notebooks for experiments
├── reports/
│   └── Project_report.pdf     # Project report (Download it)
├── src/
│   └── nn_project.py          # Main training & evaluation script
└── README.md                  # Project documentation

````

---

##  How to Run

### 1️ Clone the repo
````
git clone https://github.com/sahilkumar-sk/neural-network-deep-learning-project.git
cd neural-network-deep-learning-project
````

### 2️ Install dependencies

```bash
pip install torch torchvision timm matplotlib scikit-learn
```

### 3️ Run the training script

```bash
python src/nn_project.py
```

This will:
- Download PASCAL VOC 2008
- Build datasets/dataloaders
- Train all selected models
- Compute evaluation metrics
- Save best weights

---

## Results Summary

| Model              | Performance (mAP)                              |
| ------------------ | ---------------------------------------------- |
| Vision Transformer | Higher mAP and stronger generalization         |
| Swin Transformer   | Performs well on structured features           |
| ResNet-50          | Fast training, competitive accuracy            |
| DenseNet-201       | Deep architecture with good feature extraction |
