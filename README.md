# Breast Cancer Detection Using CNN (PyTorch)

This repository contains a Convolutional Neural Network (CNN) model to detect breast cancer from images. It provides a complete pipeline for dataset preparation, model training, evaluation, and inference. The project can be run locally or via Docker for easy replication. 

---

## Features

- Dataset download and preprocessing from Kaggle (`dataset.py`)
- Custom dataset creartion and annotation file (`src/create_dataset.py`, `src/create_annotation_file.py`)
- Data loading and transformations (`src/dataloader`, `src/transforms`)
- CNN model implementation (`src/simple_cnn_model.py`)
- Loss function and optimizer (`src/loss_function.py`, `optimizer.py`)
- Training pipeline with validation (`src/train_model.py`)
- Evaluation metrics and threshold calculation (`src/evaluation.py`, `src/threshold.py`)
- Model saving and inference (`src/save_model.py`, `src/inference`)
- Display random images from dataset (`src/display_random_images.py`)
- Example to use the pipeline (`train.ipynb`)

---

## Repository structure

``` bash
.
├── Dockerfile
├── .dockerignore
├── dataset.py
├── train.ipynb
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py
    ├── create_annotation_file.py
    ├── create_dataset.py
    ├── dataloader.py
    ├── display_random_images.py
    ├── evaluate.py
    ├── inference.py
    ├── load_model.py
    ├── loss_function.py
    ├── optimizer.py
    ├── save_model.py
    ├── simple_cnn_model.py
    ├── threshold.py
    ├── train_model.py
    └── transforms.py
```

---

## How to start?

### 1. Using clone repository

``` bash
git clone https://github.com/bhanarkarjetal/breast_cancer_detection.git
cd breast_cancer_detection
```

### 2. Using Docker (recommended and easiest)
This repository also includes a Docker image for easy replication

``` bash
docker pull bhanarkarjetal/breast-cancer-detection:v1
docker run -it --rm bhanarkarjetal/breast-cancer-detection:v1
```

### 3. Local Setup

Install dependencies and run the Notebook for example

``` bash
python3 -m venv .venv
source .venv/bin/activate   # for Linux/Mac

.venv\Scripts\activate      # for windows

pip install -r requirements.txt

jupyter notebook train.ipynb
```

---

## Usage

1. Prepare dataset: Use `dataset.py` to download the dataset from Kaggle
2. Create dataset: Use `create_annotation_file.py` to create an annotation file for the dataset and use `create_dataset.py` to generate dataset
3. Define loss function and optimizer: Use `loss_function.py` and `optimizer.py`
3. Train model: Use `train_model.py` to train the model 
4. Evaluate: Use `evaluate.py` and `threshold.py` 
5. Save model: Use `save_model.py` to save state_dict and entire model
6. Inference: User `inference.py` on new images

---

## Notes

- All functions and classes are modular inside the src package for reuse.
- .dockerignore ensures unnecessary files are excluded from Docker builds.
- Docker image ensures consistent environment without local setup issues.
```

---

This version highlights the **fastest way to run the project via Docker**, while still mentioning local setup and repo structure.  

---

## Licence

This project is licensed under the MIT License.