# 🧠 Brain MRI Anomaly Detection

This repository provides a full pipeline for **Brain MRI anomaly detection** using **autoencoder models**.  
It includes scripts for dataset management, preprocessing, and visualization — all within a clean and reproducible environment.

---

## 📦 Project Overview

The goal of this project is to train deep autoencoders to detect **abnormal brain regions** in MRI scans.  
The utilities in `src/utils/` help automate each step:

| Script | Purpose |
|--------|----------|
| `download_datasets.py` | Downloads the BraTS20 MRI dataset automatically from Kaggle. |
| `preprocessing.py` | Converts raw `.nii` MRI files into preprocessed NumPy volumes for model training. |
| `convert_npy_to_nifti.py` | Converts processed `.npy` arrays back to `.nii` format for visualization or post-processing. |
| `visualize_slices.py` | Provides simple visualization utilities (2D slices, 3D renderings, GIFs) for MRI volumes. |

Each script is modular and can be run independently.

---

## 🧰 Prerequisites

- **Python** ≥ 3.9  
- **pip** (latest version recommended)  
- **GNU Make** for task automation  
- *(Optional)* **PowerShell** for Windows convenience scripts  
- *(Optional)* **Kaggle CLI** (for dataset download)

---

## ⚙️ Environment Setup

The project uses a local virtual environment stored in `.venv`.  
The provided **Makefile** works on both Linux/macOS and Windows.

### 1️⃣ Create the virtual environment

```bash
make venv
