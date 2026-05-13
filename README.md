# Image Classification with CNN on CIFAR-10

A transfer learning project that fine-tunes **ResNet18** on the **CIFAR-10** dataset using PyTorch.
The training pipeline includes augmentation, validation tracking, learning-rate scheduling, and reproducible module-level utilities with test coverage.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Highlights](#-key-highlights)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Model & Training Pipeline](#-model--training-pipeline)
- [Data Processing & Augmentation](#-data-processing--augmentation)
- [Configuration](#-configuration)
- [How to Run](#-how-to-run)
- [Testing & CI](#-testing--ci)
- [Current Results](#-current-results)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🎯 Project Overview

This repository implements an end-to-end image classification workflow for CIFAR-10.
Instead of training a convolutional network from scratch, it adapts a pretrained ImageNet backbone (**ResNet18**) and replaces the final classification layer for CIFAR-10’s 10 classes.

The codebase is organized around reusable functions for:

- selecting execution device (CPU/CUDA),
- building train/eval transforms,
- loading datasets,
- creating train/validation/test loaders,
- training for one epoch,
- evaluation,
- full training orchestration with best-checkpoint tracking.

---

## ✨ Key Highlights

- **Transfer learning with ResNet18** (`torchvision.models.resnet18` with pretrained weights)
- **Custom classifier head** for 10-way CIFAR-10 prediction
- **Data augmentation pipeline** for stronger generalization
- **Validation-driven checkpointing** (best model weights by validation accuracy)
- **Learning-rate scheduling** (`StepLR`) with Adam optimizer
- **Pytest-based test suite** and GitHub Actions CI

---

## 🗂 Repository Structure

```text
Image-Classification-with-CNN-on-CIFAR-10/
├── src/
│   └── transfer_cnn.py          # Core training + evaluation pipeline
├── tests/
│   └── test_transfer_cnn.py     # Unit and workflow tests
├── .github/workflows/
│   └── python-ci.yml            # CI: dependency install + pytest
├── report/
│   └── report.pdf               # Project report
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── script.sh                    # Environment check + training launch script
└── README.md
```

---

## 🧪 Dataset

The project uses **CIFAR-10**:

- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **60,000 images total**
  - 50,000 training images
  - 10,000 test images
- **Image size**: 32×32 RGB

### Split strategy in this project

From the CIFAR-10 training set, the code performs an internal split:

- **Train split**: 80% (40,000)
- **Validation split**: 20% (10,000)
- **Test split**: official CIFAR-10 test set (10,000)

---

## 🧠 Model & Training Pipeline

### Backbone

- Pretrained **ResNet18** loaded via `ResNet18_Weights.DEFAULT`
- Final fully connected layer replaced with `nn.Linear(in_features, 10)`

### Optimization setup

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (`lr = 0.001`)
- **Scheduler**: StepLR (`step_size = 5`, `gamma = 0.1`)
- **Epochs**: 50
- **Dataloader batch size used in training loop**: 32

### Training behavior

For each epoch:

1. Train on train loader (`train_one_epoch`)
2. Evaluate on validation loader (`evaluate`)
3. Step LR scheduler
4. Save best model weights when validation accuracy improves

At the end of training, the best validation checkpoint is restored.

---

## 🖼 Data Processing & Augmentation

### Train transform

- Resize to 256×256
- Random resized crop to 224×224
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness/contrast/saturation/hue)
- Normalize with ImageNet mean/std

### Validation/Test transform

- Resize to 224×224
- Normalize with ImageNet mean/std

This design aligns CIFAR-10 images with the expected input style of pretrained ResNet18.

---

## ⚙️ Configuration

The main configuration values in `src/transfer_cnn.py` are:

- `learning_rate = 0.001`
- `num_epochs = 50`
- `image_size = 224`

> Note: `batch_size` is defined globally as `256`, but `run_training()` currently calls `create_dataloaders(..., loader_batch_size=32)`, so effective training uses batch size 32.

---

## ▶️ How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Ensure CIFAR-10 data availability

By default, `run_training()` loads with `download=False`, so dataset files should already exist under `./data`.

### 3) Start training

```bash
python src/transfer_cnn.py
```

### Optional script

You can also use:

```bash
bash script.sh
```

It checks Python 3.10 availability, verifies/install modules from `requirements.txt`, and runs `src/transfer_cnn.py`.

---

## ✅ Testing & CI

### Run tests locally

```bash
pytest -v
```

### CI workflow

GitHub Actions (`.github/workflows/python-ci.yml`) automatically:

1. Sets up Python 3.10
2. Installs dependencies from `requirements.txt`
3. Runs `pytest -v`

---

## 📈 Current Results

As documented in the original project abstract:

- **Approximate test accuracy: 92.8%**

This demonstrates that transfer learning can achieve strong CIFAR-10 performance with a compact, maintainable training pipeline.

---

## ⚠️ Limitations

- `download=False` in dataset loading requires pre-downloaded CIFAR-10 data unless changed.
- No checkpoint file export utility is included yet (best weights are kept in memory during run).
- No command-line argument parsing for hyperparameter overrides.
- No inference script for single-image prediction in the current `src` module.

---

## 🛣 Roadmap

Potential improvements:

- Add CLI flags (epochs, batch size, learning rate, data root, download mode)
- Add model checkpoint save/load support
- Add confusion matrix + per-class metrics
- Add TensorBoard or experiment tracking integration
- Add inference script for external images

---

## 📄 License

This project is distributed under the repository’s [LICENSE](LICENSE).
