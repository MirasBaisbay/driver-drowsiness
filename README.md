# ROBT 310: Image Processing Final Project done by Miras Baisbay, Nazerke Abdrakhmanova, Anel Salmenova, and Kamila Spanova
# Driver Drowsiness Detection

This project implements a driver drowsiness detection system using convolutional neural networks (CNNs) and image preprocessing techniques. It evaluates three different CNN architectures and incorporates DeepFace for facial feature extraction to enhance the detection process.

---

## Features

1. **Preprocessing**:
   - Utilizes DeepFace to extract faces from input images.
   - Provides clean and consistent datasets for model training and evaluation.

2. **Model Comparisons**:
   - **CNN-2 blocks**: Lightweight architecture with 2 blocks of convolutional layers.
   - **CNN-4 blocks**: Extended architecture with 4 blocks of convolutional layers.
   - **EfficientNet-B2**: Advanced model for better feature extraction and performance.

3. **Training Variations**:
   - Models are trained with and without data augmentation for comparative analysis.

4. **Evaluation**:
   - Visualizes training history, confusion matrices, and misclassified cases.
   - Predicts drowsiness states from input images and video streams.

---

## Project Structure

| File/Folder                 | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `preprocessing.py`          | Extracts faces from images using DeepFace.                                  |
| `CNN2blocks/`               | Contains scripts and results for the CNN-2 blocks model.                    |
| `CNN4block/`                | Contains scripts and results for the CNN-4 blocks model.                    |
| `EfficientNetB2/`           | Contains scripts and results for the EfficientNet-B2 model.                 |
| `README.md`                 | Project documentation.                                                     |
| `aug/`                      | Includes model scripts and outputs with data augmentation.                 |
| `no_aug/`                   | Includes model scripts and outputs without data augmentation.              |

---

## Dataset

The project uses a dataset from Roboflow for drowsiness detection. Images were preprocessed with DeepFace to ensure consistent face extraction. Link to the dataset: https://universe.roboflow.com/augmented-startups/drowsiness-detection-cntmz

---

## Prerequisites

- **Python** 3.8+
- TensorFlow or PyTorch
- DeepFace library
- A CUDA-enabled GPU for model training
