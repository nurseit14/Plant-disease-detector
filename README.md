# Plant Disease Detection using Deep Learning

This repository contains a deep learning project aimed at the binary classification of plant leaf images into two categories: **Healthy** and **Diseased**. The project implements transfer learning using the MobileNetV2 architecture in TensorFlow/Keras to achieve efficient and accurate image classification.

## ⚠️ Important Note Regarding the Dataset
**The dataset used for training and validation of this model has not been uploaded to this repository.** 
To run the code yourself, you will need to provide your own image dataset organized into `train` and `validation` subdirectories, where each subdirectory contains folders representing the classes (i.e., `Diseased` and `Healthy`).

## Project Overview
The model takes advantage of a pre-trained **MobileNetV2** network as a feature extractor, followed by custom layers (Global Average Pooling, Dropout, and Dense layers) to perform the final binary classification.

### Key Features:
- **Data Augmentation:** Utilizes `ImageDataGenerator` to artificially expand the training dataset and prevent overfitting. Applied augmentations include rotation, zoom, width/height shifts, shear, and horizontal flips.
- **Transfer Learning:** Fine-tunes a MobileNetV2 model (pre-trained on ImageNet), which is highly optimized and efficient for image classification tasks.
- **Evaluation Metrics:** The project evaluates the trained model's performance using classification reports and confusion matrices from `scikit-learn`.

## Repository Contents
- `Plant_Disease_Project.ipynb`: The main Jupyter Notebook containing the entire pipeline: data loading, preprocessing, model definition, training, and evaluation.
- `Plant_disease_project_1820259062_18059064_18059074.docx`: Detailed project report and documentation.

## Requirements
To run the notebook locally, you need the following Python libraries installed:
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib

You can install the dependencies via pip:
```bash
pip install tensorflow numpy scikit-learn matplotlib
```

## How to use
1. Clone this repository to your local machine.
2. Place your images inside a folder named `dataset` in the root directory. Ensure it follows this structure:
    ```text
    dataset/
      ├── train/
      │     ├── Diseased/
      │     └── Healthy/
      └── validation/
            ├── Diseased/
            └── Healthy/
    ```
3. Open `Plant_Disease_Project.ipynb` using Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells sequentially to train the model and view the evaluation results.
