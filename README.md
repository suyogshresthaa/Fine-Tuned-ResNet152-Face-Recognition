# Fine-Tuned ResNet152 Face Recognition Using Transfer Learning

## Author
Suyog Shrestha

Data Science & Business @ Knox College - 2027

---

## Overview

This project fine-tunes **ResNet152**, a state-of-the-art convolutional neural network, on a custom facial image dataset to classify whether a given image is of **Cristiano Ronaldo** or not.

The project demonstrates the complete transfer learning pipeline: dataset preparation, data augmentation, model fine-tuning, evaluation, and deployment via a **Streamlit** web application.

---

## How to Run This Project

**To run the notebook:**
1. Clone the repository
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Launch Jupyter Notebook
4. Run all cells sequentially in:
    - `CNN_Transfer_Learning.ipynb`

> **Note:** The trained model weights (`best_model.pth`) are not included in this repository due to file size. To generate them, run all cells in the notebook before launching the Streamlit app.

**To run the Streamlit app:**
1. Make sure `best_model.pth` is in the same directory as `cnn_app.py`
2. Run:
    ```bash
    streamlit run app.py
    ```

> **Note:** A GPU is strongly recommended for training. PyTorch will automatically detect and use a GPU if available. Training on CPU will be significantly slower.

---

## Dataset

The dataset consists of facial images of 4 celebrities:
- **Positive class:** Cristiano Ronaldo (109 images)
- **Negative class:** Kane Williamson, Kobe Bryant, Maria Sharapova (332 images)
- **Total:** 441 images

The dataset was split 80/20 into training and validation sets:
- **Train:** 352 images
- **Validation:** 89 images

Data augmentation techniques were applied to the training set including:
- Random resized crop
- Random horizontal flip
- Color jitter

---

## Modeling Approach

The problem is formulated as a **binary image classification task** with 2 output classes (Ronaldo / Not Ronaldo).

The model used is **ResNet152** from PyTorch's Torchvision package, pretrained on ImageNet. ResNet152 was chosen because it is the deepest and most accurate ResNet architecture, making it better at capturing fine-grained facial features.

Fine-tuning was performed using the following configuration:

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning Rate | 0.001 |
| Momentum | 0.9 |
| Epochs | 25 |
| Batch Size | 4 |
| LR Scheduler | StepLR (step=7, gamma=0.1) |

---

## Key Findings and Evaluation

Model performance was evaluated using a **Confusion Matrix** on the validation set:

|  | Predicted Not Ronaldo | Predicted Ronaldo |
|---|---|---|
| **Actual Not Ronaldo** | 61 (TN) | 3 (FP) |
| **Actual Ronaldo** | 2 (FN) | 23 (TP) |

- **Best Validation Accuracy: 96.63%**
- Only 5 misclassifications out of 89 validation images
- The model correctly identified 23 out of 25 Ronaldo images
- The model correctly identified 61 out of 64 Not Ronaldo images

---

## Streamlit App

The trained model is deployed as a **Streamlit** web application that allows users to:
- Upload a facial image
- Get a prediction (Ronaldo / Not Ronaldo)
- View the model's confidence score

---

## Future Work

Potential extensions of this project include:
- Expanding the dataset with more Ronaldo images for improved accuracy
- Experimenting with other architectures such as EfficientNet or VGG16
- Adding face detection preprocessing to automatically crop faces before classification
- Extending to a multi-class classifier to identify multiple celebrities