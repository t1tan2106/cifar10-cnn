# CIFAR-10 CNN Classifier

A convolutional neural network (CNN) built in PyTorch to classify images from the CIFAR-10 dataset. This project was developed as part of a hands-on learning exercise in deep learning and AI safety.

---

## Project Overview

CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The goal of this project was to build a CNN capable of classifying these images with high accuracy on unseen data, while practicing good ML workflow and coding practices.

---
## Code and Logic

### Model Architecture

The CNN is defined in `src/model.py`:

- **3 Convolutional Layers** with increasing filter sizes (32 → 64 → 128)  
- **Max Pooling** after each convolution to reduce spatial dimensions  
- **Batch Normalization** to stabilize training and speed up convergence  
- **Dropout** layers to reduce overfitting  
- **Fully Connected Layers** to map features to class probabilities  
- **ReLU Activation** for non-linearity  

This structure allows the network to capture increasingly complex patterns in the images, while controlling overfitting and maintaining stable gradients.

### Data Augmentation

To improve generalization and reduce overfitting:

- **Random horizontal flips**  
- **Random crops with padding**  
- **Normalization** to center pixel values  

This ensures the model sees slightly different versions of the same image during training, improving robustness on unseen data.

### Training Strategy

- **Loss function:** CrossEntropyLoss (suitable for multi-class classification)  
- **Optimizer:** Adam (adaptive learning rate for faster convergence)  
- **Learning Rate Scheduler:** ReduceLROnPlateau to decrease learning rate when validation loss stagnates  
- **Early Stopping:** Stops training if validation loss does not improve for several epochs  
- **Training/Validation Split:** Ensures we monitor generalization performance  

These strategies collectively reduce overfitting and improve test accuracy.

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
