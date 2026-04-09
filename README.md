# Image Classification with CNN on CIFAR-10

## Abstract

This project focuses on implementing an image classification system using Convolutional Neural Networks (CNNs) with transfer learning. A pre-trained ResNet18 model is fine-tuned on the CIFAR-10 dataset to classify images into ten categories. The model leverages pre-trained weights from ImageNet, improving performance and reducing training time. The final model achieves approximately 92.8% test accuracy, demonstrating the effectiveness of transfer learning for image classification tasks.

## Introduction

Image classification is one of the fundamental problems in computer vision, where the goal is to assign a label to an input image from a predefined set of categories. Traditional machine learning approaches relied heavily on handcrafted feature extraction, which often failed to generalize across diverse datasets.

With the advancement of deep learning, Convolutional Neural Networks (CNNs) have become the standard approach for image-related tasks. However, training deep CNNs from scratch requires large datasets and significant computational resources.

To overcome this limitation, transfer learning is widely used. In transfer learning, a model pre-trained on a large dataset (such as ImageNet) is adapted to a new task. This project uses a pre-trained ResNet18 model and fine-tunes it on the CIFAR-10 dataset to achieve high classification accuracy with reduced training effort.

## Objective

The main objectives of this project are:

  1) To implement an image classification pipeline using CNNs.
  2) To apply transfer learning using a pre-trained ResNet18 model.
  3) To improve model performance using data augmentation techniques.
  4) To evaluate the model using training, validation, and test datasets.
  5) To demonstrate real-world prediction on external images.

