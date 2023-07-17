# Paperplay: LeNet-5

> An ongoing series of me replicating paper!

This repository contains my own implementation of LeNet-5, a convolutional neural network architecture proposed by Yann LeCun et al. in their paper "Gradient-Based Learning Applied to Document Recognition". LeNet-5 is a pioneering model that played a significant role in the advancement of deep learning, particularly in the field of image recognition.

## Introduction

The goal of this project is to provide an implementation of LeNet-5 that closely follows the details described in the original paper. By reading and understanding the paper, I have developed this implementation to gain practical insights into the architecture and learn about the inner workings of LeNet-5.

## Features

- Follows the original LeNet-5 architecture as described in the paper.
- Built using PyTorch, a popular deep learning framework.
- Utilizes common convolutional neural network components such as convolutional layers, average pooling, and fully connected layers.
- Supports training and inference on datasets suitable for LeNet-5, such as the MNIST dataset.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch (install with `conda install torch`)

### Usage

```python
python3 train.py --batchsize [batch_size] --learningrate [learningrate] --epochs [epochs]
```

Please note that you can change the value of every square-bracketed words

## References

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11), 2278–2324.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
