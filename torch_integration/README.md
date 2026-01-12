# Torch Integration for hypATen

This directory contains integration modules for various Torch-based libraries to enhance the hypATen model with deep learning capabilities.

## Integrated Repositories

### 1. hypermind
**Repository**: https://github.com/nicholas-leonard/hypermind
**Purpose**: Hyperparameter optimization for Torch neural networks
**Integration**: See `hypermind/` directory

### 2. ATen (A Tensor Library)
**Repository**: https://github.com/zdevito/ATen
**Purpose**: C++ tensor library for efficient tensor operations (core of PyTorch)
**Integration**: See `aten/` directory

### 3. torch/nn
**Repository**: https://github.com/torch/nn
**Purpose**: Neural network modules and layers
**Integration**: See `nn/` directory

### 4. torch/rnn
**Repository**: https://github.com/torch/rnn
**Purpose**: Recurrent neural network modules (LSTM, GRU, etc.)
**Integration**: See `rnn/` directory

### 5. torch/nngraph
**Repository**: https://github.com/torch/nngraph
**Purpose**: Graph computation for neural networks with complex architectures
**Integration**: See `nngraph/` directory

### 6. torch/sys
**Repository**: https://github.com/torch/sys
**Purpose**: System utilities for Torch
**Integration**: See `sys/` directory

### 7. torch/distro
**Repository**: https://github.com/torch/distro
**Purpose**: Torch distribution and package management
**Integration**: See `distro/` directory

## Usage

The integration modules are designed to work with the existing MiniVIE signal processing framework, providing deep learning capabilities for:
- Pattern recognition in EMG signals
- Neural network-based classification
- Recurrent models for temporal signal processing
- Hyperparameter optimization for model training

## Installation

To use these integrations, you'll need to have Torch installed. Follow the installation instructions in each subdirectory.

## Integration with MiniVIE

These modules can be used alongside the existing MATLAB-based pattern recognition in `MiniVIE/+PatternRecognition/` and signal analysis in `MiniVIE/+SignalAnalysis/`.
