# torch/nn Integration

## Overview
The nn package provides neural network modules and layers for Torch.

**Repository**: https://github.com/torch/nn

## Purpose
Provides building blocks for neural networks including:
- Linear layers
- Convolutional layers
- Activation functions
- Loss functions
- Containers (Sequential, Parallel, etc.)

## Integration with hypATen

The nn modules can be used for EMG signal classification and pattern recognition, replacing or augmenting the existing MATLAB-based classifiers.

## Key Modules

### Layers
- `nn.Linear`: Fully connected layer
- `nn.SpatialConvolution`: 2D convolution
- `nn.TemporalConvolution`: 1D convolution (useful for time series)
- `nn.LSTM`: Long Short-Term Memory layer
- `nn.GRU`: Gated Recurrent Unit

### Activations
- `nn.ReLU`: Rectified Linear Unit
- `nn.Sigmoid`: Sigmoid activation
- `nn.Tanh`: Hyperbolic tangent
- `nn.Softmax`: Softmax for classification

### Containers
- `nn.Sequential`: Sequential container
- `nn.Parallel`: Parallel container
- `nn.Concat`: Concatenation

## Usage Example

```lua
require 'nn'

-- Build a simple feedforward network
local model = nn.Sequential()
model:add(nn.Linear(numFeatures, 128))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(128, 64))
model:add(nn.ReLU())
model:add(nn.Linear(64, numClasses))
model:add(nn.LogSoftMax())

-- Define loss function
local criterion = nn.ClassNLLCriterion()

-- Forward pass
local output = model:forward(input)
local loss = criterion:forward(output, target)

-- Backward pass
local gradOutput = criterion:backward(output, target)
model:backward(input, gradOutput)
```

## EMG Signal Classification Example

```lua
-- Network for EMG pattern classification
local emgClassifier = nn.Sequential()
emgClassifier:add(nn.TemporalConvolution(numChannels, 32, 5))
emgClassifier:add(nn.ReLU())
emgClassifier:add(nn.TemporalMaxPooling(2))
emgClassifier:add(nn.TemporalConvolution(32, 64, 3))
emgClassifier:add(nn.ReLU())
emgClassifier:add(nn.Reshape(64 * timeSteps))
emgClassifier:add(nn.Linear(64 * timeSteps, numClasses))
emgClassifier:add(nn.LogSoftMax())
```

## Installation

```bash
luarocks install nn
```

## References
- Original repository: https://github.com/torch/nn
- Documentation: https://github.com/torch/nn/tree/master/doc
