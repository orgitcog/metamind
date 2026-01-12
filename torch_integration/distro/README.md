# torch/distro Integration

## Overview
The distro package provides the Torch distribution and package management system.

**Repository**: https://github.com/torch/distro

## Purpose
Torch7 distribution that includes:
- Complete Torch installation
- Package manager (luarocks)
- Core libraries and dependencies
- Build system
- Documentation

## Integration with hypATen

The distro provides the foundation for running all Torch-based components in hypATen. It ensures all dependencies are properly installed and configured.

## What's Included

### Core Components
- **Torch7**: Core tensor library
- **nn**: Neural network modules
- **optim**: Optimization algorithms
- **paths**: File system utilities
- **image**: Image processing
- **gnuplot**: Plotting utilities

### Scientific Computing
- **torch**: Tensor operations
- **cutorch**: CUDA tensors
- **cunn**: CUDA neural networks

### Utilities
- **trepl**: Torch REPL (interactive shell)
- **dok**: Documentation system
- **luarocks**: Package manager

## Installation

### Ubuntu/Linux
```bash
# Clone the distribution
git clone https://github.com/torch/distro.git ~/torch --recursive

# Install dependencies
cd ~/torch
bash install-deps

# Install Torch
./install.sh

# Add to path
source ~/.bashrc
```

### macOS
```bash
# Install dependencies via homebrew
brew install cmake

# Clone and install
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
./install.sh
```

## Package Management

### Installing Packages
```bash
# Install a package with luarocks
luarocks install nn
luarocks install rnn
luarocks install nngraph
luarocks install optim

# Install from source
luarocks make
```

### Listing Installed Packages
```bash
luarocks list
```

### Searching for Packages
```bash
luarocks search rnn
```

## Setting Up hypATen Environment

```bash
# Install all required packages for hypATen
luarocks install nn
luarocks install rnn
luarocks install nngraph
luarocks install optim
luarocks install image
luarocks install gnuplot
luarocks install hdf5  # For data storage
```

## Integration Script

Create a setup script for hypATen:

```bash
#!/bin/bash
# setup_torch_env.sh

echo "Setting up Torch environment for hypATen..."

# Check if Torch is installed
if ! command -v th &> /dev/null; then
    echo "Torch not found. Please install from https://github.com/torch/distro"
    exit 1
fi

# Install required packages
echo "Installing required Torch packages..."
luarocks install nn
luarocks install rnn
luarocks install nngraph
luarocks install optim
luarocks install sys
luarocks install hdf5

echo "Installation complete!"
echo "Run 'th' to start Torch REPL"
```

## Verifying Installation

```lua
-- verify_install.lua
require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'
require 'optim'
require 'sys'

print('Torch version: ' .. torch.version())
print('CUDA available: ' .. tostring(cutorch ~= nil))

-- Test basic operations
local x = torch.randn(10)
print('Torch tensor created successfully')

local model = nn.Sequential():add(nn.Linear(10, 5))
print('nn module created successfully')

print('\nAll tests passed! Torch is ready to use.')
```

Run verification:
```bash
th verify_install.lua
```

## Configuration for hypATen

Create a configuration file:

```lua
-- torch_config.lua
local config = {}

-- Paths
config.dataPath = '/path/to/emg/data'
config.modelPath = '/path/to/models'
config.logPath = '/path/to/logs'

-- Training settings
config.learningRate = 1e-3
config.batchSize = 32
config.numEpochs = 100
config.cuda = cutorch ~= nil

-- Model settings
config.inputSize = 8  -- Number of EMG channels
config.hiddenSize = 128
config.numClasses = 10  -- Number of gestures

-- Data settings
config.sampleRate = 1000  -- Hz
config.windowSize = 200   -- ms
config.overlap = 100      -- ms

return config
```

## CUDA Support

If you have NVIDIA GPUs:

```bash
# Install CUDA libraries
luarocks install cutorch
luarocks install cunn

# Verify CUDA
th -e "require 'cutorch'; print(cutorch.getDeviceCount())"
```

## Building Custom Packages

```lua
-- rockspec for hypATen integration
package = "hypaten-torch"
version = "1.0-0"

source = {
   url = "git://github.com/o9nn/hypATen.git",
   tag = "v1.0"
}

description = {
   summary = "Torch integration for hypATen EMG processing",
   detailed = [[
      Integration of Torch neural network libraries with hypATen
      for EMG signal processing and pattern recognition.
   ]],
   homepage = "https://github.com/o9nn/hypATen",
   license = "GPL-3.0"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "rnn >= 1.0",
   "nngraph >= 1.0",
   "optim >= 1.0"
}

build = {
   type = "builtin",
   modules = {}
}
```

## Use Cases in hypATen

1. **Complete Torch environment**: Unified installation of all dependencies
2. **Package management**: Easy installation of additional modules
3. **Version control**: Consistent package versions across deployments
4. **CUDA support**: GPU acceleration for neural networks
5. **Development tools**: REPL and debugging utilities

## References
- Original repository: https://github.com/torch/distro
- Installation guide: http://torch.ch/docs/getting-started.html
- Package list: https://github.com/torch/torch7/wiki/Cheatsheet
