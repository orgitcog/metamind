# hypATen

**hypATen** integrates advanced deep learning capabilities with the myopen EMG processing platform, combining hyperparameter optimization (hypermind) and tensor operations (ATen) with neural network modules from the Torch ecosystem.

## Project Overview

The Myopen project has expanded to include a number of separate modules: EMG hardware (wired and wireless), EMG software, wireless neural recording hardware, wireless control software, and wired neural recording software. Everything is released under GPL v3.

### Torch Integration

hypATen now includes integrated support for state-of-the-art deep learning through Torch libraries:

- **[hypermind](torch_integration/hypermind/)**: Hyperparameter optimization for neural networks
- **[ATen](torch_integration/aten/)**: High-performance C++ tensor library
- **[torch/nn](torch_integration/nn/)**: Neural network building blocks
- **[torch/rnn](torch_integration/rnn/)**: Recurrent neural networks (LSTM, GRU)
- **[torch/nngraph](torch_integration/nngraph/)**: Graph-based network construction
- **[torch/sys](torch_integration/sys/)**: System utilities
- **[torch/distro](torch_integration/distro/)**: Torch distribution and package management

See the [torch_integration](torch_integration/) directory for detailed documentation and examples.

## Quick Start with Torch

```bash
# Setup Torch integration
cd torch_integration
./setup.sh

# Run examples
th examples/emg_classifier.lua
th examples/emg_rnn_classifier.lua
th examples/multi_channel_fusion.lua
```

## Features

### Deep Learning for EMG
- Multi-layer perceptron classifiers
- LSTM/GRU for temporal pattern recognition
- Multi-channel fusion with attention mechanisms
- Hyperparameter optimization
- GPU acceleration (CUDA support)

### Hardware & Firmware
- Multiple EMG acquisition stages
- Wireless and wired configurations
- Real-time signal processing
- Various client implementations

### MiniVIE Integration
The existing MiniVIE framework in MATLAB continues to provide:
- Pattern recognition and classification
- Signal analysis and feature extraction
- Virtual integration environment
- Real-time prosthetic control

## Original Introduction

OPP is seeking to encourage experimentation with myoelectric control in order to inspire more rapid development of mechatronic prostheses for amputees. Because of the extremely small market size for upper extremity prosthetics, we think that one way to encourage this activity is to develop toys or user customizable devices that are capable of myoelectric signal processing. With wider access and experimentation with the technology, perhaps we could see interesting developments beyond the traditional venues of corporate R&D and educational institution research.

Along with a couple of collaborators (and we'd love to have more), we are exploring the development of an extremely flexible open hardware module that could be used with the Buglabs device, the LEGO NXT platform, and as a USB device that could be used with the OLPC (laptop.org), and traditional linux and windows computers. The components of the device would be modular, so we could populate them as needed, and use the same design for a variety of applications. The USB interface could also be used for connecting to game consoles, such as the Xbox360 and Playstation 3; the I2C bus may be useful for connecting to the Nintendo Wii via the Nunchuck input.

## License

GPL v3

