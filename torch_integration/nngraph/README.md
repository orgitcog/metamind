# torch/nngraph Integration

## Overview
The nngraph package provides graph-based neural network construction for Torch.

**Repository**: https://github.com/torch/nngraph

## Purpose
Enables construction of complex neural network architectures using a graph-based approach:
- Multi-input, multi-output networks
- Complex connection patterns
- Network visualization
- Flexible architecture design

## Integration with hypATen

nngraph is useful for creating complex neural network architectures for multi-modal EMG processing, combining different signal sources, or creating encoder-decoder architectures.

## Key Features

- **Graph-based construction**: Build networks as computational graphs
- **Multiple inputs/outputs**: Handle multiple data streams
- **Visualization**: Generate graphviz diagrams of network architecture
- **Flexible connections**: Connect any layer to any other layer
- **Module reuse**: Share parameters across different parts of the network

## Usage Example

```lua
require 'nngraph'

-- Simple example: network with two inputs
local input1 = nn.Identity()()
local input2 = nn.Identity()()

-- Process each input separately
local h1 = nn.Linear(20, 10)(input1)
local h2 = nn.Linear(30, 10)(input2)

-- Combine and process
local combined = nn.JoinTable(1)({h1, h2})
local output = nn.Linear(20, 5)(combined)
local finalOutput = nn.Tanh()(output)

-- Create the module
local model = nn.gModule({input1, input2}, {finalOutput})

-- Forward pass
local out = model:forward({torch.randn(20), torch.randn(30)})
```

## Multi-Channel EMG Processing

```lua
require 'nngraph'

-- Multi-channel EMG with attention
local emgChannels = {}
local channelFeatures = {}

-- Process each EMG channel
for i = 1, numChannels do
    emgChannels[i] = nn.Identity()()
    local conv = nn.TemporalConvolution(1, 32, 5)(emgChannels[i])
    local relu = nn.ReLU()(conv)
    local pool = nn.TemporalMaxPooling(2)(relu)
    channelFeatures[i] = nn.Linear(32 * timeSteps, 64)(nn.Reshape(-1)(pool))
end

-- Combine channel features
local combined = nn.JoinTable(1)(channelFeatures)
local attention = nn.Sequential()
    :add(nn.Linear(64 * numChannels, numChannels))
    :add(nn.SoftMax())(combined)

-- Apply attention weights
local weighted = nn.CMulTable()({combined, attention})
local output = nn.Linear(64 * numChannels, numClasses)(weighted)
local classifier = nn.LogSoftMax()(output)

local model = nn.gModule(emgChannels, {classifier})
```

## Complex Architecture: Encoder-Decoder

```lua
-- Encoder-decoder for signal denoising or prediction
require 'nngraph'

-- Encoder
local input = nn.Identity()()
local enc1 = nn.Linear(inputSize, 256)(input)
local enc1_act = nn.ReLU()(enc1)
local enc2 = nn.Linear(256, 128)(enc1_act)
local enc2_act = nn.ReLU()(enc2)
local bottleneck = nn.Linear(128, 64)(enc2_act)

-- Decoder
local dec1 = nn.Linear(64, 128)(bottleneck)
local dec1_act = nn.ReLU()(dec1)
local dec2 = nn.Linear(128, 256)(dec1_act)
local dec2_act = nn.ReLU()(dec2)
local reconstruction = nn.Linear(256, inputSize)(dec2_act)

local autoencoder = nn.gModule({input}, {reconstruction})
```

## Visualization

```lua
-- Save network graph as DOT file for graphviz
graph.dot(model.fg, 'network', 'network_graph')

-- Then convert to image:
-- dot -Tpdf network_graph.dot -o network_graph.pdf
```

## Multi-Modal Integration

```lua
-- Combine EMG signals with other sensor data
require 'nngraph'

local emgInput = nn.Identity()()
local imuInput = nn.Identity()()
local forceInput = nn.Identity()()

-- Process each modality
local emgFeatures = nn.Sequential()
    :add(nn.TemporalConvolution(numEMGChannels, 64, 5))
    :add(nn.ReLU())
    :add(nn.Reshape(-1))
    :add(nn.Linear(64 * timeSteps, 128))(emgInput)

local imuFeatures = nn.Linear(6, 64)(imuInput)  -- 3-axis accel + 3-axis gyro
local forceFeatures = nn.Linear(3, 32)(forceInput)  -- Force sensors

-- Fusion layer
local allFeatures = nn.JoinTable(1)({emgFeatures, imuFeatures, forceFeatures})
local fusion = nn.Linear(128 + 64 + 32, 128)(allFeatures)
local fusionAct = nn.ReLU()(fusion)
local output = nn.Linear(128, numClasses)(fusionAct)
local prediction = nn.LogSoftMax()(output)

local multiModalModel = nn.gModule(
    {emgInput, imuInput, forceInput},
    {prediction}
)
```

## Installation

```bash
luarocks install nngraph
```

## Use Cases in hypATen

1. **Multi-channel EMG fusion**: Combine signals from multiple electrode sites
2. **Multi-modal sensing**: Integrate EMG with IMU, force sensors, etc.
3. **Hierarchical processing**: Build encoder-decoder architectures
4. **Attention mechanisms**: Focus on important channels or time steps
5. **Transfer learning**: Freeze encoder, train decoder

## References
- Original repository: https://github.com/torch/nngraph
- Tutorial: https://github.com/torch/nngraph#nngraph
