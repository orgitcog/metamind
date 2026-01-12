# Torch Integration Examples

This directory contains example scripts demonstrating the integrated Torch libraries for EMG signal processing in hypATen.

## Prerequisites

Make sure you have Torch installed and all required packages. Run:

```bash
cd ..
./setup.sh
```

Or manually verify installation:

```bash
th verify_installation.lua
```

## Examples

### 1. verify_installation.lua

Verifies that all required Torch packages are properly installed.

```bash
th verify_installation.lua
```

**Purpose**: Ensure your environment is correctly configured before running other examples.

### 2. emg_classifier.lua

Basic feedforward neural network for EMG signal classification.

```bash
th emg_classifier.lua
```

**Features**:
- Multi-layer perceptron architecture
- Dropout regularization
- Training with Adam optimizer
- Model saving/loading

**Use case**: Static gesture classification from EMG feature vectors.

### 3. emg_rnn_classifier.lua

Temporal classifier using LSTM/GRU for sequential EMG data.

```bash
th emg_rnn_classifier.lua
```

**Features**:
- LSTM-based sequence processing
- Bidirectional LSTM option
- Temporal pattern recognition
- Handles variable-length sequences

**Use case**: Dynamic gesture recognition, continuous movement classification.

### 4. multi_channel_fusion.lua

Multi-channel EMG processing with attention mechanisms using nngraph.

```bash
th multi_channel_fusion.lua
```

**Features**:
- Channel-specific processing
- Attention-based fusion
- Graph-based network construction
- Multi-input architecture

**Use case**: Combining signals from multiple electrode sites, adaptive channel weighting.

### 5. hyperparameter_optimization.lua

Hyperparameter search for EMG classifiers (inspired by hypermind).

```bash
th hyperparameter_optimization.lua
```

**Features**:
- Random search
- Grid search
- Automatic hyperparameter tuning
- Performance tracking

**Use case**: Finding optimal model configurations for your dataset.

## Running Examples with Custom Data

### Data Format

The examples use dummy data for demonstration. To use your own EMG data:

1. **Feature vectors** (for `emg_classifier.lua`):
```lua
-- Format: [nSamples x nFeatures] tensor
local myData = torch.load('my_emg_features.t7')
local myLabels = torch.load('my_labels.t7')
```

2. **Sequential data** (for `emg_rnn_classifier.lua`):
```lua
-- Format: table of [seqLength x nChannels] tensors
local mySequences = {}
for i = 1, nSamples do
    mySequences[i] = torch.load(string.format('sequence_%d.t7', i))
end
```

3. **Multi-channel data** (for `multi_channel_fusion.lua`):
```lua
-- Format: table of tables, {channel1, channel2, ..., channelN}
local myMultiChannelData = {}
for i = 1, nSamples do
    myMultiChannelData[i] = {}
    for ch = 1, numChannels do
        myMultiChannelData[i][ch] = torch.load(
            string.format('sample_%d_ch_%d.t7', i, ch))
    end
end
```

## Modifying Examples

### Adjust Network Architecture

In `emg_classifier.lua`:
```lua
local config = {
    inputSize = 16,        -- Change to your feature dimension
    hiddenSize = 256,      -- Adjust hidden layer size
    numClasses = 5,        -- Change to your number of classes
    learningRate = 5e-4,   -- Tune learning rate
}
```

### Use GPU Acceleration

Add CUDA support (requires `cutorch` and `cunn`):

```lua
require 'cutorch'
require 'cunn'

-- Move model to GPU
model = model:cuda()
criterion = criterion:cuda()

-- Move data to GPU
data = data:cuda()
labels = labels:cuda()
```

### Save/Load Models

```lua
-- Save model
torch.save('my_model.t7', model)

-- Load model
local model = torch.load('my_model.t7')
model:evaluate()  -- Set to evaluation mode
```

## Integration with MiniVIE

See `../INTEGRATION_GUIDE.md` for details on integrating these models with the MATLAB-based MiniVIE framework.

Quick example:
```matlab
% In MATLAB
features = extractEMGFeatures(signalData);
save('features.mat', 'features', '-v7');

% Run Torch classifier
system('th ../examples/emg_classifier.lua --test features.mat');
```

## Benchmarking

To benchmark inference time:

```lua
require 'sys'

-- Load model
local model = torch.load('model.t7')
model:evaluate()

-- Sample input
local input = torch.randn(inputSize)

-- Warmup
for i = 1, 100 do
    model:forward(input)
end

-- Benchmark
sys.tic()
for i = 1, 1000 do
    model:forward(input)
end
local elapsed = sys.toc()

print(string.format('Average inference time: %.3f ms', elapsed))
print(string.format('Throughput: %.0f samples/sec', 1000 / elapsed))
```

## Common Issues

### Out of Memory
- Reduce batch size
- Reduce model size (fewer/smaller layers)
- Use gradient checkpointing
- Enable GPU if available

### Slow Training
- Increase batch size
- Use GPU (10-100x speedup)
- Reduce model complexity
- Use FastLSTM instead of LSTM

### Poor Accuracy
- Collect more training data
- Add data augmentation
- Tune hyperparameters (see `hyperparameter_optimization.lua`)
- Try different architectures
- Check for data leakage

## Further Reading

- **nn package**: https://github.com/torch/nn/tree/master/doc
- **rnn package**: https://github.com/torch/rnn
- **nngraph tutorial**: https://github.com/torch/nngraph
- **Optimization**: https://github.com/torch/optim

## Contributing

To add new examples:
1. Follow the existing code structure
2. Include comments explaining key concepts
3. Provide sample output in comments
4. Update this README with your example

## Support

For issues or questions:
- Check `../README.md` for general documentation
- See `../*/README.md` for package-specific documentation
- Open an issue on the GitHub repository
