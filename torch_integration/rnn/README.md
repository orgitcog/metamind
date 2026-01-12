# torch/rnn Integration

## Overview
The rnn package provides recurrent neural network modules for Torch.

**Repository**: https://github.com/torch/rnn

## Purpose
Specialized modules for sequential data processing:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- RNN (Vanilla Recurrent Neural Networks)
- Sequence processing utilities
- Attention mechanisms

## Integration with hypATen

RNN modules are particularly useful for temporal EMG signal processing where the sequential nature of the data is important for pattern recognition.

## Key Modules

### Recurrent Layers
- `nn.LSTM`: Long Short-Term Memory
- `nn.GRU`: Gated Recurrent Unit
- `nn.FastLSTM`: Optimized LSTM implementation
- `nn.Recurrent`: Generic recurrent layer
- `nn.SeqLSTM`: Sequence-to-sequence LSTM

### Sequence Processing
- `nn.Sequencer`: Apply module to each element in sequence
- `nn.BiSequencer`: Bidirectional sequence processing
- `nn.RecurrentAttention`: Attention mechanism

### Utilities
- `nn.NormStabilizer`: Normalize gradients
- `nn.TrimZero`: Remove zero-padded elements

## Usage Example

```lua
require 'rnn'

-- Build LSTM network for sequence classification
local model = nn.Sequential()
   :add(nn.FastLSTM(inputSize, hiddenSize))
   :add(nn.FastLSTM(hiddenSize, hiddenSize))
   :add(nn.Select(1, -1))  -- Select last time step
   :add(nn.Linear(hiddenSize, numClasses))
   :add(nn.LogSoftMax())

-- Use Sequencer for variable length sequences
local seqModel = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(inputSize, hiddenSize))
      :add(nn.Linear(hiddenSize, numClasses))
      :add(nn.LogSoftMax())
)
```

## EMG Temporal Pattern Recognition

```lua
-- LSTM-based EMG gesture classifier
local gestureClassifier = nn.Sequential()

-- Add bidirectional LSTM for better temporal context
local fwd = nn.FastLSTM(numChannels, 128)
local bwd = nn.FastLSTM(numChannels, 128)
local biLSTM = nn.BiSequencer(fwd, bwd)

gestureClassifier:add(biLSTM)
gestureClassifier:add(nn.JoinTable(2))  -- Concatenate forward and backward
gestureClassifier:add(nn.Sequencer(nn.Linear(256, 64)))
gestureClassifier:add(nn.Select(1, -1))  -- Last time step
gestureClassifier:add(nn.Linear(64, numGestures))
gestureClassifier:add(nn.LogSoftMax())
```

## Advanced: Attention for EMG

```lua
-- Attention-based model for important time steps
require 'rnn'

local model = nn.Sequential()
local rnn = nn.FastLSTM(inputSize, hiddenSize)
model:add(nn.Sequencer(rnn))
model:add(nn.RecurrentAttention(hiddenSize, numClasses))
```

## Installation

```bash
luarocks install rnn
```

## Use Cases in hypATen

1. **Temporal gesture recognition**: Classify hand gestures from EMG signal sequences
2. **Movement prediction**: Predict future movements from signal history
3. **Context-aware control**: Use temporal context for better prosthetic control
4. **Adaptive filtering**: Learn temporal patterns for noise reduction

## References
- Original repository: https://github.com/torch/rnn
- Element Research RNN tutorial: https://github.com/Element-Research/rnn
