# hypermind Integration

## Overview
Hypermind is a hyperparameter optimization framework for Torch neural networks.

**Repository**: https://github.com/nicholas-leonard/hypermind

## Purpose
Provides hyperparameter search and optimization capabilities for neural network models, particularly useful for:
- Finding optimal learning rates
- Network architecture search
- Regularization parameter tuning
- Optimization algorithm selection

## Integration with hypATen

Hypermind can be used to optimize the neural network models used in EMG signal classification and pattern recognition within the MiniVIE framework.

## Key Features
- Random search
- Grid search
- Bayesian optimization
- Early stopping
- Cross-validation support

## Usage Example

```lua
local hypermind = require 'hypermind'

-- Define hyperparameter search space
local space = {
    learningRate = {1e-4, 1e-1, 'log'},
    hiddenSize = {64, 512, 'int'},
    dropout = {0.0, 0.5}
}

-- Define objective function
local function objective(params)
    local model = createModel(params)
    local accuracy = trainAndEvaluate(model)
    return accuracy
end

-- Run optimization
local best = hypermind.optimize(objective, space, {maxEvals = 100})
```

## Installation

```bash
luarocks install hypermind
```

## References
- Original repository: https://github.com/nicholas-leonard/hypermind
- Related to hyperopt (Python) and similar hyperparameter optimization frameworks
