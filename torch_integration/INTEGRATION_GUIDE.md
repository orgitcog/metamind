# Integration with MiniVIE

This document describes how to integrate Torch-based neural networks with the existing MiniVIE MATLAB framework.

## Overview

The hypATen project combines:
- **MiniVIE (MATLAB)**: Existing EMG signal acquisition, feature extraction, and pattern recognition
- **Torch Integration**: Deep learning models for advanced classification and temporal processing

## Integration Approaches

### 1. Export MATLAB Features to Torch

Use MATLAB to extract features from EMG signals, then train/infer with Torch models.

#### MATLAB Side (Feature Extraction)
```matlab
% In MiniVIE/+SignalAnalysis/
function features = extractFeaturesForTorch(signalData)
    % Extract time-domain features
    mav = mean(abs(signalData), 1);
    rms = sqrt(mean(signalData.^2, 1));
    var_feat = var(signalData, 0, 1);
    
    % Extract frequency-domain features (optional)
    % fft_feat = computeFFTFeatures(signalData);
    
    % Combine features
    features = [mav, rms, var_feat];
    
    % Save for Torch
    save('emg_features.mat', 'features', '-v7');
end
```

#### Torch Side (Training/Inference)
```lua
-- Load features from MATLAB
require 'torch'
require 'matio'  -- Install with: luarocks install matio

local mat = matio.load('emg_features.mat')
local features = mat.features

-- Use with trained model
local model = torch.load('emg_classifier.t7')
model:evaluate()
local predictions = model:forward(features)
```

### 2. Real-time Integration via File/Socket

#### MATLAB (Data Provider)
```matlab
% Real-time EMG streaming to Torch
function streamToTorch(signalSource, port)
    % Create UDP socket
    udpSocket = udp('127.0.0.1', port);
    fopen(udpSocket);
    
    while signalSource.isRunning()
        % Get EMG data
        emgData = signalSource.getData();
        
        % Extract features
        features = extractFeaturesForTorch(emgData);
        
        % Send to Torch (as binary)
        fwrite(udpSocket, features, 'double');
        
        pause(0.01);  % 100 Hz update rate
    end
    
    fclose(udpSocket);
end
```

#### Torch (Model Server)
```lua
-- Receive EMG features and classify
local socket = require('socket')

-- Create UDP socket
local udp = socket.udp()
udp:setsockname('127.0.0.1', 5000)
udp:settimeout(0)

-- Load model
local model = torch.load('emg_classifier.t7')
model:evaluate()

print('Listening for EMG data on port 5000...')

while true do
    local data, ip, port = udp:receivefrom()
    
    if data then
        -- Convert binary data to tensor
        local features = torch.DoubleTensor(data)
        
        -- Classify
        local output = model:forward(features)
        local _, predicted_class = torch.max(output, 1)
        
        print(string.format('Predicted class: %d', predicted_class[1]))
        
        -- Send back prediction
        udp:sendto(tostring(predicted_class[1]), ip, port)
    end
end
```

### 3. Hybrid MATLAB/Torch Pipeline

Use MATLAB for data acquisition and Torch for model training/inference.

```matlab
% MiniVIE/Utilities/TorchIntegration.m
classdef TorchIntegration < handle
    properties
        modelPath
        featuresPath
        pythonBridge  % Use Python bridge to call Torch
    end
    
    methods
        function obj = TorchIntegration(modelPath)
            obj.modelPath = modelPath;
            obj.featuresPath = 'temp_features.mat';
        end
        
        function prediction = classify(obj, emgData)
            % Extract features
            features = obj.extractFeatures(emgData);
            
            % Save features temporarily
            save(obj.featuresPath, 'features', '-v7');
            
            % Call Torch via system command
            cmd = sprintf('th classify.lua --model %s --features %s', ...
                obj.modelPath, obj.featuresPath);
            [status, result] = system(cmd);
            
            % Parse prediction
            prediction = str2double(result);
        end
        
        function features = extractFeatures(obj, emgData)
            % Use existing MiniVIE feature extraction
            mav = mean(abs(emgData), 1);
            rms = sqrt(mean(emgData.^2, 1));
            features = [mav, rms];
        end
    end
end
```

## Data Format Conventions

### EMG Signal Format
- **Channels**: Rows represent time samples, columns represent channels
- **Sampling Rate**: Typically 1000 Hz
- **Window Size**: 200-300 ms windows with 50-100 ms overlap
- **Normalization**: Zero-mean, unit variance per channel

### Feature Format
```
[MAV_ch1, MAV_ch2, ..., MAV_chN, 
 RMS_ch1, RMS_ch2, ..., RMS_chN,
 VAR_ch1, VAR_ch2, ..., VAR_chN]
```

### Label Format
- Integer class labels: 1, 2, 3, ..., N
- One-hot encoding for neural networks: [0, 1, 0, ...]

## Example Workflow

### Training Pipeline

1. **Data Collection (MATLAB/MiniVIE)**
```matlab
% Collect training data using MiniVIE
trainingData = collectEMGData();
features = extractFeaturesForTorch(trainingData.signals);
labels = trainingData.labels;

% Save for Torch
save('training_data.mat', 'features', 'labels', '-v7');
```

2. **Model Training (Torch)**
```bash
th emg_classifier.lua --train training_data.mat --output model.t7
```

3. **Model Deployment (MATLAB/MiniVIE)**
```matlab
% Use trained model in MiniVIE
torchModel = TorchIntegration('model.t7');

% Real-time classification
while signalSource.isRunning()
    emgData = signalSource.getData();
    prediction = torchModel.classify(emgData);
    updateDisplay(prediction);
end
```

## Performance Considerations

### MATLAB ↔ Torch Communication
- **File-based**: Simple but slower (disk I/O overhead)
- **Socket-based**: Faster, suitable for real-time (< 10 ms latency)
- **Shared memory**: Fastest but more complex to implement

### Batch Processing
- Process multiple windows together for better GPU utilization
- Buffer 10-20 windows before sending to Torch

### Model Optimization
- Use CUDA for GPU acceleration (10-100x speedup)
- Quantize models for faster inference
- Consider converting to ONNX for deployment

## Directory Structure

```
hypATen/
├── MiniVIE/                    # MATLAB framework
│   ├── +PatternRecognition/    # Existing classifiers
│   ├── +SignalAnalysis/        # Feature extraction
│   └── Utilities/
│       └── TorchIntegration.m  # Torch bridge (NEW)
└── torch_integration/          # Torch models
    ├── examples/               # Example scripts
    ├── models/                 # Trained models
    └── utils/                  # Helper functions
```

## Best Practices

1. **Feature Consistency**: Use the same feature extraction in both MATLAB and Torch
2. **Normalization**: Apply consistent normalization (same mean/std)
3. **Validation**: Validate models on held-out data before deployment
4. **Error Handling**: Handle communication failures gracefully
5. **Logging**: Log predictions and errors for debugging

## Troubleshooting

### Issue: Dimension mismatch
- **Cause**: Different feature dimensions between MATLAB and Torch
- **Solution**: Print feature dimensions on both sides, ensure consistency

### Issue: Poor performance
- **Cause**: Different preprocessing or normalization
- **Solution**: Save preprocessing parameters from training, reuse in inference

### Issue: Communication timeout
- **Cause**: Network issues or slow model inference
- **Solution**: Increase timeout, check network connectivity, optimize model

## References

- MiniVIE documentation: See `MiniVIE/` directory
- Torch examples: See `torch_integration/examples/`
- Feature extraction: `MiniVIE/+SignalAnalysis/`
