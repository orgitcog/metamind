# torch/sys Integration

## Overview
The sys package provides system utilities for Torch.

**Repository**: https://github.com/torch/sys

## Purpose
Provides essential system-level utilities:
- File system operations
- Process control
- Time and date functions
- System information
- Clock and timing utilities

## Integration with hypATen

System utilities are useful for:
- File I/O for EMG data
- Benchmarking and profiling
- System resource monitoring
- Cross-platform compatibility

## Key Modules

### File System
- `sys.dirp()`: Directory listing
- `sys.filep()`: Check if file exists
- `sys.dirname()`: Get directory name
- `sys.basename()`: Get base file name
- `sys.concat()`: Concatenate paths

### Process Control
- `sys.execute()`: Execute system commands
- `sys.sleep()`: Sleep for specified time
- `sys.exit()`: Exit program

### Timing
- `sys.clock()`: Get current time
- `sys.tic()` / `sys.toc()`: Measure elapsed time

### System Information
- `sys.os()`: Get operating system
- `sys.uname()`: Get system information
- `sys.hostname()`: Get hostname

## Usage Examples

```lua
require 'sys'

-- Timing code execution
sys.tic()
-- ... your code here ...
local elapsed = sys.toc()
print('Elapsed time: ' .. elapsed .. ' seconds')

-- File operations
local files = sys.dirp('/path/to/emg/data')
for _, file in ipairs(files) do
    if sys.filep(file) then
        print('Processing file: ' .. file)
    end
end

-- System information
print('Operating system: ' .. sys.os())
print('Hostname: ' .. sys.hostname())
```

## Data Loading for EMG

```lua
require 'sys'

-- Load EMG data files from directory
function loadEMGData(dataDir)
    local files = sys.dirp(dataDir)
    local data = {}
    
    for _, file in ipairs(files) do
        local fullPath = sys.concat(dataDir, file)
        if sys.filep(fullPath) and file:match('%.dat$') then
            -- Load EMG data file
            local emgData = torch.load(fullPath)
            table.insert(data, emgData)
        end
    end
    
    return data
end
```

## Benchmarking Neural Networks

```lua
require 'sys'

-- Benchmark model inference time
function benchmarkModel(model, input, iterations)
    -- Warmup
    for i = 1, 10 do
        model:forward(input)
    end
    
    -- Actual benchmark
    sys.tic()
    for i = 1, iterations do
        model:forward(input)
    end
    local totalTime = sys.toc()
    
    local avgTime = totalTime / iterations
    local fps = 1.0 / avgTime
    
    print(string.format('Average inference time: %.4f ms', avgTime * 1000))
    print(string.format('Throughput: %.2f samples/second', fps))
    
    return avgTime, fps
end
```

## Real-time Data Acquisition

```lua
require 'sys'

-- Monitor real-time EMG acquisition
function monitorAcquisition(callback, sampleRate)
    local dt = 1.0 / sampleRate
    local lastTime = sys.clock()
    
    while true do
        local currentTime = sys.clock()
        local elapsed = currentTime - lastTime
        
        if elapsed >= dt then
            -- Acquire sample
            local sample = acquireEMGSample()
            callback(sample)
            lastTime = currentTime
        else
            -- Sleep to avoid busy waiting
            sys.sleep(dt - elapsed)
        end
    end
end
```

## Cross-Platform Path Handling

```lua
require 'sys'

-- Platform-independent path construction
function getDataPath(dataset)
    local baseDir = sys.os() == 'windows' and 'C:\\Data\\EMG' or '/data/emg'
    return sys.concat(baseDir, dataset)
end

-- Check if data exists
function checkDataExists(dataset)
    local path = getDataPath(dataset)
    if not sys.dirp(path) then
        error('Dataset directory not found: ' .. path)
    end
    return path
end
```

## Installation

```bash
luarocks install sys
```

## Use Cases in hypATen

1. **Data management**: Load and organize EMG datasets
2. **Performance profiling**: Measure inference time and throughput
3. **Real-time processing**: Implement timing for real-time control
4. **Cross-platform support**: Handle paths and system differences
5. **Logging and monitoring**: Track system resources during experiments

## References
- Original repository: https://github.com/torch/sys
- Torch documentation: http://torch.ch/
