-- multi_channel_fusion.lua
-- Example multi-channel EMG processing with nngraph

require 'torch'
require 'nn'
require 'nngraph'

-- Configuration
local config = {
    numChannels = 8,      -- Number of EMG channels
    channelSize = 100,    -- Samples per channel
    fusionSize = 64,      -- Fusion layer size
    numClasses = 10       -- Number of output classes
}

print('Multi-Channel EMG Fusion with nngraph')
print('=====================================')
print('')

-- Build channel-specific processing networks
function buildChannelProcessor(channelSize, outputSize)
    local processor = nn.Sequential()
    processor:add(nn.Linear(channelSize, outputSize))
    processor:add(nn.ReLU())
    processor:add(nn.Dropout(0.3))
    return processor
end

-- Build attention mechanism
function buildAttentionModule(inputSize, numChannels)
    local attention = nn.Sequential()
    attention:add(nn.Linear(inputSize * numChannels, numChannels))
    attention:add(nn.SoftMax())
    return attention
end

-- Build multi-channel fusion network with nngraph
function buildFusionNetwork(config)
    -- Create input nodes for each channel
    local channelInputs = {}
    local channelFeatures = {}
    
    print('Building channel-specific processors...')
    for i = 1, config.numChannels do
        channelInputs[i] = nn.Identity()()
        
        -- Process each channel independently
        local processor = buildChannelProcessor(config.channelSize, config.fusionSize)
        channelFeatures[i] = processor(channelInputs[i])
    end
    
    -- Concatenate channel features
    print('Building fusion layer...')
    local concatenated = nn.JoinTable(1)(channelFeatures)
    
    -- Apply attention mechanism
    local attentionWeights = buildAttentionModule(config.fusionSize, config.numChannels)(concatenated)
    
    -- Reshape for element-wise multiplication
    local reshapedFeatures = nn.Reshape(config.numChannels, config.fusionSize)(concatenated)
    local expandedWeights = nn.Replicate(config.fusionSize, 2)(attentionWeights)
    
    -- Apply attention
    local weighted = nn.CMulTable()({reshapedFeatures, expandedWeights})
    local summed = nn.Sum(1)(weighted)
    
    -- Classification head
    local classifier = nn.Sequential()
    classifier:add(nn.Linear(config.fusionSize, 32))
    classifier:add(nn.ReLU())
    classifier:add(nn.Linear(32, config.numClasses))
    classifier:add(nn.LogSoftMax())
    
    local output = classifier(summed)
    
    -- Create the graph module
    local model = nn.gModule(channelInputs, {output})
    
    return model
end

-- Build simpler multi-channel network
function buildSimpleFusionNetwork(config)
    print('Building simple fusion network...')
    
    local channelInputs = {}
    local channelFeatures = {}
    
    -- Process each channel
    for i = 1, config.numChannels do
        channelInputs[i] = nn.Identity()()
        local features = nn.Sequential()
            :add(nn.Linear(config.channelSize, config.fusionSize))
            :add(nn.ReLU())(channelInputs[i])
        channelFeatures[i] = features
    end
    
    -- Concatenate and classify
    local combined = nn.JoinTable(1)(channelFeatures)
    local output = nn.Sequential()
        :add(nn.Linear(config.fusionSize * config.numChannels, 128))
        :add(nn.ReLU())
        :add(nn.Dropout(0.5))
        :add(nn.Linear(128, config.numClasses))
        :add(nn.LogSoftMax())(combined)
    
    local model = nn.gModule(channelInputs, {output})
    
    return model
end

-- Generate multi-channel data
function generateMultiChannelData(nSamples, config)
    local data = {}
    local labels = torch.LongTensor(nSamples)
    
    for i = 1, nSamples do
        data[i] = {}
        for ch = 1, config.numChannels do
            data[i][ch] = torch.randn(config.channelSize)
        end
        labels[i] = torch.random(config.numClasses)
    end
    
    return data, labels
end

-- Test forward pass
function testModel(model, config)
    print('Testing forward pass...')
    
    -- Create sample input
    local sampleInput = {}
    for i = 1, config.numChannels do
        sampleInput[i] = torch.randn(config.channelSize)
    end
    
    -- Forward pass
    model:evaluate()
    local output = model:forward(sampleInput)
    
    print('Input: ' .. config.numChannels .. ' channels × ' .. config.channelSize .. ' samples')
    print('Output: ' .. output:size(1) .. ' classes')
    print('Output probabilities:')
    print(torch.exp(output))
    
    return true
end

-- Visualize network structure
function visualizeNetwork(model, filename)
    local graphviz = require 'graphviz'
    local dotfile = filename or 'fusion_network'
    
    print('Generating network visualization...')
    graph.dot(model.fg, dotfile, dotfile)
    print('Network graph saved to ' .. dotfile .. '.dot')
    print('Convert to image with: dot -Tpng ' .. dotfile .. '.dot -o ' .. dotfile .. '.png')
end

-- Main execution
print('Configuration:')
print(string.format('  Channels: %d', config.numChannels))
print(string.format('  Channel size: %d', config.channelSize))
print(string.format('  Fusion size: %d', config.fusionSize))
print(string.format('  Classes: %d', config.numClasses))
print('')

-- Build and test simple fusion network
local simpleModel = buildSimpleFusionNetwork(config)
print('')
print('Simple fusion model:')
testModel(simpleModel, config)

print('')
print('---')
print('')

-- Build and test attention-based fusion network
local attentionModel = buildFusionNetwork(config)
print('')
print('Attention-based fusion model:')
testModel(attentionModel, config)

print('')
print('Multi-channel fusion networks created successfully!')
print('')
print('These networks can process multiple EMG channels simultaneously')
print('and learn to weight channels based on their importance.')
