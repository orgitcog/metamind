-- emg_rnn_classifier.lua
-- Example temporal EMG classifier using RNN modules

require 'torch'
require 'nn'
require 'rnn'
require 'optim'

-- Configuration
local config = {
    inputSize = 8,          -- Number of EMG channels
    hiddenSize = 128,       -- LSTM hidden size
    numClasses = 10,        -- Number of gestures
    seqLength = 50,         -- Sequence length (time steps)
    learningRate = 1e-3,
    batchSize = 16,
    numEpochs = 50
}

-- Build LSTM-based temporal classifier
function buildTemporalClassifier(config)
    local model = nn.Sequential()
    
    -- LSTM layers
    model:add(nn.FastLSTM(config.inputSize, config.hiddenSize))
    model:add(nn.FastLSTM(config.hiddenSize, config.hiddenSize))
    
    -- Select last time step
    model:add(nn.Select(1, -1))
    
    -- Classification head
    model:add(nn.Linear(config.hiddenSize, 64))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.3))
    model:add(nn.Linear(64, config.numClasses))
    model:add(nn.LogSoftMax())
    
    return model
end

-- Build bidirectional LSTM classifier
function buildBidirectionalClassifier(config)
    -- Forward and backward LSTMs
    local fwd = nn.Sequential()
        :add(nn.FastLSTM(config.inputSize, config.hiddenSize))
        :add(nn.Select(1, -1))
    
    local bwd = nn.Sequential()
        :add(nn.FastLSTM(config.inputSize, config.hiddenSize))
        :add(nn.Select(1, -1))
    
    -- Combine bidirectional outputs
    local model = nn.Sequential()
    model:add(nn.ConcatTable()
        :add(fwd)
        :add(bwd))
    model:add(nn.JoinTable(2))
    
    -- Classification head
    model:add(nn.Linear(config.hiddenSize * 2, 64))
    model:add(nn.ReLU())
    model:add(nn.Linear(64, config.numClasses))
    model:add(nn.LogSoftMax())
    
    return model
end

-- Generate dummy sequential data
function generateSequentialData(nSamples, config)
    -- Generate sequences: [seqLength x inputSize]
    local data = {}
    local labels = torch.LongTensor(nSamples)
    
    for i = 1, nSamples do
        data[i] = torch.randn(config.seqLength, config.inputSize)
        labels[i] = torch.random(config.numClasses)
    end
    
    return data, labels
end

-- Training function for sequential data
function trainSequential(model, criterion, data, labels, config)
    model:training()
    
    local parameters, gradParameters = model:getParameters()
    
    local optimState = {
        learningRate = config.learningRate
    }
    
    print('Training temporal classifier...')
    
    for epoch = 1, config.numEpochs do
        local epochLoss = 0
        local shuffled = torch.randperm(#data)
        
        for i = 1, #data, config.batchSize do
            local batchSize = math.min(config.batchSize, #data - i + 1)
            local loss = 0
            
            for j = 1, batchSize do
                local idx = shuffled[i + j - 1]
                
                local function feval(params)
                    gradParameters:zero()
                    
                    local output = model:forward(data[idx])
                    local sampleLoss = criterion:forward(output, labels[idx])
                    local gradOutput = criterion:backward(output, labels[idx])
                    model:backward(data[idx], gradOutput)
                    
                    return sampleLoss, gradParameters
                end
                
                local _, batchLoss = optim.adam(feval, parameters, optimState)
                loss = loss + batchLoss[1]
            end
            
            epochLoss = epochLoss + loss / batchSize
        end
        
        epochLoss = epochLoss / math.ceil(#data / config.batchSize)
        
        if epoch % 5 == 0 then
            print(string.format('Epoch %d/%d - Loss: %.4f', 
                epoch, config.numEpochs, epochLoss))
        end
    end
end

-- Evaluation function
function evaluateSequential(model, data, labels)
    model:evaluate()
    
    local correct = 0
    
    for i = 1, #data do
        local output = model:forward(data[i])
        local _, prediction = torch.max(output, 1)
        if prediction[1] == labels[i] then
            correct = correct + 1
        end
    end
    
    return correct / #data
end

-- Main execution
print('EMG Temporal Classifier with RNN')
print('================================')
print('')

-- Build model
local model = buildTemporalClassifier(config)
print('Model architecture:')
print(model)
print('')

-- Loss function
local criterion = nn.ClassNLLCriterion()

-- Generate data
print('Generating temporal EMG data...')
local trainData, trainLabels = generateSequentialData(500, config)
local testData, testLabels = generateSequentialData(100, config)
print(string.format('Training samples: %d', #trainData))
print(string.format('Test samples: %d', #testData))
print('')

-- Train
trainSequential(model, criterion, trainData, trainLabels, config)

-- Evaluate
print('')
print('Evaluating on test data...')
local accuracy = evaluateSequential(model, testData, testLabels)
print(string.format('Test accuracy: %.2f%%', accuracy * 100))

-- Save model
print('')
print('Saving model...')
torch.save('emg_rnn_classifier.t7', model)
print('Model saved to emg_rnn_classifier.t7')

print('')
print('Example: Using bidirectional LSTM')
local biModel = buildBidirectionalClassifier(config)
print('Bidirectional model created.')
