-- emg_classifier.lua
-- Example EMG signal classifier using nn modules

require 'torch'
require 'nn'
require 'optim'

-- Configuration
local config = {
    inputSize = 8,        -- Number of EMG channels
    hiddenSize = 128,     -- Hidden layer size
    numClasses = 10,      -- Number of gestures/classes
    learningRate = 1e-3,
    batchSize = 32,
    numEpochs = 100
}

-- Build the classifier model
function buildClassifier(config)
    local model = nn.Sequential()
    
    -- Input layer
    model:add(nn.Linear(config.inputSize, config.hiddenSize))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5))
    
    -- Hidden layer
    model:add(nn.Linear(config.hiddenSize, math.floor(config.hiddenSize / 2)))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.3))
    
    -- Output layer
    model:add(nn.Linear(math.floor(config.hiddenSize / 2), config.numClasses))
    model:add(nn.LogSoftMax())
    
    return model
end

-- Loss function
local criterion = nn.ClassNLLCriterion()

-- Build model
local model = buildClassifier(config)

print('EMG Classifier Model:')
print(model)
print('')

-- Generate dummy data for demonstration
function generateDummyData(nSamples, config)
    local data = torch.randn(nSamples, config.inputSize)
    local labels = torch.random(nSamples, config.numClasses)
    return data, labels
end

-- Training function
function train(model, criterion, data, labels, config)
    model:training()
    
    local parameters, gradParameters = model:getParameters()
    
    local optimState = {
        learningRate = config.learningRate
    }
    
    for epoch = 1, config.numEpochs do
        local epochLoss = 0
        local numBatches = math.floor(data:size(1) / config.batchSize)
        
        for i = 1, numBatches do
            local batchStart = (i - 1) * config.batchSize + 1
            local batchEnd = i * config.batchSize
            local batch = data[{{batchStart, batchEnd}}]
            local batchLabels = labels[{{batchStart, batchEnd}}]
            
            local function feval(params)
                gradParameters:zero()
                
                local outputs = model:forward(batch)
                local loss = criterion:forward(outputs, batchLabels)
                local gradOutputs = criterion:backward(outputs, batchLabels)
                model:backward(batch, gradOutputs)
                
                return loss, gradParameters
            end
            
            local _, loss = optim.adam(feval, parameters, optimState)
            epochLoss = epochLoss + loss[1]
        end
        
        epochLoss = epochLoss / numBatches
        
        if epoch % 10 == 0 then
            print(string.format('Epoch %d/%d - Loss: %.4f', 
                epoch, config.numEpochs, epochLoss))
        end
    end
end

-- Evaluation function
function evaluate(model, data, labels)
    model:evaluate()
    
    local outputs = model:forward(data)
    local _, predictions = torch.max(outputs, 2)
    predictions = predictions:squeeze()
    
    local correct = predictions:eq(labels):sum()
    local accuracy = correct / data:size(1)
    
    return accuracy
end

-- Main execution
print('Generating dummy EMG data...')
local trainData, trainLabels = generateDummyData(1000, config)
local testData, testLabels = generateDummyData(200, config)

print('Training classifier...')
train(model, criterion, trainData, trainLabels, config)

print('')
print('Evaluating on test data...')
local accuracy = evaluate(model, testData, testLabels)
print(string.format('Test accuracy: %.2f%%', accuracy * 100))

-- Save model
print('')
print('Saving model...')
torch.save('emg_classifier.t7', model)
print('Model saved to emg_classifier.t7')
