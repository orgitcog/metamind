-- hyperparameter_optimization.lua
-- Example hyperparameter optimization for EMG classifiers

require 'torch'
require 'nn'
require 'optim'

print('Hyperparameter Optimization Example')
print('====================================')
print('')
print('This example demonstrates hyperparameter optimization')
print('for EMG signal classification, inspired by hypermind.')
print('')

-- Configuration
local baseConfig = {
    inputSize = 8,
    numClasses = 10,
    numEpochs = 20,
    batchSize = 32
}

-- Hyperparameter search space
local searchSpace = {
    learningRate = {1e-4, 1e-2},
    hiddenSize = {64, 256},
    dropout = {0.2, 0.6},
    numLayers = {1, 3}
}

-- Build model with given hyperparameters
function buildModel(params)
    local model = nn.Sequential()
    
    local inputSize = baseConfig.inputSize
    
    -- Add hidden layers
    for i = 1, params.numLayers do
        model:add(nn.Linear(inputSize, params.hiddenSize))
        model:add(nn.ReLU())
        model:add(nn.Dropout(params.dropout))
        inputSize = params.hiddenSize
    end
    
    -- Output layer
    model:add(nn.Linear(inputSize, baseConfig.numClasses))
    model:add(nn.LogSoftMax())
    
    return model
end

-- Generate dummy training data
function generateData(nSamples)
    local data = torch.randn(nSamples, baseConfig.inputSize)
    local labels = torch.random(nSamples, baseConfig.numClasses)
    return data, labels
end

-- Objective function to minimize (returns validation loss)
function objective(params)
    print(string.format('Testing: lr=%.5f, hidden=%d, dropout=%.2f, layers=%d',
        params.learningRate, params.hiddenSize, params.dropout, params.numLayers))
    
    -- Generate data
    local trainData, trainLabels = generateData(500)
    local valData, valLabels = generateData(100)
    
    -- Build model
    local model = buildModel(params)
    local criterion = nn.ClassNLLCriterion()
    
    local modelParams, gradParams = model:getParameters()
    local optimState = {learningRate = params.learningRate}
    
    -- Quick training
    model:training()
    for epoch = 1, baseConfig.numEpochs do
        local function feval(x)
            gradParams:zero()
            local outputs = model:forward(trainData)
            local loss = criterion:forward(outputs, trainLabels)
            local gradOutputs = criterion:backward(outputs, trainLabels)
            model:backward(trainData, gradOutputs)
            return loss, gradParams
        end
        
        optim.adam(feval, modelParams, optimState)
    end
    
    -- Evaluate on validation set
    model:evaluate()
    local valOutputs = model:forward(valData)
    local valLoss = criterion:forward(valOutputs, valLabels)
    
    -- Calculate accuracy
    local _, predictions = torch.max(valOutputs, 2)
    local accuracy = predictions:squeeze():eq(valLabels):sum() / valLabels:size(1)
    
    print(string.format('  -> Val loss: %.4f, Accuracy: %.2f%%', valLoss, accuracy * 100))
    
    return valLoss, accuracy
end

-- Random search
function randomSearch(searchSpace, numTrials)
    print('Starting random search...')
    print('')
    
    local bestParams = nil
    local bestScore = math.huge
    local results = {}
    
    for trial = 1, numTrials do
        print(string.format('Trial %d/%d', trial, numTrials))
        
        -- Sample random hyperparameters
        local params = {
            learningRate = math.exp(torch.uniform(
                math.log(searchSpace.learningRate[1]),
                math.log(searchSpace.learningRate[2])
            )),
            hiddenSize = torch.random(
                searchSpace.hiddenSize[1],
                searchSpace.hiddenSize[2]
            ),
            dropout = torch.uniform(
                searchSpace.dropout[1],
                searchSpace.dropout[2]
            ),
            numLayers = torch.random(
                searchSpace.numLayers[1],
                searchSpace.numLayers[2]
            )
        }
        
        -- Evaluate
        local loss, accuracy = objective(params)
        
        -- Track results
        table.insert(results, {
            params = params,
            loss = loss,
            accuracy = accuracy
        })
        
        -- Update best
        if loss < bestScore then
            bestScore = loss
            bestParams = params
            print('  *** New best! ***')
        end
        
        print('')
    end
    
    return bestParams, bestScore, results
end

-- Grid search (coarse)
function gridSearch(searchSpace)
    print('Starting grid search...')
    print('')
    
    local bestParams = nil
    local bestScore = math.huge
    local results = {}
    
    local lrValues = {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
    local hiddenValues = {64, 128, 256}
    local dropoutValues = {0.2, 0.4, 0.6}
    local layerValues = {1, 2, 3}
    
    local totalTrials = #lrValues * #hiddenValues * #dropoutValues * #layerValues
    local trialNum = 0
    
    for _, lr in ipairs(lrValues) do
        for _, hidden in ipairs(hiddenValues) do
            for _, dropout in ipairs(dropoutValues) do
                for _, layers in ipairs(layerValues) do
                    trialNum = trialNum + 1
                    print(string.format('Trial %d/%d', trialNum, totalTrials))
                    
                    local params = {
                        learningRate = lr,
                        hiddenSize = hidden,
                        dropout = dropout,
                        numLayers = layers
                    }
                    
                    local loss, accuracy = objective(params)
                    
                    table.insert(results, {
                        params = params,
                        loss = loss,
                        accuracy = accuracy
                    })
                    
                    if loss < bestScore then
                        bestScore = loss
                        bestParams = params
                        print('  *** New best! ***')
                    end
                    
                    print('')
                end
            end
        end
    end
    
    return bestParams, bestScore, results
end

-- Print results
function printResults(bestParams, bestScore, results)
    print('')
    print('Optimization complete!')
    print('======================')
    print('')
    print('Best hyperparameters:')
    print(string.format('  Learning rate: %.5f', bestParams.learningRate))
    print(string.format('  Hidden size: %d', bestParams.hiddenSize))
    print(string.format('  Dropout: %.2f', bestParams.dropout))
    print(string.format('  Num layers: %d', bestParams.numLayers))
    print(string.format('  Best validation loss: %.4f', bestScore))
    print('')
    
    -- Sort results by loss
    table.sort(results, function(a, b) return a.loss < b.loss end)
    
    print('Top 5 configurations:')
    for i = 1, math.min(5, #results) do
        local r = results[i]
        print(string.format('%d. Loss: %.4f, Acc: %.2f%%, lr=%.5f, hidden=%d, dropout=%.2f, layers=%d',
            i, r.loss, r.accuracy * 100,
            r.params.learningRate, r.params.hiddenSize,
            r.params.dropout, r.params.numLayers))
    end
end

-- Main execution
print('Choose optimization method:')
print('1. Random search (10 trials)')
print('2. Grid search (coarse, ~135 trials)')
print('')

local method = 1  -- Default to random search for demo

if method == 1 then
    local bestParams, bestScore, results = randomSearch(searchSpace, 10)
    printResults(bestParams, bestScore, results)
elseif method == 2 then
    local bestParams, bestScore, results = gridSearch(searchSpace)
    printResults(bestParams, bestScore, results)
end

print('')
print('Note: This is a simplified example.')
print('For real hyperparameter optimization, consider using:')
print('  - Cross-validation')
print('  - Early stopping')
print('  - Bayesian optimization')
print('  - More sophisticated search strategies')
