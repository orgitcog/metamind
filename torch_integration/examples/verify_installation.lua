-- verify_installation.lua
-- Verify that all required Torch packages are installed

require 'torch'

local packages = {
    'nn',
    'rnn',
    'nngraph',
    'optim',
    'sys'
}

local optional = {
    'cutorch',
    'cunn',
    'hdf5'
}

print('Verifying Torch installation for hypATen...')
print('')
print('Torch version: ' .. torch.version())
print('')

print('Required packages:')
local allRequired = true
for _, pkg in ipairs(packages) do
    local status, _ = pcall(require, pkg)
    if status then
        print('  ✓ ' .. pkg)
    else
        print('  ✗ ' .. pkg .. ' (MISSING)')
        allRequired = false
    end
end

print('')
print('Optional packages:')
for _, pkg in ipairs(optional) do
    local status, _ = pcall(require, pkg)
    if status then
        print('  ✓ ' .. pkg)
    else
        print('  ⚠ ' .. pkg .. ' (not installed)')
    end
end

print('')

if allRequired then
    -- Test basic functionality
    print('Testing basic functionality...')
    
    -- Test tensor operations
    local x = torch.randn(10, 5)
    local y = torch.randn(10, 5)
    local z = x + y
    print('  ✓ Tensor operations')
    
    -- Test nn module
    local model = nn.Sequential()
    model:add(nn.Linear(10, 5))
    model:add(nn.ReLU())
    local input = torch.randn(10)
    local output = model:forward(input)
    print('  ✓ Neural network forward pass')
    
    -- Test RNN
    local lstm = nn.FastLSTM(10, 20)
    local rnnInput = torch.randn(10)
    local rnnOutput = lstm:forward(rnnInput)
    print('  ✓ RNN module')
    
    -- Test nngraph
    local input1 = nn.Identity()()
    local input2 = nn.Identity()()
    local combined = nn.JoinTable(1)({input1, input2})
    local graphModel = nn.gModule({input1, input2}, {combined})
    local out = graphModel:forward({torch.randn(5), torch.randn(5)})
    print('  ✓ nngraph')
    
    print('')
    print('All tests passed! ✓')
    print('')
    print('Torch is properly configured for hypATen.')
else
    print('ERROR: Some required packages are missing.')
    print('Please run: luarocks install <package>')
    os.exit(1)
end
