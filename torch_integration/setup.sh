#!/bin/bash
# Setup script for Torch integration in hypATen

set -e

echo "================================"
echo "hypATen Torch Integration Setup"
echo "================================"
echo ""

# Check if Torch is installed
if ! command -v th &> /dev/null; then
    echo "ERROR: Torch not found."
    echo "Please install Torch7 from: https://github.com/torch/distro"
    echo ""
    echo "Quick install:"
    echo "  git clone https://github.com/torch/distro.git ~/torch --recursive"
    echo "  cd ~/torch && bash install-deps && ./install.sh"
    exit 1
fi

echo "✓ Torch found: $(which th)"
echo ""

# Install required packages
echo "Installing required Torch packages..."
echo ""

packages=(
    "nn"
    "rnn" 
    "nngraph"
    "optim"
    "sys"
    "hdf5"
)

for package in "${packages[@]}"; do
    echo -n "Installing $package... "
    if luarocks install "$package" > /dev/null 2>&1; then
        echo "✓"
    else
        # Package might already be installed
        if luarocks show "$package" > /dev/null 2>&1; then
            echo "✓ (already installed)"
        else
            echo "✗ (failed)"
        fi
    fi
done

echo ""
echo "Checking for CUDA support..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found: $(nvcc --version | grep release)"
    echo -n "Installing CUDA packages... "
    luarocks install cutorch > /dev/null 2>&1 || true
    luarocks install cunn > /dev/null 2>&1 || true
    echo "✓"
else
    echo "⚠ CUDA not found (GPU acceleration unavailable)"
fi

echo ""
echo "Verifying installation..."

# Determine correct path to verification script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/examples/verify_installation.lua" ]; then
    th "$SCRIPT_DIR/examples/verify_installation.lua"
elif [ -f "torch_integration/examples/verify_installation.lua" ]; then
    th torch_integration/examples/verify_installation.lua
elif [ -f "examples/verify_installation.lua" ]; then
    th examples/verify_installation.lua
else
    echo "⚠ Could not find verify_installation.lua, skipping verification"
fi

echo ""
echo "================================"
echo "Installation complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. See torch_integration/README.md for overview"
echo "2. Run examples in torch_integration/examples/"
echo "3. Start Torch REPL with: th"
echo ""
