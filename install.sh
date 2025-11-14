#!/bin/bash
# Quick installation script for tiny_corpus_prep using UV

set -e

echo "=========================================="
echo "tiny_corpus_prep Installation Script"
echo "=========================================="
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell configuration to get UV in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    echo ""
    echo "✓ UV installed successfully"
    echo ""
else
    echo "✓ UV is already installed"
    echo ""
fi

# Check UV version
echo "UV version:"
uv --version
echo ""

# Ask user which installation type
echo "Select installation type:"
echo "  1) Full installation (with Gemini annotator support)"
echo "  2) Core only (without Gemini)"
echo ""
read -p "Enter choice [1-2] (default: 1): " choice
choice=${choice:-1}

echo ""
echo "Installing tiny_corpus_prep..."

if [ "$choice" = "1" ]; then
    echo "Installing with annotator support..."
    uv pip install -e ".[annotators]"
else
    echo "Installing core only..."
    uv pip install -e .
fi

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test: python test_refactored.py"
echo "  2. Examples: python examples/basic_usage.py"
echo "  3. Docs: cat README.md"
echo ""
echo "For Gemini annotator, set your API key:"
echo "  export GOOGLE_API_KEY='your_key_here'"
echo ""

