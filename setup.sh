#!/bin/bash
# Setup script for OTC Trading Bot on AWS EC2

echo "Setting up OTC Trading Bot on AWS EC2..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $(python3 --version)"

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

# Make script executable
chmod +x otc_trading_bot.py

# Create necessary directories
mkdir -p logs
mkdir -p data

echo ""
echo "Setup complete!"
echo ""
echo "To run the bot:"
echo "  source venv/bin/activate  # Activate virtual environment"
echo "  python3 otc_trading_bot.py  # Run the bot"
echo ""
echo "To view options:"
echo "  python3 otc_trading_bot.py --help"
echo ""

