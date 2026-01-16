#!/bin/bash
# Daily runner script for OTC Trading Bot
# This script should be called by cron

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the bot
# You can specify symbols file or pass symbols directly
python3 otc_trading_bot.py

# If you have a symbols file, uncomment the line below and update the path:
# python3 otc_trading_bot.py --symbols-file otc_symbols.txt

# Exit with the bot's exit code
exit $?

