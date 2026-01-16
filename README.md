# OTC Market Stock Trading Strategy

A comprehensive Google Colab notebook for automated OTC (Over-The-Counter) stock trading using the Alpaca API.

## Strategy Overview

The bot implements the following automated trading strategy:

- **Price Range**: Screens stocks priced between $0.30 - $2.00
- **Volume Activity**: Identifies stocks with increasing volume over 3 consecutive days
- **Selection**: Selects up to 3 best candidates daily
- **Buy Strategy**: Buys $1000 worth daily for 5 consecutive days at market price
- **Sell Strategy**: Automatically sells all shares when price reaches 50% above average cost

## Features

- ✅ Automated stock screening based on price and volume criteria
- ✅ Multi-day accumulation strategy (5 days of buying)
- ✅ Automatic profit-taking at 50% gain
- ✅ Position tracking and monitoring
- ✅ Buy schedule persistence across notebook sessions
- ✅ Real-time account and portfolio monitoring

## Setup Instructions

### 1. Open in Google Colab

1. Upload the `otc_trading_strategy.ipynb` file to Google Colab
2. Or use the direct link to open it in Colab

### 2. Configure API Keys

The notebook already contains your Alpaca API credentials:
- **Endpoint**: `https://paper-api.alpaca.markets/v2`
- **API Key**: Already configured
- **Secret**: Already configured

⚠️ **Note**: These are paper trading credentials. For live trading, you'll need to update them.

### 3. Install Dependencies

Run the first cell to install required packages:
```python
!pip install alpaca-trade-api pandas numpy -q
```

### 4. Initialize and Run

1. Run all cells sequentially to load functions
2. The notebook will automatically:
   - Check existing positions for profit targets
   - Process daily buy orders for stocks in schedule
   - Screen for new stock candidates
   - Execute trades based on the strategy

## Usage

### Daily Execution

Simply run the main execution cell:
```python
daily_screen_and_trade()
```

### Custom OTC Symbol List

Since Alpaca's OTC coverage may vary, you can provide your own list:
```python
daily_screen_and_trade(['SYMBOL1', 'SYMBOL2', 'SYMBOL3'])
```

### Monitoring

Use the monitoring cells to:
- View account status and buying power
- Check current positions and P/L
- View active buy schedules

## How It Works

### 1. Screening Process

For each OTC symbol:
- ✅ Check if price is between $0.30 - $2.00
- ✅ Verify volume has increased by at least 50% over 3 days
- ✅ Rank by volume change percentage

### 2. Buy Strategy

When a stock qualifies:
- Day 1-5: Buy $1000 worth at market price each day
- Tracks average cost across all purchases
- Schedule persists across sessions via `buy_schedules.json`

### 3. Sell Strategy

Automatic selling when:
- Stock price reaches 50% above average entry cost
- Sells entire position at once
- Removes from buy schedule tracking

### 4. Position Management

- Maximum 3 active stocks at a time
- Skips new candidates if already at capacity
- Prevents duplicate positions in same stock

## Important Notes

⚠️ **Alpaca OTC Coverage**: Alpaca's OTC stock coverage may be limited. The notebook attempts to auto-detect OTC symbols, but you may need to manually provide a list of OTC symbols available on Alpaca.

⚠️ **Paper Trading**: The current configuration uses Alpaca's paper trading environment. This is recommended for testing.

⚠️ **Risk Management**: 
- This is an automated trading strategy. Always monitor your positions.
- OTC stocks can be highly volatile and illiquid.
- Set appropriate position limits based on your risk tolerance.

⚠️ **Market Hours**: OTC markets may have different trading hours than regular markets. The bot will execute during market hours only.

## File Structure

- `otc_trading_strategy.ipynb` - Main Colab notebook
- `buy_schedules.json` - Auto-generated file to track buy schedules (created on first run)

## Customization

You can adjust strategy parameters at the top of the notebook:

```python
MIN_PRICE = 0.3              # Minimum stock price
MAX_PRICE = 2.0              # Maximum stock price
DAILY_PURCHASE_AMOUNT = 1000 # Amount to buy per day
BUY_DAYS = 5                 # Number of days to buy
VOLUME_ACTIVITY_DAYS = 3     # Days to check for volume increase
PROFIT_TARGET_PCT = 50       # Profit percentage to trigger sell
MAX_STOCKS_PER_DAY = 3       # Maximum stocks to select daily
```

## Troubleshooting

**"No OTC stocks found"**
- Alpaca may have limited OTC coverage
- Provide a manual list of OTC symbols to screen

**"Cannot get price for symbol"**
- Symbol may not be available on Alpaca
- Check if symbol is tradable on the platform

**"Order rejected"**
- Check account buying power
- Verify symbol is tradable
- Ensure market is open

## Support

For issues with:
- **Alpaca API**: Check [Alpaca Documentation](https://alpaca.markets/docs/)
- **Strategy Logic**: Review the code comments in the notebook

## Disclaimer

This is an automated trading strategy for educational purposes. Trading stocks involves risk, especially in OTC markets. Always:
- Test thoroughly in paper trading first
- Monitor your positions regularly
- Never invest more than you can afford to lose
- Consider consulting with a financial advisor

