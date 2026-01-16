# Alternative Trading Platforms for OTC Stock Automation

This document lists trading platforms that support OTC stocks and provide APIs for automated trading.

## Top Platforms for OTC Automated Trading

### 1. **Interactive Brokers (IBKR)** ⭐⭐⭐⭐⭐
- **OTC Support**: ✅ Excellent (supports OTCQX, OTCQB, Pink sheets)
- **API**: ✅ Trader Workstation (TWS) API / IBKR API
- **Python Library**: `ib_insync` or `ibapi`
- **Pros**:
  - Comprehensive OTC market access
  - Low commissions
  - Powerful API with real-time data
  - Supports algorithmic trading
- **Cons**:
  - More complex API setup
  - Requires TWS desktop application or Gateway
- **Docs**: https://interactivebrokers.github.io/tws-api/
- **Python Package**: `pip install ib_insync` or `pip install ibapi`

### 2. **TD Ameritrade (now Charles Schwab)** ⭐⭐⭐⭐
- **OTC Support**: ✅ Good (OTC markets available)
- **API**: ✅ TD Ameritrade API (transitioning to Schwab)
- **Python Library**: `td-ameritrade-python-api`
- **Pros**:
  - Free API access
  - Good documentation
  - OTC stocks accessible
- **Cons**:
  - API is being migrated to Schwab
  - May have some limitations on OTC trades
- **Docs**: https://developer.tdameritrade.com/
- **Status**: Currently transitioning - check Schwab API

### 3. **Charles Schwab** ⭐⭐⭐⭐
- **OTC Support**: ✅ Good (acquired TD Ameritrade)
- **API**: ✅ Schwab API (new)
- **Python Library**: `schwab-py` or direct REST API
- **Pros**:
  - Inherited TD Ameritrade's capabilities
  - Modern API
  - Good OTC access
- **Cons**:
  - Newer API (may have growing pains)
  - Documentation still evolving
- **Docs**: https://developer.schwab.com/

### 4. **TradeStation** ⭐⭐⭐⭐
- **OTC Support**: ✅ Good
- **API**: ✅ TradeStation API / EasyLanguage
- **Python Library**: REST API or EasyLanguage
- **Pros**:
  - Professional-grade platform
  - Excellent OTC support
  - Advanced order types
- **Cons**:
  - More expensive
  - Steeper learning curve
- **Docs**: https://tradestation.github.io/webapi-docs/

### 5. **Fidelity** ⭐⭐⭐
- **OTC Support**: ⚠️ Limited (via Active Trader Pro)
- **API**: ⚠️ Limited public API
- **Python Library**: Limited options
- **Pros**:
  - Large broker
  - Some OTC access
- **Cons**:
  - Limited API for automation
  - Not ideal for programmatic trading
- **Status**: Check Fidelity's developer portal

### 6. **ETRADE** ⭐⭐⭐
- **OTC Support**: ✅ Moderate
- **API**: ✅ ETrade API
- **Python Library**: `python-etrade`
- **Pros**:
  - Free API
  - Good documentation
  - OTC access available
- **Cons**:
  - Being acquired by Morgan Stanley
  - Future API status uncertain
- **Docs**: https://developer.etrade.com/

### 7. **Alpaca** ⭐⭐⭐ (Current Platform)
- **OTC Support**: ⚠️ Limited (many OTC stocks not tradable)
- **API**: ✅ Excellent API
- **Python Library**: `alpaca-trade-api`
- **Pros**:
  - Easy to use API
  - Commission-free
  - Good for screening (Yahoo Finance fallback helps)
- **Cons**:
  - Limited OTC tradability
  - Many OTC stocks marked as non-tradable
- **Status**: **You're currently using this - good for screening, limited for trading**

### 8. **Tradier** ⭐⭐⭐
- **OTC Support**: ⚠️ Limited
- **API**: ✅ Tradier API
- **Python Library**: `tradier-python`
- **Pros**:
  - Low cost
  - REST API
- **Cons**:
  - Limited OTC support
  - Smaller platform
- **Docs**: https://developer.tradier.com/

### 9. **Tastyworks** ⭐⭐
- **OTC Support**: ⚠️ Limited
- **API**: ✅ Tastyworks API
- **Python Library**: `tastyworks-api`
- **Pros**:
  - Options-focused
  - Modern API
- **Cons**:
  - Limited OTC stock focus
- **Docs**: https://tastyworks-api.readthedocs.io/

## Recommendation: Best Options for OTC Trading

### For Maximum OTC Access:
**1. Interactive Brokers (IBKR)** - Best overall for OTC automated trading
   - Best OTC market coverage
   - Powerful API
   - Low commissions
   - Python: `ib_insync` or `ibapi`

### For Ease of Use:
**2. TD Ameritrade / Charles Schwab** - Good balance
   - Free API
   - Good OTC access
   - Easier setup than IBKR
   - Transition period (monitor Schwab migration)

### Current Setup (Screening Only):
**3. Alpaca + Yahoo Finance** - What you have now
   - ✅ Great for screening
   - ⚠️ Limited for actual OTC trading
   - Best for: Finding picks, then trade elsewhere

## Implementation Examples

### Interactive Brokers Example:
```python
from ib_insync import IB, Stock, MarketOrder

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # TWS/Gateway must be running

# Place OTC order
stock = Stock('VXRT', 'SMART', 'USD')  # OTC stocks work with SMART routing
order = MarketOrder('BUY', 100)
trade = ib.placeOrder(stock, order)
```

### TD Ameritrade Example:
```python
from td.client import TDClient

td_client = TDClient(client_id='YOUR_CLIENT_ID', 
                     redirect_uri='http://localhost',
                     credentials_path='td_state.json')

# Authenticate
td_client.login()

# Place OTC order
order = {
    'orderType': 'MARKET',
    'session': 'NORMAL',
    'duration': 'DAY',
    'orderStrategyType': 'SINGLE',
    'orderLegCollection': [{
        'instruction': 'BUY',
        'quantity': 100,
        'instrument': {'symbol': 'VXRT', 'assetType': 'EQUITY'}
    }]
}
response = td_client.place_order(account_id='YOUR_ACCOUNT', order=order)
```

## Migration Strategy

1. **Keep current setup for screening** (Alpaca + Yahoo Finance)
   - Continue using for finding qualified OTC stocks
   - Generates top 10 picks

2. **Set up IBKR for actual trading**
   - Open IBKR account
   - Install TWS or IB Gateway
   - Use `ib_insync` library for automation
   - Trade the qualified stocks from screening

3. **Hybrid approach**
   - Screen with Alpaca/Yahoo Finance (current setup)
   - Export qualified stocks list
   - Trade via IBKR API (or manual review)

## Resources

- **IBKR Python**: https://github.com/erdewit/ib_insync
- **TD Ameritrade API**: https://developer.tdameritrade.com/
- **Schwab API**: https://developer.schwab.com/
- **OTC Market Data**: https://www.otcmarkets.com/

## Next Steps

1. ✅ **Current**: Keep screening with Alpaca + Yahoo Finance (working well)
2. 🔄 **Next**: Set up IBKR account for actual OTC trading
3. 📊 **Future**: Implement dual-platform approach (screen → trade)

## Notes

- **Alpaca** is excellent for **screening** but limited for **trading** OTC
- **Interactive Brokers** offers the best OTC trading capabilities
- Many platforms have OTC access but vary in API quality
- Consider costs: Some platforms charge per trade, others have monthly fees


