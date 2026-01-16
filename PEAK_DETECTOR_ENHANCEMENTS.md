# OTC Peak Detector - Model Training & Enhancement

## Overview

The OTC Peak Detector has been enhanced with additional technical indicators and a backtesting framework to validate model accuracy on historical data.

## Enhancements Added

### 1. Technical Indicators

#### RSI (Relative Strength Index)
- **Purpose**: Identify overbought (>70) and oversold (<30) conditions
- **Usage**: 
  - Entry: Avoid overbought stocks (RSI 30-70 preferred)
  - Exit: RSI >70 indicates potential peak/reversal
- **Implementation**: `calculate_rsi()` method with 14-day period

#### MACD (Moving Average Convergence Divergence)
- **Purpose**: Detect momentum changes and trend reversals
- **Usage**:
  - Entry: MACD bullish (above signal line) = positive momentum
  - Exit: MACD bearish (below signal line) = momentum weakening
- **Implementation**: `calculate_macd()` method with 12/26/9 parameters

#### Moving Averages (SMA/EMA)
- **Purpose**: Identify trend direction and support/resistance levels
- **Usage**:
  - Entry: Golden cross (SMA20 > SMA50) or price above SMA20 = uptrend
  - Exit: Death cross (SMA20 < SMA50) = downtrend starting
- **Implementation**: `calculate_moving_averages()` with SMA(20, 50) and EMA(12, 26)

### 2. Enhanced Entry Signals

Entry confidence now factors in:
- Volume spike ratio (40% weight)
- Price momentum (30% weight)
- RSI condition (10% weight)
- MACD bullish signal (10% weight)
- Moving average signals (10% weight)

**Entry Criteria**:
- Volume spike ≥ 2× average (within 3 days)
- Positive momentum (>5%) or uptrend
- Not at peak (peak confidence <50%)
- RSI not overbought (30-70 range preferred)
- MACD bullish (optional but preferred)
- Price above SMA20 or golden cross (optional but preferred)

### 3. Enhanced Exit Signals

Exit confidence now factors in:
- Volume divergence (price up, volume down)
- Recent peak detection (scipy find_peaks)
- Volume decline + price drop
- RSI overbought (>70)
- MACD bearish (below signal line)

**Exit Criteria**:
- Peak confidence ≥ 50% (multiple signals align)
- RSI >70 (overbought)
- MACD turning bearish
- Volume divergence (price up, volume down)
- Price dropping with declining volume

## Backtesting Framework

### Files Created

1. **`backtest_peak_detector.py`**: Full backtesting framework
   - `validate_peak_detection()`: Validates peak detection accuracy
   - `backtest_symbol()`: Simulates trading on historical data
   - `test_parameters()`: Tests different parameter combinations

2. **`run_backtest.py`**: Quick validation runner

### Usage

```bash
# Quick validation
python3 run_backtest.py

# Full backtest with custom symbols
python3 backtest_peak_detector.py --symbols VXRT SYMBOL2 --backtest --validate

# Test different parameters
python3 backtest_peak_detector.py --test-params --symbols VXRT
```

### Metrics Tracked

- **Win Rate**: Percentage of profitable trades
- **Average Profit**: Mean profit per trade
- **Total Profit**: Sum of all trade profits
- **Max Profit/Loss**: Best and worst trade outcomes
- **Hold Days**: Average holding period
- **Detection Accuracy**: How often peaks are correctly identified

## Parameter Tuning

### Current Parameters

```python
detector = OTCPeakDetector(
    volume_threshold_multiplier=2.0,  # Volume must be 2× average
    price_momentum_days=3,            # Measure momentum over 3 days
    peak_detection_window=10,         # 10-day window for peak detection
    min_peak_prominence=0.05,         # 5% minimum peak prominence
    use_rsi=True,                     # Enable RSI indicator
    use_macd=True,                    # Enable MACD indicator
    use_ma=True                       # Enable Moving Averages
)
```

### Tuning Recommendations

1. **Volume Threshold** (2.0 - 3.0):
   - Lower (2.0): More trades, may catch more pumps
   - Higher (3.0): Fewer trades, only strong pumps

2. **Momentum Days** (3 - 7):
   - Shorter (3): Faster entry, may miss trends
   - Longer (7): Slower entry, more confirmation

3. **Peak Prominence** (0.05 - 0.10):
   - Lower (0.05): Detect smaller peaks
   - Higher (0.10): Only major peaks

4. **RSI Thresholds**:
   - Entry: 30-70 (not overbought/oversold)
   - Exit: >70 (overbought)

## Model Validation Results

### Current Status
- **Peak Detection**: 0-20% accuracy (needs improvement)
- **Entry Signals**: Working, but may be too conservative
- **Exit Signals**: Need better timing

### Areas for Improvement

1. **Historical Data Slicing**: 
   - Current: Uses latest data from yfinance
   - Needed: Use historical slices for true backtesting

2. **Parameter Optimization**:
   - Test different threshold combinations
   - Find optimal RSI/MACD/MA weights
   - Adjust peak detection sensitivity

3. **Signal Timing**:
   - Reduce false positives (entry too early)
   - Improve exit timing (capture more profit)
   - Add trailing stop loss

4. **Additional Indicators to Consider**:
   - **Bollinger Bands**: Volatility-based entry/exit
   - **Stochastic Oscillator**: Momentum confirmation
   - **Volume Profile**: Identify support/resistance levels
   - **Support/Resistance Levels**: Key price levels

## Testing Checklist

- [x] RSI calculation and integration
- [x] MACD calculation and integration
- [x] Moving averages (SMA/EMA) integration
- [x] Enhanced entry signal logic
- [x] Enhanced exit signal logic
- [x] Backtesting framework
- [ ] Historical data slicing for true backtesting
- [ ] Parameter optimization
- [ ] Multi-symbol backtesting
- [ ] Performance metrics dashboard

## Next Steps

1. **Improve Backtesting**: Implement proper historical data slicing
2. **Parameter Optimization**: Use grid search to find best parameters
3. **Add More Indicators**: Bollinger Bands, Stochastic, etc.
4. **Machine Learning**: Consider ML models for pattern recognition
5. **Real-time Validation**: Compare predictions with actual outcomes

## Example: Using Enhanced Detector

```python
from otc_peak_detector import OTCPeakDetector

# Initialize with all indicators enabled
detector = OTCPeakDetector(
    volume_threshold_multiplier=2.0,
    use_rsi=True,
    use_macd=True,
    use_ma=True
)

# Get entry signal
entry = detector.get_entry_signal('VXRT')
print(f"Entry: {entry['signal']} ({entry['confidence']:.0%})")
print(f"  RSI: {entry.get('rsi', 'N/A')}")
print(f"  MACD: {entry.get('macd_bullish', 'N/A')}")

# Get exit signal
exit_sig = detector.get_exit_signal('VXRT', entry_price=0.50)
print(f"Exit: {exit_sig['signal']} ({exit_sig['confidence']:.0%})")
```

## References

- **RSI**: https://www.investopedia.com/terms/r/rsi.asp
- **MACD**: https://www.investopedia.com/terms/m/macd.asp
- **Moving Averages**: https://www.investopedia.com/terms/m/movingaverage.asp
- **Peak Detection**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

