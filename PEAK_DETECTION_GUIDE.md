# OTC Peak Detection Algorithm Guide

## Overview

The OTC Peak Detection Algorithm is designed to identify optimal entry and exit points for OTC stocks based on historical market patterns:

### OTC Market Pattern (Pump & Dump Cycle):
1. **Pump Phase**: Volume spikes → Price momentum builds
2. **Peak Formation**: Price reaches high, volume diverges (declines while price up)
3. **Dump Phase**: Price declines → Volume decreases

## Algorithm Features

### Entry Signals (Buy Timing)
- **Volume Spike Detection**: Identifies when volume is 2×+ average (pump starting)
- **Price Momentum**: Measures rate of price increase
- **Early Entry**: Catches the pump phase before peak
- **Confidence Score**: 0-100% confidence in entry timing

**Entry Conditions:**
- ✅ Volume spike within last 3 days
- ✅ Positive price momentum (5%+ gain)
- ✅ Not yet at peak
- ✅ High confidence score

### Exit Signals (Sell Timing - Peak Detection)
- **Volume Divergence**: Price up but volume decreasing (peak signal)
- **Peak Detection**: Uses scipy.signal to find local price peaks
- **Reversal Risk**: Measures probability of price decline
- **Dump Detection**: Volume decline + price drop (dump starting)

**Exit Conditions:**
- ✅ Peak detected with high confidence (70%+)
- ✅ Volume divergence confirmed
- ✅ Reversal risk high
- ✅ Profit target reached (50% or 30% if peak detected)

## How It Works

### 1. Volume Spike Detection
```python
# Compares recent volume (5-day avg) to 30-day average
volume_ratio = recent_volume / 30day_average
has_spike = volume_ratio >= 2.0  # 2× threshold
```

### 2. Price Momentum Analysis
```python
# Measures price change over 3-5 days
momentum = (current_price - start_price) / start_price * 100
trend = 'up' if momentum > 5% and consistent upward movement
```

### 3. Peak Detection
```python
# Uses scipy.signal.find_peaks to detect local maxima
peaks = find_peaks(prices, prominence=5%, distance=3 days)
# Checks if current price is near a detected peak
```

### 4. Volume Divergence (Key Peak Signal)
```python
# Price trend vs Volume trend
price_trend = price increasing
volume_trend = volume decreasing
divergence = price_trend > 0 AND volume_trend < 0  # Peak signal!
```

### 5. Reversal Risk Calculation
```python
# Combines multiple signals:
- Volume divergence (30% weight)
- Peak proximity (30% weight)  
- Dump signals (40% weight)
reversal_risk = weighted_average(all_signals)
```

## Integration with Trading Bot

### Screening Phase
- Each qualified stock gets a `peak_score` (0-1)
- Higher score = better entry timing (catching pump early)
- Stocks sorted by peak_score first, then volume, then performance

### Trading Phase
- `check_profit_targets()` uses peak detection for exit signals
- Sells when peak detected (even if profit < 50%)
- Catches peaks before dump phase starts

## Usage Examples

### Test Peak Detection on a Symbol
```bash
python3 otc_peak_detector.py VXRT
```

### Use in Trading Bot
The bot automatically uses peak detection:
1. During screening: Ranks stocks by entry timing (peak_score)
2. During monitoring: Checks for exit signals (peak detected)

### Manual Analysis
```python
from otc_peak_detector import analyze_otc_peak

# Analyze entry signal
analysis = analyze_otc_peak('VXRT')
print(f"Entry: {analysis['entry_signal']['signal']}")
print(f"Confidence: {analysis['entry_signal']['confidence']:.0%}")

# Analyze exit signal (with entry price)
analysis = analyze_otc_peak('VXRT', entry_price=0.50)
print(f"Exit: {analysis['exit_signal']['signal']}")
print(f"Reason: {analysis['exit_signal']['reason']}")
```

## Algorithm Parameters

You can tune these in `otc_peak_detector.py`:

```python
OTCPeakDetector(
    volume_threshold_multiplier=2.0,  # Volume must be 2× average
    price_momentum_days=3,            # Days to measure momentum
    peak_detection_window=10,         # Window for peak detection
    min_peak_prominence=0.05          # Min 5% prominence for peaks
)
```

## Key Insights from OTC Market Analysis

### Historical Pattern Observations:
1. **Volume precedes price**: Volume spikes happen before big price moves
2. **Peak formation**: Volume diverges (decreases) as price peaks
3. **Dump timing**: Price drops sharply after peak, often 20-50% in days
4. **Duration**: Pump phases typically last 3-10 days
5. **Patterns repeat**: Similar patterns across many OTC stocks

### Why This Algorithm Works:
- **Early entry**: Catches pump phase early (volume spike + momentum)
- **Peak detection**: Identifies peaks before dump starts
- **Volume divergence**: Key signal that price is at peak
- **Multi-signal confirmation**: Combines multiple indicators for confidence

## Performance Expectations

Based on historical OTC patterns:
- **Entry timing**: Should catch stocks during pump phase (early)
- **Exit timing**: Should sell near peak (before dump)
- **Win rate**: Higher when entry confidence > 70%
- **Risk reduction**: Peak detection prevents holding through dump

## Limitations

1. **No guarantees**: Past patterns don't guarantee future results
2. **Market changes**: OTC patterns can evolve
3. **False signals**: Some peaks may be false positives
4. **Data quality**: OTC data can be sparse/noisy
5. **Timing**: Perfect timing is difficult to achieve consistently

## Best Practices

1. **Use high confidence scores**: Only trade signals with >70% confidence
2. **Combine with fundamentals**: Don't rely solely on technical signals
3. **Manage risk**: Use stop-losses and position sizing
4. **Monitor continuously**: Check positions daily for exit signals
5. **Backtest**: Test strategies before using real money

## Next Steps

1. ✅ Algorithm implemented and integrated
2. 🔄 Monitor performance on real positions
3. 📊 Adjust parameters based on results
4. 🎯 Refine entry/exit criteria

