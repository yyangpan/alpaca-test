#!/usr/bin/env python3
"""
OTC Peak Detection Algorithm
Based on historical OTC market patterns:
1. Volume increase (pump phase)
2. Price momentum building
3. Peak formation with volume divergence
4. Price decline (dump phase)

This module detects peaks and provides entry/exit signals for OTC stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class OTCPeakDetector:
    """
    Enhanced peak detection with technical indicators:
    - Volume spike patterns
    - Price momentum (RSI-like)
    - Moving averages (SMA/EMA)
    - Volume-price divergence (peak signal)
    - MACD-like momentum
    - Reversal signals
    """
    
    def __init__(self, 
                 volume_threshold_multiplier: float = 2.0,
                 price_momentum_days: int = 3,
                 peak_detection_window: int = 10,
                 min_peak_prominence: float = 0.05,
                 use_rsi: bool = True,
                 use_macd: bool = True,
                 use_ma: bool = True):
        """
        Initialize peak detector with parameters.
        
        Args:
            volume_threshold_multiplier: Volume must be N× average for signal
            price_momentum_days: Days to measure price momentum
            peak_detection_window: Window size for peak detection
            min_peak_prominence: Minimum peak prominence (5% of price range)
            use_rsi: Use RSI-like momentum indicator
            use_macd: Use MACD-like momentum indicator
            use_ma: Use moving average crossovers
        """
        self.volume_threshold = volume_threshold_multiplier
        self.momentum_days = price_momentum_days
        self.peak_window = peak_detection_window
        self.min_prominence = min_peak_prominence
        self.use_rsi = use_rsi
        self.use_macd = use_macd
        self.use_ma = use_ma
    
    def get_historical_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """
        Get historical data for analysis.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist is None or len(hist) < 10:
                return None
            
            return hist
        except Exception:
            return None
    
    def detect_volume_spike(self, hist: pd.DataFrame) -> Dict:
        """
        Detect volume spike patterns indicating pump phase.
        
        Returns dict with:
        - has_spike: bool
        - volume_ratio: float
        - spike_intensity: float
        - days_since_spike: int
        """
        if hist is None or len(hist) < 30:
            return {'has_spike': False, 'volume_ratio': 0, 'spike_intensity': 0, 'days_since_spike': 999}
        
        volume_col = 'Volume' if 'Volume' in hist.columns else 'volume'
        
        # Calculate 30-day average volume
        avg_volume = hist[volume_col].tail(30).mean()
        
        # Get recent volumes (last 5 days)
        recent_volumes = hist[volume_col].tail(5)
        max_recent_volume = recent_volumes.max()
        recent_avg_volume = recent_volumes.mean()
        
        # Volume spike criteria
        volume_ratio = recent_avg_volume / avg_volume if avg_volume > 0 else 0
        has_spike = volume_ratio >= self.volume_threshold
        
        # Find when spike occurred
        days_since_spike = 999
        spike_intensity = 0
        
        if has_spike:
            # Find the day with maximum volume spike
            volume_ratios = hist[volume_col].tail(10) / avg_volume if avg_volume > 0 else pd.Series([0])
            max_idx = volume_ratios.idxmax() if len(volume_ratios) > 0 else None
            
            if max_idx:
                spike_date_idx = hist.index.get_loc(max_idx)
                days_since_spike = len(hist) - spike_date_idx - 1
                spike_intensity = volume_ratios.max()
        
        return {
            'has_spike': has_spike,
            'volume_ratio': volume_ratio,
            'spike_intensity': spike_intensity,
            'days_since_spike': max(0, days_since_spike)
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI-like indicator (Relative Strength Index).
        Returns RSI value (0-100). >70 = overbought, <30 = oversold.
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """
        Calculate MACD-like indicator (Moving Average Convergence Divergence).
        Returns dict with MACD line, signal line, and histogram.
        """
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'bullish': False}
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Bullish = MACD above signal line
        bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
            'bullish': bullish
        }
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict:
        """
        Calculate moving averages for trend detection.
        Returns dict with SMA (20, 50) and EMA (12, 26).
        """
        if len(prices) < 50:
            return {'sma20': None, 'sma50': None, 'ema12': None, 'ema26': None, 'golden_cross': False, 'death_cross': False}
        
        sma20 = prices.rolling(window=20).mean().iloc[-1]
        sma50 = prices.rolling(window=50).mean().iloc[-1]
        ema12 = prices.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = prices.ewm(span=26, adjust=False).mean().iloc[-1]
        
        current_price = prices.iloc[-1]
        
        # Golden cross = short MA crosses above long MA (bullish)
        golden_cross = sma20 > sma50 if sma20 and sma50 else False
        
        # Death cross = short MA crosses below long MA (bearish)
        death_cross = sma20 < sma50 if sma20 and sma50 else False
        
        return {
            'sma20': sma20,
            'sma50': sma50,
            'ema12': ema12,
            'ema26': ema26,
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'price_above_sma20': current_price > sma20 if sma20 else False
        }
    
    def detect_price_momentum(self, hist: pd.DataFrame) -> Dict:
        """
        Detect price momentum and trend direction.
        
        Returns dict with:
        - momentum: float (positive = up, negative = down)
        - trend: str ('up', 'down', 'neutral')
        - momentum_strength: float (0-1)
        """
        if hist is None or len(hist) < self.momentum_days + 5:
            return {'momentum': 0, 'trend': 'neutral', 'momentum_strength': 0}
        
        price_col = 'Close' if 'Close' in hist.columns else 'close'
        prices = hist[price_col].tail(self.momentum_days + 5)
        
        # Calculate momentum (rate of change)
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        momentum_pct = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
        
        # Calculate momentum strength (how consistent is the trend)
        price_changes = prices.pct_change().dropna()
        positive_days = (price_changes > 0).sum()
        momentum_strength = positive_days / len(price_changes) if len(price_changes) > 0 else 0
        
        # Determine trend
        if momentum_pct > 5 and momentum_strength > 0.6:
            trend = 'up'
        elif momentum_pct < -5 and momentum_strength < 0.4:
            trend = 'down'
        else:
            trend = 'neutral'
        
        return {
            'momentum': momentum_pct,
            'trend': trend,
            'momentum_strength': momentum_strength
        }
    
    def detect_peak_signals(self, hist: pd.DataFrame) -> Dict:
        """
        Detect peak formation signals:
        - Volume divergence (price up but volume decreasing)
        - Price peak with RSI overbought
        - Volume spike followed by price decline
        
        Returns dict with peak signals.
        """
        if hist is None or len(hist) < 15:
            return {'is_at_peak': False, 'peak_confidence': 0, 'reversal_risk': 0}
        
        price_col = 'Close' if 'Close' in hist.columns else 'close'
        volume_col = 'Volume' if 'Volume' in hist.columns else 'volume'
        
        # Get recent data
        recent = hist.tail(10)
        prices = recent[price_col]
        volumes = recent[volume_col]
        
        # Signal 1: Volume divergence (price up, volume down = peak signal)
        price_trend = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] if prices.iloc[-5] > 0 else 0
        volume_trend = (volumes.iloc[-5:].mean() - volumes.iloc[-10:-5].mean()) / volumes.iloc[-10:-5].mean() if volumes.iloc[-10:-5].mean() > 0 else 0
        
        volume_divergence = price_trend > 0.1 and volume_trend < -0.2  # Price up 10%, volume down 20%
        
        # Signal 1b: RSI overbought (if enabled) - enhanced peak detection
        rsi_overbought = False
        if self.use_rsi and len(prices) >= 15:
            try:
                # Need full history for RSI, not just recent tail
                price_col_full = 'Close' if 'Close' in hist.columns else 'close'
                prices_full = hist[price_col_full]
                rsi = self.calculate_rsi(prices_full, period=14)
                rsi_overbought = rsi > 70  # Overbought = potential peak
            except Exception:
                pass
        
        # Signal 1c: MACD bearish divergence (if enabled) - enhanced peak detection
        macd_bearish = False
        if self.use_macd and len(hist) >= 26:
            try:
                price_col_full = 'Close' if 'Close' in hist.columns else 'close'
                prices_full = hist[price_col_full]
                macd_data = self.calculate_macd(prices_full)
                macd_bearish = not macd_data.get('bullish', True)  # MACD turning bearish
            except Exception:
                pass
        
        # Signal 2: Recent peak detection using scipy
        try:
            price_values = prices.values
            price_range = price_values.max() - price_values.min()
            prominence = max(self.min_prominence * price_range, 0.01)
            
            peaks, properties = find_peaks(price_values, 
                                         prominence=prominence,
                                         distance=3)
            
            # Check if most recent day is near a peak
            is_near_peak = len(peaks) > 0 and peaks[-1] >= len(price_values) - 2
        except Exception:
            is_near_peak = False
        
        # Signal 3: Volume spike followed by decline (dump starting)
        avg_volume = volumes.iloc[-15:-5].mean() if len(volumes) >= 15 else volumes.mean()
        recent_volume = volumes.iloc[-3:].mean()
        price_change_last_3 = (prices.iloc[-1] - prices.iloc[-3]) / prices.iloc[-3] if prices.iloc[-3] > 0 else 0
        
        volume_decline = recent_volume < avg_volume * 0.7  # Volume dropped 30%
        price_dropping = price_change_last_3 < -0.05  # Price down 5% in last 3 days
        
        dump_signal = volume_decline and price_dropping
        
        # Calculate peak confidence (0-1) - enhanced with technical indicators
        peak_signals_list = [volume_divergence, is_near_peak, dump_signal]
        
        # Add RSI and MACD signals if enabled
        if rsi_overbought:
            peak_signals_list.append(True)
        if macd_bearish:
            peak_signals_list.append(True)
        
        # Weighted confidence calculation
        signal_count = len([s for s in peak_signals_list if s])
        total_possible = max(3, len(peak_signals_list))  # At least 3 base signals
        
        # Higher confidence when multiple signals align
        peak_confidence = signal_count / total_possible
        
        # Calculate reversal risk (higher when multiple signals align)
        reversal_risk = peak_confidence
        
        is_at_peak = peak_confidence >= 0.5  # 50% confidence threshold
        
        return {
            'is_at_peak': is_at_peak,
            'peak_confidence': peak_confidence,
            'reversal_risk': reversal_risk,
            'volume_divergence': volume_divergence,
            'near_peak': is_near_peak,
            'dump_signal': dump_signal
        }
    
    def get_entry_signal(self, symbol: str) -> Dict:
        """
        Determine if this is a good entry point.
        Entry signals:
        - Volume spike just starting
        - Price momentum building
        - Not yet at peak
        
        Returns:
            Dict with entry recommendation and confidence
        """
        hist = self.get_historical_data(symbol)
        
        if hist is None:
            return {'signal': 'NO_DATA', 'confidence': 0, 'reason': 'Insufficient data'}
        
        # Check volume spike
        volume_analysis = self.detect_volume_spike(hist)
        
        # Check momentum
        momentum_analysis = self.detect_price_momentum(hist)
        
        # Check if already at peak
        peak_analysis = self.detect_peak_signals(hist)
        
        # Entry criteria (enhanced with technical indicators)
        has_volume_spike = volume_analysis['has_spike']
        days_since_spike = volume_analysis['days_since_spike']
        positive_momentum = momentum_analysis['momentum'] > 5 or momentum_analysis['trend'] == 'up'
        not_at_peak = not peak_analysis['is_at_peak']
        
        # Additional technical indicator signals
        rsi_ok = momentum_analysis.get('rsi') is None or (30 < momentum_analysis['rsi'] < 70)  # Not overbought/oversold
        macd_bullish = momentum_analysis.get('macd') is None or momentum_analysis['macd'].get('bullish', False)
        ma_bullish = momentum_analysis.get('moving_averages') is None or momentum_analysis['moving_averages'].get('golden_cross', False) or momentum_analysis['moving_averages'].get('price_above_sma20', False)
        
        # Calculate confidence using multiple factors
        confidence_factors = []
        
        if has_volume_spike:
            confidence_factors.append(min(0.4, volume_analysis['volume_ratio'] / 10.0))  # Volume contribution
        
        if positive_momentum:
            confidence_factors.append(min(0.3, abs(momentum_analysis['momentum']) / 100.0))  # Momentum contribution
        
        if rsi_ok:
            confidence_factors.append(0.1)  # RSI contribution
        
        if macd_bullish:
            confidence_factors.append(0.1)  # MACD contribution
        
        if ma_bullish:
            confidence_factors.append(0.1)  # MA contribution
        
        # Entry signal logic (enhanced criteria)
        if has_volume_spike and days_since_spike <= 3 and positive_momentum and not_at_peak and (rsi_ok or not self.use_rsi):
            confidence = min(0.95, sum(confidence_factors))
            signal = 'BUY'
            reason = f"Volume spike ({volume_analysis['volume_ratio']:.1f}× avg), momentum {momentum_analysis['momentum']:.1f}%, trend {momentum_analysis['trend']}"
            if momentum_analysis.get('rsi'):
                reason += f", RSI {momentum_analysis['rsi']:.0f}"
            if macd_bullish:
                reason += ", MACD bullish"
        elif has_volume_spike and days_since_spike <= 5 and positive_momentum:
            confidence = 0.6
            signal = 'WATCH'
            reason = f"Volume spike detected but may be late entry. Momentum {momentum_analysis['momentum']:.1f}%"
        else:
            signal = 'WAIT'
            confidence = 0.3
            reason = "Waiting for volume spike and momentum alignment"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'volume_analysis': volume_analysis,
            'momentum_analysis': momentum_analysis,
            'peak_analysis': peak_analysis
        }
    
    def get_exit_signal(self, symbol: str, entry_price: Optional[float] = None) -> Dict:
        """
        Determine if this is a good exit point.
        Exit signals:
        - Peak detected (high confidence)
        - Price declining after peak
        - Volume divergence
        - Profit target reached (if entry_price provided)
        
        Returns:
            Dict with exit recommendation and confidence
        """
        hist = self.get_historical_data(symbol)
        
        if hist is None:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        price_col = 'Close' if 'Close' in hist.columns else 'close'
        current_price = hist[price_col].iloc[-1]
        
        # Check peak signals
        peak_analysis = self.detect_peak_signals(hist)
        
        # Check momentum (if declining, may be time to exit)
        momentum_analysis = self.detect_price_momentum(hist)
        
        # Check profit if entry price provided
        profit_pct = 0
        profit_target_reached = False
        if entry_price and entry_price > 0:
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            profit_target_reached = profit_pct >= 50  # 50% profit target
        
        # Exit signal logic
        if peak_analysis['is_at_peak'] and peak_analysis['peak_confidence'] >= 0.7:
            confidence = peak_analysis['peak_confidence']
            signal = 'SELL'
            reason = f"Peak detected (confidence: {confidence:.0%}). Volume divergence or reversal signals active."
        elif peak_analysis['reversal_risk'] >= 0.6 and momentum_analysis['trend'] == 'down':
            confidence = peak_analysis['reversal_risk']
            signal = 'SELL'
            reason = f"Reversal risk high ({confidence:.0%}). Price declining after peak."
        elif profit_target_reached:
            confidence = 0.9
            signal = 'SELL'
            reason = f"Profit target reached: {profit_pct:.1f}% gain"
        elif peak_analysis['reversal_risk'] >= 0.5:
            confidence = peak_analysis['reversal_risk']
            signal = 'CAUTION'
            reason = f"Reversal risk moderate ({confidence:.0%}). Consider taking profits."
        else:
            signal = 'HOLD'
            confidence = 0.3
            reason = f"No exit signals. Momentum: {momentum_analysis['trend']}, Peak risk: {peak_analysis['reversal_risk']:.0%}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'current_price': current_price,
            'profit_pct': profit_pct,
            'peak_analysis': peak_analysis,
            'momentum_analysis': momentum_analysis
        }
    
    def analyze_symbol(self, symbol: str, entry_price: Optional[float] = None) -> Dict:
        """
        Complete analysis of a symbol with entry/exit signals.
        """
        entry = self.get_entry_signal(symbol)
        exit_sig = self.get_exit_signal(symbol, entry_price)
        
        hist = self.get_historical_data(symbol, period="3mo")
        
        current_price = None
        if hist is not None and len(hist) > 0:
            price_col = 'Close' if 'Close' in hist.columns else 'close'
            current_price = hist[price_col].iloc[-1]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'entry_signal': entry,
            'exit_signal': exit_sig,
            'recommendation': self._get_overall_recommendation(entry, exit_sig)
        }
    
    def _get_overall_recommendation(self, entry: Dict, exit_sig: Dict) -> str:
        """Get overall recommendation based on entry and exit signals."""
        if exit_sig['signal'] == 'SELL':
            return 'SELL'
        elif entry['signal'] == 'BUY':
            return 'BUY'
        elif entry['signal'] == 'WATCH':
            return 'WATCH'
        else:
            return 'HOLD'


def analyze_otc_peak(symbol: str, entry_price: Optional[float] = None) -> Dict:
    """
    Convenience function to analyze a single OTC symbol for peak signals.
    """
    detector = OTCPeakDetector()
    return detector.analyze_symbol(symbol, entry_price)


if __name__ == '__main__':
    # Test the peak detector
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'VXRT'
    
    print(f"\nAnalyzing {symbol} for peak signals...")
    print("="*60)
    
    detector = OTCPeakDetector()
    analysis = detector.analyze_symbol(symbol)
    
    print(f"\nSymbol: {analysis['symbol']}")
    print(f"Current Price: ${analysis['current_price']:.2f}" if analysis['current_price'] else "Price: N/A")
    print(f"\nEntry Signal: {analysis['entry_signal']['signal']} (confidence: {analysis['entry_signal']['confidence']:.0%})")
    print(f"  Reason: {analysis['entry_signal']['reason']}")
    print(f"\nExit Signal: {analysis['exit_signal']['signal']} (confidence: {analysis['exit_signal']['confidence']:.0%})")
    print(f"  Reason: {analysis['exit_signal']['reason']}")
    print(f"\nOverall Recommendation: {analysis['recommendation']}")
    print("="*60)

