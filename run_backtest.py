#!/usr/bin/env python3
"""
Quick backtest runner - validates peak detection on historical data
"""

import sys
from otc_peak_detector import OTCPeakDetector
from backtest_peak_detector import validate_peak_detection, backtest_symbol

if __name__ == '__main__':
    print("\n" + "="*60)
    print("OTC Peak Detection - Model Validation")
    print("="*60 + "\n")
    
    # Test with a few OTC symbols
    test_symbols = ['VXRT']  # Add more symbols here
    
    print("Validating peak detection accuracy...")
    print("-" * 60)
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        result = validate_peak_detection(symbol, period="6mo")
        
        if result.get('status') == 'insufficient_data':
            print(f"  ⚠️  Insufficient data")
            continue
        
        print(f"  Actual peaks found: {result['actual_peaks']}")
        print(f"  Correct detections: {result['correct_detections']}")
        if result['actual_peaks'] > 0:
            print(f"  Detection accuracy: {result['detection_accuracy']:.1f}%")
    
    print("\n" + "="*60)
    print("Running backtest simulation...")
    print("-" * 60)
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        result = backtest_symbol(symbol, period="1y", min_confidence=0.6)
        
        if result.get('status') != 'success':
            print(f"  ⚠️  {result.get('status', 'unknown')}")
            continue
        
        print(f"  Total trades: {result['total_trades']}")
        if result['total_trades'] > 0:
            print(f"  Win rate: {result['win_rate']:.1f}%")
            print(f"  Avg profit: {result['avg_profit_pct']:.2f}%")
            print(f"  Total profit: {result['total_profit_pct']:.2f}%")
            print(f"  Best trade: {result['max_profit_pct']:.2f}%")
            print(f"  Worst trade: {result['max_loss_pct']:.2f}%")
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60 + "\n")

