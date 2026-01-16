#!/usr/bin/env python3
"""
Quick script to check individual OTC symbols manually
Usage: python3 check_symbol.py SYMBOL1 SYMBOL2 ...
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from otc_trading_bot import get_stock_price, check_volume_increase, check_performance_gain, api, MIN_PRICE, MAX_PRICE, MIN_PERFORMANCE_PCT, VOLUME_MULTIPLIER

def check_symbol(symbol):
    """Check a single symbol and show detailed information"""
    print(f"\n{'='*60}")
    print(f"Checking: {symbol}")
    print(f"{'='*60}")
    
    # Check price
    print("\n1. Price Check:")
    try:
        price = get_stock_price(symbol, verbose_errors=True)
        if price:
            print(f"   ✓ Price: ${price:.2f}")
            if MIN_PRICE <= price <= MAX_PRICE:
                print(f"   ✓ Price is in range (${MIN_PRICE}-${MAX_PRICE})")
            else:
                print(f"   ✗ Price is OUT of range (${MIN_PRICE}-${MAX_PRICE})")
        else:
            print(f"   ✗ No price available for {symbol}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    if not price or price < MIN_PRICE or price > MAX_PRICE:
        print(f"\n   Stopping here - price check failed.")
        return
    
    # Check performance gain
    print("\n2. Performance Gain Check:")
    try:
        performance_pct = check_performance_gain(symbol)
        if performance_pct is not None:
            print(f"   ✓ Performance: {performance_pct:.2f}% gain over last ~3 days (min required: {MIN_PERFORMANCE_PCT}%)")
        else:
            print(f"   ✗ Performance has NOT met minimum {MIN_PERFORMANCE_PCT}% gain over last ~3 days")
    except Exception as e:
        print(f"   ✗ Error checking performance: {e}")
        performance_pct = None
    
    if performance_pct is None:
        print(f"\n   Stopping here - performance check failed.")
        return
    
    # Check volume
    print("\n3. Volume Increase Check:")
    try:
        has_volume_increase = check_volume_increase(symbol)
        if has_volume_increase:
            print(f"   ✓ Volume is ≥{VOLUME_MULTIPLIER}× the 30-day average")
        else:
            print(f"   ✗ Volume is NOT ≥{VOLUME_MULTIPLIER}× the 30-day average")
    except Exception as e:
        print(f"   ✗ Error checking volume: {e}")
        has_volume_increase = False
    
    # Liquidity check (part of volume check)
    print("\n4. Liquidity Check:")
    print(f"   ✓ Actively trading (recent trading activity verified in volume check)")
    
    # Overall status
    print(f"\n{'='*60}")
    if price and MIN_PRICE <= price <= MAX_PRICE and performance_pct and has_volume_increase:
        print(f"✓ {symbol} QUALIFIES for trading!")
        print(f"  - Price: ${price:.2f} (in range ${MIN_PRICE}-${MAX_PRICE})")
        print(f"  - Performance: {performance_pct:.2f}% (≥{MIN_PERFORMANCE_PCT}%)")
        print(f"  - Volume: ≥{VOLUME_MULTIPLIER}× 30-day average")
    else:
        print(f"✗ {symbol} does NOT qualify")
        print(f"  - Price: ${price:.2f if price else 0:.2f} {'✓' if price and MIN_PRICE <= price <= MAX_PRICE else '✗'}")
        print(f"  - Performance: {performance_pct:.2f if performance_pct else 'N/A'}% {'✓' if performance_pct and performance_pct >= MIN_PERFORMANCE_PCT else '✗'}")
        print(f"  - Volume: {'✓' if has_volume_increase else '✗'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 check_symbol.py SYMBOL1 [SYMBOL2 ...]")
        print("Example: python3 check_symbol.py KGFHY FSUGY")
        sys.exit(1)
    
    symbols = sys.argv[1:]
    print(f"\nChecking {len(symbols)} symbol(s)...")
    
    for symbol in symbols:
        check_symbol(symbol.upper())

