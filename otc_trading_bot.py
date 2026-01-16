#!/usr/bin/env python3
"""
OTC Market Stock Trading Bot
Automated trading bot for OTC stocks using Alpaca API
Designed to run on AWS EC2 or any Linux server
"""

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import sys
import os
import argparse
from typing import List, Dict, Optional
import logging

# Try to import yfinance as fallback for OTC data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Try to import peak detector
try:
    from otc_peak_detector import OTCPeakDetector, analyze_otc_peak
    PEAK_DETECTOR_AVAILABLE = True
except ImportError:
    PEAK_DETECTOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('otc_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Alpaca API Configuration
BASE_URL = 'https://paper-api.alpaca.markets'
API_KEY = 'PKTQBTL5H2JAHP2AOGKYUNMXST'
API_SECRET = 'GjwMgV9MUu34SNQAcYsf4s5ctipjfTQqUvuuWdzbn6G1'

# Strategy Parameters
MIN_PRICE = 0.1
MAX_PRICE = 5.0
DAILY_PURCHASE_AMOUNT = 100  # dollars
BUY_DAYS = 5  # Buy for 5 consecutive days
PERFORMANCE_DAYS = 3  # Check performance over last ~3 trading days
MIN_PERFORMANCE_PCT = 0  # No minimum performance requirement (removed)
VOLUME_AVERAGE_DAYS = 30  # 30-day average for volume comparison
VOLUME_MULTIPLIER = 2.0  # Volume must be ≥ 2× 30-day average
PROFIT_TARGET_PCT = 50  # Sell when 50% profit
MAX_STOCKS_PER_DAY = 10  # Select top 10 stocks daily

# Allowed OTC Exchanges
ALLOWED_OTC_EXCHANGES = ['OTCQX', 'OTCQB', 'Pink Current', 'OTC', 'OTCMKTS']

# File paths
SCHEDULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'buy_schedules.json')

# Global tracking for buy schedules
buy_schedules = {}


def initialize_api():
    """Initialize Alpaca API connection"""
    try:
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        account = api.get_account()
        logger.info(f"Alpaca API initialized successfully! Account Status: {account.status}")
        return api
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca API: {e}")
        sys.exit(1)


api = initialize_api()


def get_all_us_stocks() -> List[Dict]:
    """
    Get list of all US stocks from all exchanges (NYSE, NASDAQ, OTC, etc.).
    Returns stocks grouped by exchange.
    """
    try:
        # Get all assets (stocks) from all US exchanges
        assets = api.list_assets(status='active', asset_class='us_equity')
        
        # Group stocks by exchange
        all_stocks = []
        exchange_counts = {}
        
        for asset in assets:
            if hasattr(asset, 'exchange'):
                exchange = asset.exchange if hasattr(asset, 'exchange') else 'UNKNOWN'
                
                # Normalize exchange names for grouping
                exchange_normalized = normalize_exchange_name(exchange)
                
                stock_info = {
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': exchange,  # Original exchange name
                    'exchange_normalized': exchange_normalized,  # Normalized for grouping
                    'tradable': asset.tradable if hasattr(asset, 'tradable') else False
                }
                
                all_stocks.append(stock_info)
                
                # Count by normalized exchange
                exchange_counts[exchange_normalized] = exchange_counts.get(exchange_normalized, 0) + 1
        
        # Log summary by exchange
        logger.info(f"Found {len(all_stocks)} US stocks across all exchanges:")
        for exch, count in sorted(exchange_counts.items()):
            logger.info(f"  {exch}: {count} stocks")
        
        return all_stocks
    except Exception as e:
        # Suppress error messages during screening
        pass
        return []


def normalize_exchange_name(exchange: str) -> str:
    """
    Normalize exchange names for grouping.
    Maps various exchange identifiers to standard names.
    """
    exchange_upper = exchange.upper() if exchange else ''
    
    # NYSE
    if 'NYSE' in exchange_upper or exchange_upper == 'N' or 'NEW YORK' in exchange_upper:
        return 'NYSE'
    
    # NASDAQ
    if 'NASDAQ' in exchange_upper or exchange_upper in ['Q', 'QNX', 'BX', 'PHLX']:
        return 'NASDAQ'
    
    # OTC Markets
    if any(otc in exchange_upper for otc in ['OTC', 'OTCQX', 'OTCQB', 'PINK', 'OTCMKTS']):
        return 'OTC'
    
    # AMEX
    if 'AMEX' in exchange_upper or exchange_upper == 'A':
        return 'AMEX'
    
    # Other regional exchanges
    if 'BATS' in exchange_upper:
        return 'BATS'
    if 'CHX' in exchange_upper or 'CHICAGO' in exchange_upper:
        return 'CHX'
    if 'ARCA' in exchange_upper:
        return 'ARCA'
    
    # Unknown or other
    return exchange if exchange else 'UNKNOWN'


def check_volume_increase(symbol: str) -> bool:
    """
    Check if current volume is ≥ 3× the 30-day average using Yahoo Finance.
    Also checks for liquidity (recent trading activity).
    Returns True if volume criteria is met.
    """
    if not YFINANCE_AVAILABLE:
        return False
    
    try:
        ticker = yf.Ticker(symbol)
        # Get 30+ days of historical data
        hist = ticker.history(period="2mo")  # ~2 months to ensure 30 trading days
        
        if hist is None or len(hist) < 5:
            return False  # Not enough data
        
        bars = hist
        
        if len(bars) < 5:
            return False  # Not enough data for liquidity check
        
        # Check liquidity: Ensure recent trading activity (last 3 days should have some volume)
        # yfinance uses capitalized column names
        volume_col = 'Volume' if 'Volume' in bars.columns else 'volume'
        close_col = 'Close' if 'Close' in bars.columns else 'close'
        
        recent_bars = bars.tail(3)
        if volume_col not in recent_bars.columns or recent_bars[volume_col].sum() == 0:
            return False  # Dead ticker - no recent trading
        
        # Calculate 30-day average volume
        # Get last 30 trading days (or as many as available)
        volume_data = bars.tail(VOLUME_AVERAGE_DAYS) if len(bars) >= VOLUME_AVERAGE_DAYS else bars
        
        # Filter out zero volume days for average calculation
        non_zero_volumes = volume_data[volume_data[volume_col] > 0][volume_col]
        
        if len(non_zero_volumes) == 0:
            return False  # No trading activity in the period
        
        avg_volume_30day = non_zero_volumes.mean()
        
        # Get recent volume (last 3 trading days average)
        recent_volumes = bars.tail(3)[volume_col]
        recent_avg_volume = recent_volumes.mean()
        
        # Check if recent volume is ≥ 3× 30-day average
        if avg_volume_30day > 0:
            volume_ratio = recent_avg_volume / avg_volume_30day
            return volume_ratio >= VOLUME_MULTIPLIER
        
        return False
        
    except Exception:
        # Skip error messages during screening
        return False


def check_performance_gain(symbol: str) -> Optional[float]:
    """
    Get performance percentage (no requirement - just for tracking).
    Performance requirement has been removed - returns any performance value.
    """
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        # Get ~3+ trading days of data (use 5 days to ensure we have ~3 trading days)
        hist = ticker.history(period="5d")
        
        if hist is None or len(hist) < 2:
            return None  # Not enough data
        
        bars = hist
        
        if len(bars) < 2:
            return None  # Not enough data
        
        # Get price from ~3 trading days ago and current price
        recent_bars = bars.tail(PERFORMANCE_DAYS + 2)
        
        if len(recent_bars) < 2:
            return None
        
        # Get price from ~3 days ago (start of period)
        price_col = 'Close' if 'Close' in recent_bars.columns else 'close'
        start_price = float(recent_bars[price_col].iloc[0])
        # Get current/latest price
        end_price = float(recent_bars[price_col].iloc[-1])
        
        if start_price <= 0:
            return None
        
        # Calculate performance percentage (no minimum threshold - returns any value)
        performance_pct = ((end_price - start_price) / start_price) * 100
        
        return performance_pct  # Return any performance value (no filtering)
        
    except Exception:
        # Skip error messages during screening
        return None


def get_stock_price(symbol: str, verbose_errors: bool = True, use_fallback: bool = True) -> Optional[float]:
    """
    Get current price of a stock using Yahoo Finance directly.
    Skips Alpaca - uses Yahoo Finance for better OTC coverage.
    Returns None if price cannot be determined.
    """
    error_msg = None
    
    # Use Yahoo Finance directly (better OTC coverage)
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try to get current price from info
            if 'currentPrice' in info and info['currentPrice']:
                price = float(info['currentPrice'])
                if price > 0:
                    if verbose_errors:
                        logger.info(f"✓ Got price from Yahoo Finance: {symbol} = ${price:.2f}")
                    return price
            
            # Try 'regularMarketPrice' as alternative
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                price = float(info['regularMarketPrice'])
                if price > 0:
                    if verbose_errors:
                        logger.info(f"✓ Got price from Yahoo Finance (regularMarket): {symbol} = ${price:.2f}")
                    return price
            
            # Try 'previousClose' as last resort (delayed data)
            if 'previousClose' in info and info['previousClose']:
                price = float(info['previousClose'])
                if price > 0:
                    if verbose_errors:
                        logger.info(f"✓ Got previous close from Yahoo Finance: {symbol} = ${price:.2f}")
                    return price
            
            # Try to get latest history
            hist = ticker.history(period="1d", interval="1m")
            if len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    if verbose_errors:
                        logger.info(f"✓ Got price from Yahoo Finance history: {symbol} = ${price:.2f}")
                    return price
            
            if verbose_errors:
                if not error_msg:
                    error_msg = f"No price data available for {symbol} (Yahoo Finance)"
                logger.debug(f"⚠️  {error_msg} - Yahoo Finance also has no data")
        except Exception as e:
            if verbose_errors:
                logger.debug(f"⚠️  Yahoo Finance error for {symbol}: {str(e)}")
    
    # Log the error if verbose mode is enabled
    if verbose_errors and error_msg:
        logger.info(f"⚠️  {error_msg}")
    
    return None


def screen_stocks(symbols: List[str], stocks_info: List[Dict] = None) -> List[Dict]:
    """
    Screen stocks based on price range and volume activity.
    Returns list of qualified stocks with their metrics, including exchange information.
    
    Args:
        symbols: List of stock symbols to screen
        stocks_info: Optional list of stock info dicts with exchange info (from get_all_us_stocks)
    """
    qualified_stocks = []
    total_stocks = len(symbols)
    screened_count = 0
    no_price_count = 0
    price_out_of_range_count = 0
    no_volume_increase_count = 0
    no_price_symbols = []  # Track symbols without prices for manual review
    
    logger.info(f"\nScreening {total_stocks} US stocks...")
    
    # Progress indicators
    progress_interval = max(1, total_stocks // 50)  # Update progress every ~2%
    
    for idx, symbol in enumerate(symbols, 1):
        try:
            # Progress indicator (use sys.stdout for real-time updates)
            if idx % progress_interval == 0 or idx == total_stocks:
                progress_pct = (idx / total_stocks) * 100
                filled = int(progress_pct / 2)
                bar = '=' * filled + ' ' * (50 - filled)
                sys.stdout.write(f"\rProgress: [{bar}] {progress_pct:.1f}% ({idx}/{total_stocks})")
                sys.stdout.flush()
            
            # Small delay to reduce API rate limiting (0.1 seconds between requests)
            if idx > 1:
                time.sleep(0.1)
            
            # Quick check: Get price first (fastest filter)
            # Set verbose_errors=False to reduce output (only show progress)
            price = get_stock_price(symbol, verbose_errors=False)
            if price is None:
                no_price_count += 1
                no_price_symbols.append(symbol)
                continue
            
            # Price range filter
            if price < MIN_PRICE or price > MAX_PRICE:
                price_out_of_range_count += 1
                continue
            
            # Get performance percentage (no requirement - just for tracking)
            # Performance is optional and only used for display, not filtering
            performance_pct = check_performance_gain(symbol)
            if performance_pct is None:
                performance_pct = 0.0  # Default to 0 if can't calculate
            
            # Check volume increase (≥ 2× 30-day average)
            if not check_volume_increase(symbol):
                no_volume_increase_count += 1
                continue
            
            # Get additional volume data for ranking (from Yahoo Finance)
            if YFINANCE_AVAILABLE:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2mo")
                volume_col = 'Volume' if 'Volume' in hist.columns else 'volume'
                avg_volume = hist[volume_col].mean() if len(hist) > 0 else 0
                recent_volume = hist.tail(3)[volume_col].mean() if len(hist) >= 3 else 0
            else:
                avg_volume = 0
                recent_volume = 0
            
            # Check if tradable (for info only - still qualify if not tradable)
            is_tradable = False
            exchange = 'UNKNOWN'
            exchange_normalized = 'UNKNOWN'
            stock_name = symbol
            try:
                asset = api.get_asset(symbol)
                is_tradable = asset.tradable if hasattr(asset, 'tradable') else False
                if hasattr(asset, 'exchange'):
                    exchange = asset.exchange
                    exchange_normalized = normalize_exchange_name(exchange)
            except Exception:
                pass
            
            # Use stock info from lookup if available
            if symbol in stocks_info_dict:
                stock_info = stocks_info_dict[symbol]
                exchange = stock_info.get('exchange', exchange)
                exchange_normalized = stock_info.get('exchange_normalized', exchange_normalized)
                stock_name = stock_info.get('name', stock_name)
                is_tradable = stock_info.get('tradable', is_tradable)
            
            # Add peak detection analysis if available
            peak_score = 0
            entry_signal = 'N/A'
            if PEAK_DETECTOR_AVAILABLE:
                try:
                    from otc_peak_detector import OTCPeakDetector
                    detector = OTCPeakDetector()
                    entry_analysis = detector.get_entry_signal(symbol)
                    peak_score = entry_analysis['confidence']
                    entry_signal = entry_analysis['signal']
                except Exception:
                    pass
            
            qualified_stocks.append({
                'symbol': symbol,
                'name': stock_name,
                'exchange': exchange,
                'exchange_normalized': exchange_normalized,
                'price': price,
                'performance_pct': performance_pct,
                'avg_volume': avg_volume,
                'recent_volume': recent_volume,
                'volume_ratio': recent_volume / avg_volume if avg_volume > 0 else 0,
                'tradable': is_tradable,
                'peak_score': peak_score,  # AI peak detection confidence (for ranking)
                'entry_signal': entry_signal  # BUY/WATCH/WAIT
            })
            
            screened_count += 1
            # Don't print individual stock info - just show progress (reduce verbose output)
            
        except Exception:
            # Skip error messages during screening (silent failure)
            continue
    
    # Print final progress and newline
    sys.stdout.write(f"\rProgress: [{'=' * 50}] 100.0% ({total_stocks}/{total_stocks})\n")
    sys.stdout.flush()
    
    # Summary
    logger.info(f"\nScreening Complete!")
    logger.info(f"  Total screened: {total_stocks}")
    logger.info(f"  No price/quote available: {no_price_count}")
    logger.info(f"  Price out of range (${MIN_PRICE}-${MAX_PRICE}): {price_out_of_range_count}")
    logger.info(f"  No volume increase (≥{VOLUME_MULTIPLIER}× 30-day avg): {no_volume_increase_count}")
    logger.info(f"  Qualified: {len(qualified_stocks)}")
    
    # Skip showing list of symbols without prices (reduce verbose output)
    
    # Sort by peak detection confidence only (peak_score) - primary ranking
    # Peak score indicates entry timing quality (catching pump early)
    qualified_stocks.sort(key=lambda x: x.get('peak_score', 0), reverse=True)
    
    return qualified_stocks[:MAX_STOCKS_PER_DAY]


def get_positions() -> Dict[str, Dict]:
    """
    Get all current positions from Alpaca account.
    Returns a dictionary with symbol as key and position info as value.
    """
    try:
        positions = api.list_positions()
        positions_dict = {}
        
        for pos in positions:
            positions_dict[pos.symbol] = {
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc) * 100  # as percentage
            }
        
        return positions_dict
    except Exception:
        # Suppress error messages
        return {}


def load_buy_schedules() -> Dict:
    """Load buy schedules from file (for persistence across runs)"""
    global buy_schedules
    try:
        if os.path.exists(SCHEDULES_FILE):
            with open(SCHEDULES_FILE, 'r') as f:
                data = json.load(f)
                # Convert date strings back to datetime
                for symbol, schedule in data.items():
                    schedule['start_date'] = datetime.fromisoformat(schedule['start_date'])
                return data
        return {}
    except Exception as e:
        # Suppress error messages
        pass
        return {}


def save_buy_schedules():
    """Save buy schedules to file"""
    try:
        data = {}
        for symbol, schedule in buy_schedules.items():
            data[symbol] = schedule.copy()
            data[symbol]['start_date'] = schedule['start_date'].isoformat()
        
        with open(SCHEDULES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Buy schedules saved to {SCHEDULES_FILE}")
    except Exception as e:
        # Suppress error messages
        pass


def execute_buy(symbol: str) -> bool:
    """
    Execute a market buy order for the specified amount.
    Note: Will skip if stock is not tradable on Alpaca.
    """
    try:
        # Check if stock is tradable
        try:
            asset = api.get_asset(symbol)
            if not asset.tradable:
                # Suppress warning - stock not tradable
                return False
        except Exception:
            # Suppress warning
            return False
        
        current_price = get_stock_price(symbol)
        if current_price is None:
            # Suppress warning
            return False
        
        # Calculate number of shares to buy (round down to whole shares)
        shares_to_buy = int(DAILY_PURCHASE_AMOUNT / current_price)
        
        if shares_to_buy < 1:
            # Suppress warning
            return False
        
        # Place market order
        order = api.submit_order(
            symbol=symbol,
            qty=shares_to_buy,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        logger.info(f"✓ Buy order submitted for {symbol}: {shares_to_buy} shares @ ~${current_price:.2f}")
        
        # Wait for order to fill
        time.sleep(2)
        
        # Update buy schedule
        if symbol not in buy_schedules:
            buy_schedules[symbol] = {
                'start_date': datetime.now(),
                'days_bought': 0,
                'total_cost': 0.0,
                'total_shares': 0.0
            }
        
        # Calculate actual cost (approximate)
        actual_cost = shares_to_buy * current_price
        buy_schedules[symbol]['days_bought'] += 1
        buy_schedules[symbol]['total_cost'] += actual_cost
        buy_schedules[symbol]['total_shares'] += shares_to_buy
        
        save_buy_schedules()
        
        return True
        
    except Exception:
        # Suppress error messages
        return False


def execute_sell(symbol: str, reason: str = "Profit target reached") -> bool:
    """
    Execute a market sell order for all shares of a symbol.
    """
    try:
        positions = get_positions()
        
        if symbol not in positions:
            # Suppress warning
            return False
        
        qty = positions[symbol]['qty']
        current_price = positions[symbol]['current_price']
        
        if qty <= 0:
            # Suppress warning
            return False
        
        # Place market sell order
        order = api.submit_order(
            symbol=symbol,
            qty=int(qty),
            side='sell',
            type='market',
            time_in_force='day'
        )
        
        logger.info(f"✓ Sell order submitted for {symbol}: {int(qty)} shares @ ~${current_price:.2f} - {reason}")
        
        # Remove from buy schedule tracking
        if symbol in buy_schedules:
            del buy_schedules[symbol]
            save_buy_schedules()
        
        return True
        
    except Exception:
        # Suppress error messages
        return False


def check_profit_targets():
    """
    Check all positions and sell if:
    1. Peak detected (using AI algorithm) - catches dump phase early
    2. Profit target reached (50% above average cost)
    3. Reversal signals detected
    """
    positions = get_positions()
    
    if not positions:
        return
    
    # Initialize peak detector if available
    peak_detector = None
    if PEAK_DETECTOR_AVAILABLE:
        try:
            from otc_peak_detector import OTCPeakDetector
            peak_detector = OTCPeakDetector()
        except Exception:
            pass
    
    for symbol, pos_info in positions.items():
        try:
            avg_entry = pos_info['avg_entry_price']
            current_price = pos_info['current_price']
            profit_pct = pos_info['unrealized_plpc']
            
            sell_reason = None
            
            # Method 1: Use AI peak detection if available (catches peaks before dump)
            if peak_detector:
                try:
                    exit_analysis = peak_detector.get_exit_signal(symbol, entry_price=avg_entry)
                    
                    if exit_analysis['signal'] == 'SELL' and exit_analysis['confidence'] >= 0.7:
                        sell_reason = f"Peak detected: {exit_analysis['reason']} (confidence: {exit_analysis['confidence']:.0%})"
                    elif exit_analysis['signal'] == 'SELL' and profit_pct >= 30:  # Lower threshold if peak detected
                        sell_reason = f"Peak detected with {profit_pct:.1f}% profit: {exit_analysis['reason']}"
                    elif exit_analysis['signal'] == 'CAUTION' and profit_pct >= PROFIT_TARGET_PCT:
                        sell_reason = f"Reversal risk detected with {profit_pct:.1f}% profit: {exit_analysis['reason']}"
                except Exception:
                    # Fall back to simple profit target if peak detection fails
                    pass
            
            # Method 2: Fallback to simple profit target if no peak detection
            if not sell_reason and profit_pct >= PROFIT_TARGET_PCT:
                sell_reason = f"Profit target reached ({profit_pct:.2f}%)"
            
            # Execute sell if reason found
            if sell_reason:
                logger.info(f"\n🎯 {symbol} - SELL SIGNAL!")
                logger.info(f"   Avg Entry: ${avg_entry:.2f}, Current: ${current_price:.2f}, Profit: {profit_pct:.2f}%")
                logger.info(f"   Reason: {sell_reason}")
                execute_sell(symbol, sell_reason)
                
        except Exception:
            # Suppress error messages
            pass


def process_daily_buys():
    """
    Process daily buy orders for stocks in the buy schedule.
    """
    today = datetime.now().date()
    
    # Check existing buy schedules
    symbols_to_continue = []
    
    for symbol, schedule in list(buy_schedules.items()):
        start_date = schedule['start_date'].date() if isinstance(schedule['start_date'], datetime) else datetime.fromisoformat(schedule['start_date']).date()
        days_elapsed = (today - start_date).days
        days_bought = schedule['days_bought']
        
        # If we haven't bought today and still within 5-day window
        if days_bought < BUY_DAYS and days_elapsed < BUY_DAYS + 2:
            # Only buy if it's a new day (to prevent multiple buys on same day)
            if days_bought == 0 or days_elapsed >= days_bought:
                symbols_to_continue.append(symbol)
    
    # Execute buys for stocks in schedule
    for symbol in symbols_to_continue:
        execute_buy(symbol)
    
    # Clean up completed schedules
    for symbol, schedule in list(buy_schedules.items()):
        if schedule['days_bought'] >= BUY_DAYS:
            logger.info(f"✓ Completed 5-day buy schedule for {symbol}")


def daily_screen_and_trade(symbols: List[str] = None):
    """
    Main function to screen stocks from all US markets and execute trades.
    """
    logger.info("\n" + "="*60)
    logger.info(f"Daily US Market Stock Screening - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Step 1: Check existing positions for profit targets
    logger.info("\n[Step 1] Checking existing positions for profit targets...")
    check_profit_targets()
    
    # Step 2: Process daily buys for stocks in schedule
    logger.info("\n[Step 2] Processing daily buy orders...")
    process_daily_buys()
    
    # Step 3: Screen for new stock candidates
    logger.info("\n[Step 3] Screening for new stock candidates...")
    
    # Get all US stocks if not provided
    all_stocks = None
    if symbols is None:
        all_stocks = get_all_us_stocks()
        symbols = [stock['symbol'] for stock in all_stocks]
    else:
        # If symbols provided, try to get exchange info for them
        all_stocks = get_all_us_stocks()
    
    if not symbols:
        # Suppress warning - no symbols available
        pass
        return
    
    # If we already have MAX_STOCKS_PER_DAY in buy schedule, skip screening
    active_symbols_count = sum(1 for s in buy_schedules.values() if s['days_bought'] < BUY_DAYS)
    
    if active_symbols_count >= MAX_STOCKS_PER_DAY:
        logger.info(f"Already have {active_symbols_count} active stocks in buy schedule. Skipping new screening.")
        return
    
    # Screen stocks (pass stock info for exchange information)
    qualified_stocks = screen_stocks(symbols, stocks_info=all_stocks)
    
    if not qualified_stocks:
        logger.info("\n❌ No qualified stocks found today. No new trades.")
        return
    
    # Step 4: Select top candidates (avoid duplicates)
    logger.info(f"\n[Step 4] Selecting up to {MAX_STOCKS_PER_DAY} stocks from {len(qualified_stocks)} candidates...")
    
    selected_count = 0
    for stock in qualified_stocks:
        if selected_count >= MAX_STOCKS_PER_DAY:
            break
        
        symbol = stock['symbol']
        
        # Skip if already in buy schedule
        if symbol in buy_schedules:
            logger.info(f"Skipping {symbol} - already in buy schedule")
            continue
        
        # Skip if already have a position
        positions = get_positions()
        if symbol in positions:
            logger.info(f"Skipping {symbol} - already have position")
            continue
        
        # Start buy schedule (only if tradable)
        tradable_status = "✓ Tradable" if stock.get('tradable', False) else "⚠️ Not tradable (screening only)"
        logger.info(f"\n📈 Selected: {symbol} @ ${stock['price']:.2f} - {tradable_status}")
        
        # Only try to buy if tradable
        if stock.get('tradable', False):
            if execute_buy(symbol):
                selected_count += 1
        else:
            logger.info(f"⚠️  {symbol} is NOT tradable on Alpaca - screening only (no buy order placed)")
            # Still count it as selected for screening purposes
            selected_count += 1
    
    if selected_count == 0:
        logger.info("\nNo new stocks selected today.")
    else:
        logger.info(f"\n✓ Started buy schedules for {selected_count} new stock(s)")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Trading Summary:")
    logger.info("="*60)
    
    positions = get_positions()
    if positions:
        logger.info(f"\nCurrent Positions ({len(positions)}):")
        for symbol, pos in positions.items():
            logger.info(f"  {symbol}: {pos['qty']:.0f} shares @ ${pos['avg_entry_price']:.2f} avg")
            logger.info(f"    Current: ${pos['current_price']:.2f}, P/L: {pos['unrealized_plpc']:.2f}%")
    else:
        logger.info("\nNo current positions.")
    
    # Show qualified stocks summary grouped by exchange, ranked by confidence
    if qualified_stocks:
        logger.info(f"\n📊 Qualified Stocks Summary ({len(qualified_stocks)} stocks) - Grouped by Exchange:")
        logger.info("="*60)
        
        # Group stocks by exchange
        stocks_by_exchange = {}
        for stock in qualified_stocks:
            exch = stock.get('exchange_normalized', 'UNKNOWN')
            if exch not in stocks_by_exchange:
                stocks_by_exchange[exch] = []
            stocks_by_exchange[exch].append(stock)
        
        # Sort exchanges (preferred order: NYSE, NASDAQ, OTC, others)
        exchange_order = ['NYSE', 'NASDAQ', 'AMEX', 'OTC', 'BATS', 'ARCA', 'CHX']
        sorted_exchanges = sorted(
            stocks_by_exchange.keys(),
            key=lambda x: (exchange_order.index(x) if x in exchange_order else 999, x)
        )
        
        # Print stocks grouped by exchange, ranked by confidence (peak_score)
        for exchange in sorted_exchanges:
            stocks = stocks_by_exchange[exchange]
            # Sort by confidence (peak_score) descending
            stocks.sort(key=lambda x: x.get('peak_score', 0), reverse=True)
            
            logger.info(f"\n🏛️  {exchange} ({len(stocks)} stocks):")
            for stock in stocks[:MAX_STOCKS_PER_DAY]:  # Limit per exchange for display
                confidence = stock.get('peak_score', 0) * 100  # Convert to percentage
                tradable_status = "✓ Tradable" if stock.get('tradable', False) else "⚠️ Not tradable"
                peak_info = f", Confidence: {confidence:.0f}%" if confidence > 0 else ""
                entry_sig = stock.get('entry_signal', 'N/A')
                logger.info(f"  • {stock['symbol']} @ ${stock['price']:.2f} | {stock.get('name', 'N/A')[:40]}")
                logger.info(f"    Perf: {stock['performance_pct']:.1f}%, Vol: {stock['volume_ratio']:.2f}×, Signal: {entry_sig}{peak_info} - {tradable_status}")
    
    if buy_schedules:
        logger.info(f"\nActive Buy Schedules ({len(buy_schedules)}):")
        for symbol, schedule in buy_schedules.items():
            avg_cost = schedule['total_cost'] / schedule['total_shares'] if schedule['total_shares'] > 0 else 0
            logger.info(f"  {symbol}: {schedule['days_bought']}/{BUY_DAYS} days, ")
            logger.info(f"    {schedule['total_shares']:.0f} shares @ ${avg_cost:.2f} avg")
    
    logger.info("\n" + "="*60)


def view_account():
    """Display account information"""
    try:
        account = api.get_account()
        logger.info(f"\nAccount Status: {account.status}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"Equity: ${float(account.equity):,.2f}")
    except Exception as e:
        # Suppress error messages
        pass


def view_positions():
    """Display all current positions"""
    positions = get_positions()
    if positions:
        logger.info(f"\nCurrent Positions ({len(positions)}):")
        for symbol, pos in positions.items():
            logger.info(f"\n{symbol}:")
            logger.info(f"  Quantity: {pos['qty']:.0f} shares")
            logger.info(f"  Avg Entry Price: ${pos['avg_entry_price']:.2f}")
            logger.info(f"  Current Price: ${pos['current_price']:.2f}")
            logger.info(f"  Market Value: ${pos['market_value']:,.2f}")
            logger.info(f"  Unrealized P/L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2f}%)")
    else:
        logger.info("No current positions.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='OTC Market Stock Trading Bot')
    parser.add_argument('--symbols', '-s', nargs='+', help='List of OTC symbols to screen (space-separated)')
    parser.add_argument('--symbols-file', '-f', help='File containing OTC symbols (one per line)')
    parser.add_argument('--account', '-a', action='store_true', help='View account information')
    parser.add_argument('--positions', '-p', action='store_true', help='View current positions')
    parser.add_argument('--check-only', '-c', action='store_true', help='Only check profit targets, no new trades')
    
    args = parser.parse_args()
    
    # Load buy schedules
    global buy_schedules
    buy_schedules = load_buy_schedules()
    
    # Handle command-line arguments
    if args.account:
        view_account()
        return
    
    if args.positions:
        view_positions()
        return
    
    if args.check_only:
        logger.info("Checking profit targets only...")
        check_profit_targets()
        view_positions()
        return
    
    # Get OTC symbols
    otc_symbols = None
    if args.symbols:
        otc_symbols = args.symbols
        logger.info(f"Using provided symbols: {otc_symbols}")
    elif args.symbols_file:
        try:
            with open(args.symbols_file, 'r') as f:
                otc_symbols = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(otc_symbols)} symbols from {args.symbols_file}")
        except Exception as e:
            # Suppress error messages
            pass
            sys.exit(1)
    
    # Run main trading function
    daily_screen_and_trade(otc_symbols)


if __name__ == '__main__':
    main()

