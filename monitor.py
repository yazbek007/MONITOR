import os
import pandas as pd
import numpy as np
import hashlib
from binance.client import Client
from binance.enums import *
import time
from datetime import datetime, timedelta
import requests
import logging
import warnings
import threading
from flask import Flask, jsonify, request
import pytz
from dotenv import load_dotenv
from functools import wraps

warnings.filterwarnings('ignore')
load_dotenv()

# ========== Integrated Basic Settings ==========
TRADING_SETTINGS = {
    'symbols': ["BNBUSDT", "ETHUSDT"],
    'base_trade_amount': 4,  # 4 USD
    'leverage': 50,  # 50x leverage
    'position_size': 4 * 50,  # 200 USD position size
    'max_simultaneous_trades': 2,
    'max_trades_per_symbol': 1,
    'min_balance_required': 5,
}

# Advanced Stop Loss and Take Profit Settings
RISK_SETTINGS = {
    # ⭐ Advanced Stop Loss Settings
    'stop_loss_phases': {
        'PHASE_1': {'distance_ratio': 0.7, 'allocation': 0.5},
        'PHASE_2': {'distance_ratio': 1.0, 'allocation': 0.5}
    },
    'min_stop_distance': 0.005,  # 0.5%
    'max_stop_distance': 0.022,  # 2.2%
    'emergency_stop_ratio': 0.01,  # 1%
    'max_trade_duration_hours': 1,
    'extension_duration_minutes': 30,
    'final_extension_minutes': 30,
    
    # ⭐ Advanced Take Profit Settings
    'take_profit_levels': {
        'LEVEL_1': {'target': 0.0020, 'allocation': 0.6},
        'LEVEL_2': {'target': 0.0025, 'allocation': 0.4}
    },
    
    # ⭐ Risk Management Settings
    'atr_period': 14,
    'risk_ratio': 0.5,
    'volatility_multiplier': 1.5,
    'margin_risk_threshold': 0.7,
    'position_reduction': 0.5,
}

# ⭐ Trading Levels from System
TRADING_LEVELS = {
    'LEVEL_1': {'min_confidence': 25, 'max_confidence': 65, 'allocation': 0.5},
    'LEVEL_2': {'min_confidence': 66, 'max_confidence': 80, 'allocation': 0.75},
    'LEVEL_3': {'min_confidence': 81, 'max_confidence': 100, 'allocation': 0.99}
}

# Timezone settings
damascus_tz = pytz.timezone('Asia/Damascus')
os.environ['TZ'] = 'Asia/Damascus'

# Flask app for monitoring
app = Flask(__name__)

# ========== Security Settings ==========
API_KEYS = {
    os.getenv("EXECUTOR_API_KEY", "default_key_here"): "bot_scanner",
    os.getenv("MANAGER_API_KEY", "manager_key_here"): "trade_manager"
}

def require_api_key(f):
    """API Authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not api_key or api_key not in API_KEYS:
            return jsonify({'success': False, 'message': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return decorated_function

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_trade_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Core Classes ==========

class NtfyNotifier:
    """Ntfy Notification Manager with Enhanced Error Handling"""
    
    def __init__(self, topic):
        self.topic = topic
        self.base_url = f"https://ntfy.sh/{topic}"
        self.test_connection()
    
    def test_connection(self):
        """Test Ntfy Connection"""
        try:
            if not self.topic:
                logger.error("❌ Ntfy topic not found in environment variables")
                return False
            
            test_url = f"https://ntfy.sh"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Ntfy connection active")
                return True
            else:
                logger.error(f"❌ Ntfy test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing Ntfy: {e}")
            return False
    
    def send_message(self, message, message_type='info', title=None):
        """Send message with enhanced error handling"""
        try:
            if not self.topic:
                logger.warning("⚠️ Ntfy topic not available")
                return False
            
            if not message or len(message.strip()) == 0:
                logger.warning("⚠️ Attempting to send empty message")
                return False
            
            # Set priority based on message type
            priority = 3  # Default priority
            if message_type == 'warning':
                priority = 4
            elif message_type == 'error':
                priority = 5
            elif message_type == 'success':
                priority = 2
            
            # Prepare headers
            headers = {
                'Title': title if title else message_type.upper(),
                'Priority': str(priority),
                'Tags': message_type
            }
            
            # Truncate message if too long
            if len(message) > 4096:
                original_length = len(message)
                message = message[:4090] + "..."
                logger.warning(f"📝 Trimming message from {original_length} to 4096 characters")
            
            logger.info(f"📨 Sending Ntfy notification...")
            
            response = requests.post(
                self.base_url,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                logger.info("✅ Ntfy notification sent successfully")
                return True
            else:
                error_msg = f"⚠️ Failed to send Ntfy notification: {response.status_code} - {response.text}"
                logger.warning(error_msg)
                return False
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Timeout sending Ntfy message")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("🔌 Connection error sending Ntfy message")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error sending Ntfy message: {e}")
            return False

class PrecisionManager:
    """Price and Quantity Precision Manager"""
    
    def __init__(self, client):
        self.client = client
        self.symbols_info = {}
        
    def get_symbol_info(self, symbol):
        """Get currency information"""
        try:
            if symbol not in self.symbols_info:
                self._update_symbols_info()
            return self.symbols_info.get(symbol, {})
        except Exception as e:
            logger.error(f"❌ Error getting precision info for {symbol}: {e}")
            return {}
    
    def _update_symbols_info(self):
        """Update currency information"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                self.symbols_info[symbol] = {
                    'filters': symbol_info['filters'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset']
                }
            logger.info("✅ Updated precision info for currencies")
        except Exception as e:
            logger.error(f"❌ Error updating currency info: {e}")
    
    def adjust_price(self, symbol, price):
        """Adjust price according to precision"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return round(price, 4)
            
            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            if price_filter:
                tick_size = float(price_filter['tickSize'])
                return float(int(price / tick_size) * tick_size)
            return round(price, 4)
        except Exception as e:
            logger.error(f"❌ Error adjusting price for {symbol}: {e}")
            return round(price, 4)
    
    def adjust_quantity(self, symbol, quantity):
        """Adjust quantity according to precision - enhanced version"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️ No precision info for {symbol}, using default")
                return round(quantity, 3)
        
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                min_qty = float(lot_size_filter.get('minQty', 0))
                max_qty = float(lot_size_filter.get('maxQty', float('inf')))
            
                # Calculate appropriate precision
                precision = 0
                if step_size < 1:
                    precision = len(str(step_size).split('.')[1].rstrip('0'))
            
                # Adjust quantity according to step_size
                adjusted_quantity = float(int(quantity / step_size) * step_size)
            
                # Ensure limits
                adjusted_quantity = max(adjusted_quantity, min_qty)
                adjusted_quantity = min(adjusted_quantity, max_qty)
            
                # Round to appropriate precision
                adjusted_quantity = round(adjusted_quantity, precision)
            
                logger.info(f"🎯 Adjusted quantity for {symbol}: {quantity} -> {adjusted_quantity} (step: {step_size}, precision: {precision})")
            
                return adjusted_quantity
        
            # If no filter, use safe precision
            return round(quantity, 3)
        
        except Exception as e:
            logger.error(f"❌ Error adjusting quantity for {symbol}: {e}")
            return round(quantity, 3)

class DynamicStopLoss:
    """Dynamic Stop Loss System with Two Phases"""
    
    def __init__(self, atr_period=14, risk_ratio=0.5, stop_loss_phases=None, 
                 min_stop_distance=0.003, max_stop_distance=0.015):
        self.atr_period = atr_period
        self.risk_ratio = risk_ratio
        self.stop_loss_phases = stop_loss_phases or RISK_SETTINGS['stop_loss_phases']
        self.min_stop_distance = min_stop_distance
        self.max_stop_distance = max_stop_distance
    
    def calculate_atr(self, df):
        """Calculate Average True Range with error handling"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.atr_period).mean()
            return atr
        except Exception as e:
            logger.error(f"❌ Error calculating ATR: {e}")
            return pd.Series([df['close'].iloc[-1] * 0.01] * len(df))
    
    def calculate_support_resistance(self, df):
        """Calculate support and resistance levels"""
        try:
            df_with_atr = df.copy()
            df_with_atr['atr'] = self.calculate_atr(df_with_atr)
            
            if df_with_atr['atr'].isna().all() or df_with_atr['atr'].iloc[-1] == 0:
                current_price = df_with_atr['close'].iloc[-1]
                default_atr = current_price * 0.01
                df_with_atr['atr'] = default_atr
                logger.warning(f"⚠️ Using default ATR: {default_atr:.4f}")
            
            # Calculate support and resistance
            df_with_atr['resistance'] = df_with_atr['high'].rolling(20, min_periods=1).max()
            df_with_atr['support'] = df_with_atr['low'].rolling(20, min_periods=1).min()
            
            # Fill NaN values
            df_with_atr['resistance'].fillna(method='bfill', inplace=True)
            df_with_atr['support'].fillna(method='bfill', inplace=True)
            df_with_atr['atr'].fillna(method='bfill', inplace=True)
            
            return df_with_atr
            
        except Exception as e:
            logger.error(f"❌ Error calculating support/resistance: {e}")
            df_default = df.copy()
            current_price = df['close'].iloc[-1]
            df_default['atr'] = current_price * 0.01
            df_default['resistance'] = current_price * 1.02
            df_default['support'] = current_price * 0.98
            return df_default
    
    def calculate_dynamic_stop_loss(self, symbol, entry_price, direction, df):
        """Calculate stop loss with two phases"""
        try:
            support_level = df['support'].iloc[-1]
            resistance_level = df['resistance'].iloc[-1]
            current_atr = df['atr'].iloc[-1] if not df['atr'].isna().iloc[-1] else entry_price * 0.01
        
            stop_loss_levels = {}
        
            if direction == 'LONG':
                distance_to_support = entry_price - support_level
                logger.info(f"📏 Distance to support: {distance_to_support:.4f}")
            
                for phase, config in self.stop_loss_phases.items():
                    if phase == 'PHASE_1':
                        phase_distance = (distance_to_support * 0.6) + (current_atr * 0.3)
                    else:
                        phase_distance = (distance_to_support * 1.0) + (current_atr * 0.5)
                
                    stop_price = entry_price - phase_distance
                
                    min_stop = entry_price * (1 - self.max_stop_distance)
                    max_stop = entry_price * (1 - self.min_stop_distance)
                
                    if phase == 'PHASE_2' and 'PHASE_1' in stop_loss_levels:
                        previous_stop = stop_loss_levels['PHASE_1']['price']
                        stop_price = min(stop_price, previous_stop - (entry_price * 0.001))
                
                    stop_price = max(stop_price, min_stop)
                    stop_price = min(stop_price, max_stop)
                
                    stop_loss_levels[phase] = {
                        'price': stop_price,
                        'distance_ratio': config['distance_ratio'],
                        'allocation': config['allocation'],
                        'quantity': None
                    }
                
                    logger.info(f"🔧 {phase}: Distance {phase_distance:.4f}, Stop {stop_price:.4f}")
                
            else:
                distance_to_resistance = resistance_level - entry_price
                logger.info(f"📏 Distance to resistance: {distance_to_resistance:.4f}")
            
                for phase, config in self.stop_loss_phases.items():
                    if phase == 'PHASE_1':
                        phase_distance = (distance_to_resistance * 0.6) + (current_atr * 0.3)
                    else:
                        phase_distance = (distance_to_resistance * 1.0) + (current_atr * 0.5)
                
                    stop_price = entry_price + phase_distance
                
                    min_stop = entry_price * (1 + self.min_stop_distance)
                    max_stop = entry_price * (1 + self.max_stop_distance)
                
                    if phase == 'PHASE_2' and 'PHASE_1' in stop_loss_levels:
                        previous_stop = stop_loss_levels['PHASE_1']['price']
                        stop_price = max(stop_price, previous_stop + (entry_price * 0.001))
                
                    stop_price = min(stop_price, max_stop)
                    stop_price = max(stop_price, min_stop)
                
                    stop_loss_levels[phase] = {
                        'price': stop_price,
                        'distance_ratio': config['distance_ratio'],
                        'allocation': config['allocation'],
                        'quantity': None
                    }
                
                    logger.info(f"🔧 {phase}: Distance {phase_distance:.4f}, Stop {stop_price:.4f}")
        
            logger.info(f"💰 Final stop loss for {symbol}:")
            for phase, level in stop_loss_levels.items():
                distance_pct = abs(entry_price - level['price']) / entry_price * 100
                logger.info(f"   {phase}: ${level['price']:.4f} ({distance_pct:.2f}%)")
        
            return stop_loss_levels
        
        except Exception as e:
            logger.error(f"❌ Error calculating stop loss: {e}")
            return self.get_default_stop_loss(symbol, entry_price, direction)
    
    def get_default_stop_loss(self, symbol, entry_price, direction):
        """Safe default stop loss values in case of error"""
        default_levels = {}
        
        for phase, config in self.stop_loss_phases.items():
            if direction == 'LONG':
                stop_price = entry_price * (1 - self.min_stop_distance)
            else:
                stop_price = entry_price * (1 + self.min_stop_distance)
            
            default_levels[phase] = {
                'price': stop_price,
                'distance_ratio': config['distance_ratio'],
                'allocation': config['allocation'],
                'quantity': None
            }
        
        logger.warning(f"⚠️ Using safe default stop loss for {symbol}")
        return default_levels

class DynamicTakeProfit:
    """Dynamic Take Profit System"""
    
    def __init__(self, base_levels=None, volatility_multiplier=1.5):
        self.base_levels = base_levels or RISK_SETTINGS['take_profit_levels']
        self.volatility_multiplier = volatility_multiplier
    
    def calculate_dynamic_take_profit(self, symbol, entry_price, direction, df):
        """Calculate dynamic take profit with volatility adjustment"""
        try:
            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].isna().iloc[-1] else 0
            current_close = df['close'].iloc[-1]
            
            take_profit_levels = {}
            
            for level, config in self.base_levels.items():
                base_target = config['target']
                
                if current_atr > 0 and current_close > 0:
                    atr_ratio = current_atr / current_close
                    volatility_factor = 1 + (atr_ratio * self.volatility_multiplier)
                    adjusted_target = base_target * volatility_factor
                else:
                    adjusted_target = base_target
                
                if direction == 'LONG':
                    tp_price = entry_price * (1 + adjusted_target)
                else:
                    tp_price = entry_price * (1 - adjusted_target)
                
                take_profit_levels[level] = {
                    'price': tp_price,
                    'target_percent': adjusted_target * 100,
                    'allocation': config['allocation'],
                    'quantity': None
                }
            
            tp_info = [f'{level}: {config["price"]:.4f}' for level, config in take_profit_levels.items()]
            logger.info(f"🎯 Take profit for {symbol}: {tp_info}")
            return take_profit_levels
            
        except Exception as e:
            logger.error(f"❌ Error calculating take profit: {e}")
            default_levels = {}
            for level, config in self.base_levels.items():
                if direction == 'LONG':
                    tp_price = entry_price * (1 + config['target'])
                else:
                    tp_price = entry_price * (1 - config['target'])
                
                default_levels[level] = {
                    'price': tp_price,
                    'target_percent': config['target'] * 100,
                    'allocation': config['allocation'],
                    'quantity': None
                }
            return default_levels
    
    def calculate_partial_close_quantity(self, total_quantity, level_allocation):
        """Calculate quantity for partial close"""
        return total_quantity * level_allocation

class MarginMonitor:
    """Margin Monitoring and Risk Adjustment"""
    
    def __init__(self, risk_threshold=0.7, position_reduction=0.5):
        self.risk_threshold = risk_threshold
        self.position_reduction = position_reduction
    
    def check_margin_health(self, client):
        """Check margin health"""
        try:
            account_info = client.futures_account()
        
            total_wallet_balance = float(account_info['totalWalletBalance'])
            available_balance = float(account_info['availableBalance'])
            total_margin_balance = float(account_info['totalMarginBalance'])
        
            if total_wallet_balance > 0:
                margin_used = total_wallet_balance - available_balance
                margin_ratio = margin_used / total_wallet_balance
            
                return {
                    'total_wallet_balance': total_wallet_balance,
                    'available_balance': available_balance,
                    'total_margin_balance': total_margin_balance,
                    'margin_used': margin_used,
                    'margin_ratio': margin_ratio,
                    'is_risk_high': margin_ratio > self.risk_threshold
                }
            return None
        
        except Exception as e:
            logger.error(f"❌ Error checking margin: {e}")
            return None

class UnifiedTradeManager:
    """Integrated Trade Manager"""
    
    def __init__(self, client, notifier):
        self.client = client
        self.notifier = notifier
        self.precision_manager = PrecisionManager(client)
        self.stop_loss_manager = DynamicStopLoss()
        self.take_profit_manager = DynamicTakeProfit()
        self.margin_monitor = MarginMonitor()
        
        # Data storage
        self.managed_trades = {}  # Managed trades
        self.active_trades = {}   # Temporarily open trades
        self.trade_history = []   # Trade history
        
        # Performance statistics
        self.performance_stats = {
            'total_trades_managed': 0,
            'profitable_trades': 0,
            'stopped_trades': 0,
            'take_profit_hits': 0,
            'timeout_trades': 0,
            'total_pnl': 0
        }
        
        # Cache settings
        self.last_heartbeat = datetime.now(damascus_tz)
        self.symbols_info = {}
        self.price_cache = {}
        self.cache_timeout = 30
        self.last_api_call = {}
        
        # Start helper services
        self.start_periodic_cleanup()
        logger.info("✅ Integrated Trade Manager initialized")
    
    def start_periodic_cleanup(self):
        """Start periodic cleanup of closed and pending trades"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)
                    self.cleanup_pending_trades()
                    logger.info("🔄 Periodic cleanup of closed and pending trades")
                except Exception as e:
                    logger.error(f"❌ Error in periodic cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _get_current_price(self, symbol):
        """Get current price"""
        for attempt in range(3):
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                if price > 0:
                    return price
            except Exception as e:
                if attempt == 2:
                    logger.error(f"❌ Error getting price for {symbol}: {e}")
                time.sleep(1)
        return None
    
    def get_current_price(self, symbol):
        """Get current price with caching system"""
        try:
            current_time = time.time()
            
            if (symbol in self.price_cache and 
                current_time - self.price_cache[symbol]['timestamp'] < self.cache_timeout):
                return self.price_cache[symbol]['price']
            
            if symbol in self.last_api_call:
                time_since_last_call = current_time - self.last_api_call[symbol]
                if time_since_last_call < 1:
                    time.sleep(1 - time_since_last_call)
            
            time.sleep(np.random.uniform(0.5, 1.0))
            
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            
            self.price_cache[symbol] = {
                'price': price,
                'timestamp': current_time
            }
            
            self.last_api_call[symbol] = current_time
            
            return price
            
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def cleanup_pending_trades(self):
        """Detect and clean up pending trades"""
        try:
            pending_trades = []
        
            positions = self.client.futures_account()['positions']
            active_symbols_in_binance = set()
        
            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt != 0:
                    active_symbols_in_binance.add(position['symbol'])
        
            for trade_id, trade in list(self.active_trades.items()):
                if trade['status'] == 'open':
                    symbol = trade['symbol']
                
                    if symbol not in active_symbols_in_binance:
                        pending_trades.append(trade_id)
                        logger.warning(f"🔍 Detected pending trade: {trade_id}")
        
            for trade_id in pending_trades:
                trade = self.active_trades[trade_id]
                trade.update({
                    'status': 'closed',
                    'close_price': trade.get('current_price', trade['entry_price']),
                    'close_time': datetime.now(damascus_tz),
                    'close_reason': 'Pending detection - Trade not found in Binance',
                    'pnl_pct': 0,
                    'pnl_usd': 0
                })
                logger.info(f"🧹 Cleaning pending trade: {trade_id}")
        
            if pending_trades:
                logger.info(f"✅ Cleaned {len(pending_trades)} pending trades")
            
                if self.notifier and pending_trades:
                    message = f"🧹 <b>Pending Trade Cleanup</b>\nDetected and cleaned {len(pending_trades)} pending trades"
                    self.notifier.send_message(message, title="Trade Cleanup")
        
            return len(pending_trades)
        
        except Exception as e:
            logger.error(f"❌ Error cleaning pending trades: {e}")
            return 0
    
    def sync_with_binance_positions(self):
        """Sync active trades with actual positions in Binance"""
        try:
            positions = self.client.futures_account()['positions']
            
            active_symbols_in_binance = set()
            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt != 0:
                    active_symbols_in_binance.add(position['symbol'])
            
            trades_to_close = []
            
            for trade_id, trade in list(self.active_trades.items()):
                symbol = trade['symbol']
                
                if trade['status'] == 'open':
                    if symbol not in active_symbols_in_binance:
                        logger.warning(f"🔄 Detected closed trade in Binance: {trade_id}")
                        trades_to_close.append(trade_id)
                    else:
                        current_price = self._get_current_price(symbol)
                        if current_price:
                            trade['current_price'] = current_price
            
            for trade_id in trades_to_close:
                trade = self.active_trades[trade_id]
                trade.update({
                    'status': 'closed',
                    'close_price': trade.get('current_price', trade['entry_price']),
                    'close_time': datetime.now(damascus_tz),
                    'close_reason': 'Detected closure from Binance',
                    'pnl_pct': 0,
                    'pnl_usd': 0
                })
                logger.info(f"✅ Synced trade closure: {trade_id}")
            
            return len(trades_to_close)
            
        except Exception as e:
            logger.error(f"❌ Error syncing trades with Binance: {e}")
            return 0
    
    def get_trade_level(self, confidence_score):
        """Determine trading level based on confidence score"""
        for level_name, level_config in TRADING_LEVELS.items():
            if level_config['min_confidence'] <= confidence_score <= level_config['max_confidence']:
                return level_name, level_config['allocation']
        return None, None
    
    def calculate_position_size(self, symbol, current_price, confidence_score):
        """Calculate position size according to confidence level"""
        try:
            trade_level, allocation = self.get_trade_level(confidence_score)
            if not trade_level:
                return None, None, 0
            
            total_size = TRADING_SETTINGS['position_size']
            allocated_size = total_size * allocation
            
            quantity = allocated_size / current_price
            adjusted_quantity = self.precision_manager.adjust_quantity(symbol, quantity)
            
            if adjusted_quantity > 0:
                logger.info(f"💰 Trade size for {symbol} - Level {trade_level}: {adjusted_quantity:.6f}")
                return adjusted_quantity, allocated_size, trade_level
            
            return None, None, 0
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return None, None, 0
    
    def can_execute_trade(self, symbol, direction):
        """Check if trade can be executed"""
        try:
            # Initial sync
            self.sync_with_binance_positions()
        
            # Check Binance directly
            positions = self.client.futures_account()['positions']
            active_symbols = []
            symbol_positions = 0
        
            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt != 0:
                    if position['symbol'] == symbol:
                        symbol_positions += 1
                    active_symbols.append(position['symbol'])
        
            # Check max allowed per symbol
            max_per_symbol = TRADING_SETTINGS['max_trades_per_symbol']
            if symbol_positions >= max_per_symbol:
                return False, f"Reached max trades for {symbol} ({symbol_positions}/{max_per_symbol})"
        
            # Check total count
            unique_active_symbols = [s for s in active_symbols if s in TRADING_SETTINGS['symbols']]
            total_active_trades = len(set(unique_active_symbols))
        
            max_simultaneous = TRADING_SETTINGS['max_simultaneous_trades']
            if total_active_trades >= max_simultaneous:
                return False, f"Reached max active trades: {total_active_trades}/{max_simultaneous}"
        
            # Check balance
            balance_info = self.client.futures_account_balance()
            usdt_balance = next((float(b['balance']) for b in balance_info if b['asset'] == 'USDT'), 0)
        
            required_margin = TRADING_SETTINGS['base_trade_amount']
            min_balance_required = TRADING_SETTINGS.get('min_balance_required', 2)
        
            if usdt_balance < min_balance_required:
                return False, f"Insufficient total balance: {usdt_balance:.2f} USDT"
        
            account_info = self.client.futures_account()
            available_balance = float(account_info.get('availableBalance', 0))
        
            if available_balance < required_margin:
                return False, f"Insufficient available balance: {available_balance:.2f} USDT"
        
            logger.info(f"✅ Can execute {symbol} trade - all conditions met")
            return True, "Trade can be executed"

        except Exception as e:
            logger.error(f"❌ Error checking execution feasibility: {e}")
            return False, f"Check error: {str(e)}"
    
    def execute_trade(self, signal_data):
        """Execute new trade based on signal"""
        try:
            required_fields = ['symbol', 'direction', 'signal_type', 'confidence_score']
            for field in required_fields:
                if field not in signal_data:
                    return False, f"Missing field: {field}"
            
            symbol = signal_data['symbol']
            direction = signal_data['direction']
            signal_type = signal_data['signal_type']
            confidence_score = signal_data['confidence_score']
            
            if direction not in ['LONG', 'SHORT']:
                return False, f"Invalid direction: {direction}"
            
            # Check execution feasibility
            can_execute, message = self.can_execute_trade(symbol, direction)
            if not can_execute:
                return False, message
            
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                return False, "Cannot get price"
            
            # Calculate position size
            quantity, allocated_size, trade_level = self.calculate_position_size(symbol, current_price, confidence_score)
            if not quantity:
                return False, "Cannot calculate position size"
            
            # Set leverage
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=TRADING_SETTINGS['leverage'])
            except:
                logger.warning(f"⚠️ Error setting leverage for {symbol}")
            
            # Execute market order
            side = 'BUY' if direction == 'LONG' else 'SELL'
            
            logger.info(f"⚡ Executing {symbol} trade: {direction} | Level: {trade_level}")
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            if order and order.get('orderId'):
                executed_price = current_price
                try:
                    order_info = self.client.futures_get_order(symbol=symbol, orderId=order['orderId'])
                    if order_info and order_info.get('avgPrice'):
                        executed_price = float(order_info['avgPrice'])
                except:
                    logger.warning(f"⚠️ Cannot get execution price for {symbol}")
                
                # Save open trade data
                trade_id = f"{symbol}_{int(time.time())}"
                self.active_trades[trade_id] = {
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': executed_price,
                    'side': direction,
                    'order_id': order['orderId'],
                    'signal_type': signal_type,
                    'trade_level': trade_level,
                    'confidence_score': confidence_score,
                    'allocated_size': allocated_size,
                    'timestamp': datetime.now(damascus_tz),
                    'status': 'open'
                }
                
                # Automatically start trade management
                self.start_trade_management(symbol, executed_price, direction, quantity)
                
                # Send success notification
                if self.notifier:
                    message = (
                        f"✅ <b>New Trade Executed - Level {trade_level}</b>\n"
                        f"Symbol: {symbol}\n"
                        f"Direction: {direction}\n"
                        f"Level: {trade_level}\n"
                        f"Confidence: {confidence_score}%\n"
                        f"Quantity: {quantity:.6f}\n"
                        f"Size: ${allocated_size:.2f}\n"
                        f"Entry Price: ${executed_price:.4f}\n"
                        f"Order ID: {order['orderId']}\n"
                        f"📢 <b>Automatic trade management started</b>\n"
                        f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
                    )
                    self.notifier.send_message(message, title="New Trade", message_type='success')
                
                logger.info(f"✅ {direction} trade for {symbol} executed successfully")
                return True, f"Execution successful - Entry price: {executed_price:.4f}"
            
            else:
                logger.error(f"❌ Failed to execute order for {symbol}")
                return False, "Order execution failed"
                
        except Exception as e:
            logger.error(f"❌ Failed to execute trade: {e}")
            return False, f"Execution error: {str(e)}"
    
    def start_trade_management(self, symbol, entry_price, direction, quantity):
        """Automatically start trade management"""
        try:
            # Get price data for market analysis
            klines = self.client.futures_klines(symbol=symbol, interval='15m', limit=20)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate stop loss levels
            df_with_levels = self.stop_loss_manager.calculate_support_resistance(df)
            stop_loss_levels = self.stop_loss_manager.calculate_dynamic_stop_loss(
                symbol, entry_price, direction, df_with_levels
            )
            
            # Calculate take profit levels
            take_profit_levels = self.take_profit_manager.calculate_dynamic_take_profit(
                symbol, entry_price, direction, df_with_levels
            )
            
            # Calculate quantities for each level
            for phase, config in stop_loss_levels.items():
                raw_quantity = quantity * config['allocation']
                config['quantity'] = self.precision_manager.adjust_quantity(symbol, raw_quantity)
            
            for level, config in take_profit_levels.items():
                raw_quantity = quantity * config['allocation']
                config['quantity'] = self.precision_manager.adjust_quantity(symbol, raw_quantity)
            
            # Set trade expiry time
            initial_expiry = datetime.now(damascus_tz) + timedelta(hours=RISK_SETTINGS['max_trade_duration_hours'])
            
            # Save management data
            self.managed_trades[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'direction': direction,
                'stop_loss_levels': stop_loss_levels,
                'take_profit_levels': take_profit_levels,
                'closed_stop_levels': [],
                'closed_tp_levels': [],
                'last_update': datetime.now(damascus_tz),
                'status': 'managed',
                'management_start': datetime.now(damascus_tz),
                'trade_expiry': initial_expiry,
                'extension_used': False,
                'final_extension_used': False,
                'initial_direction_check': None
            }
            
            self.performance_stats['total_trades_managed'] += 1
            
            # Send management start notification
            if self.notifier:
                message = (
                    f"🔄 <b>Starting Trade Management</b>\n"
                    f"Symbol: {symbol}\n"
                    f"Direction: {direction}\n"
                    f"Entry Price: ${entry_price:.4f}\n"
                    f"Quantity: {quantity:.6f}\n"
                    f"⏰ First check after: 1 hour\n"
                    f"🔄 Extension system: Active\n"
                    f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
                )
                self.notifier.send_message(message, title="Trade Management Started")
            
            logger.info(f"✅ Started automatic management for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start trade management for {symbol}: {e}")
            return False
    
    def calculate_pnl_percentage(self, trade, current_price):
        """Calculate profit/loss percentage"""
        if trade['direction'] == 'LONG':
            return (current_price - trade['entry_price']) / trade['entry_price'] * 100
        else:
            return (trade['entry_price'] - current_price) / trade['entry_price'] * 100
    
    def check_managed_trades(self):
        """Check managed trades"""
        closed_trades = []
    
        for symbol, trade in list(self.managed_trades.items()):
            try:
                # Check if position still exists in Binance
                positions = self.client.futures_account()['positions']
                current_position = None
                for position in positions:
                    if position['symbol'] == symbol:
                        position_amt = float(position['positionAmt'])
                        if position_amt != 0:
                            current_position = position
                            break
                
                if not current_position:
                    logger.info(f"🔄 Position became zero for {symbol} - removing from management")
                    if symbol in self.managed_trades:
                        del self.managed_trades[symbol]
                    closed_trades.append(symbol)
                    continue
            
                # Get current price
                current_price = self.get_current_price(symbol)
                if not current_price:
                    time.sleep(2)
                    continue
            
                # 1. Check extension system
                if self.check_trade_extension(symbol, current_price):
                    closed_trades.append(symbol)
                    continue
            
                # 2. Check stop loss
                if self.check_stop_loss(symbol, current_price):
                    closed_trades.append(symbol)
                    continue
            
                # 3. Check take profits
                self.check_take_profits(symbol, current_price)
            
                # 4. Update dynamic levels every hour
                if (datetime.now(damascus_tz) - trade['last_update']).seconds > 3600:
                    self.update_dynamic_levels(symbol)
            
                time.sleep(3)
            
            except Exception as e:
                logger.error(f"❌ Error checking trade {symbol}: {e}")
                time.sleep(5)
    
        return closed_trades
    
    def check_trade_extension(self, symbol, current_price):
        """Check trade extension"""
        try:
            trade = self.managed_trades[symbol]
            current_time = datetime.now(damascus_tz)
            current_pnl_pct = self.calculate_pnl_percentage(trade, current_price)
            
            # First check after 1 hour
            if (not trade['extension_used'] and 
                current_time >= trade['trade_expiry']):
                
                logger.info(f"⏰ Direction check after 1 hour for {symbol}: PnL = {current_pnl_pct:+.2f}%")
                
                if current_pnl_pct >= 0:
                    new_expiry = current_time + timedelta(minutes=RISK_SETTINGS['extension_duration_minutes'])
                    self.managed_trades[symbol]['trade_expiry'] = new_expiry
                    self.managed_trades[symbol]['extension_used'] = True
                    self.managed_trades[symbol]['initial_direction_check'] = 'PROFIT'
                    
                    logger.info(f"✅ Extending {symbol} 30 minutes - direction in profit")
                    self.send_extension_notification(trade, current_price, current_pnl_pct, "30 minutes", "Direction in profit")
                    
                else:
                    logger.warning(f"🚨 Closing {symbol} - direction against profit after 1 hour")
                    success, message = self.close_entire_trade(symbol, "Direction against profit after 1 hour")
                    if success:
                        self.performance_stats['timeout_trades'] += 1
                        self.send_timeout_notification(trade, current_price, current_pnl_pct, "Direction against profit")
                        return True
            
            # Second check after 1.5 hours
            elif (trade['extension_used'] and not trade['final_extension_used'] and
                  current_time >= trade['trade_expiry']):
                
                logger.info(f"⏰ Direction check after 1.5 hours for {symbol}: PnL = {current_pnl_pct:+.2f}%")
                
                if current_pnl_pct >= 0:
                    new_expiry = current_time + timedelta(minutes=RISK_SETTINGS['final_extension_minutes'])
                    self.managed_trades[symbol]['trade_expiry'] = new_expiry
                    self.managed_trades[symbol]['final_extension_used'] = True
                    
                    logger.info(f"✅ Extending {symbol} additional 30 minutes - still in profit")
                    self.send_extension_notification(trade, current_price, current_pnl_pct, "Additional 30 minutes", "Still in profit")
                    
                else:
                    logger.warning(f"🚨 Closing {symbol} - turned against profit after 1.5 hours")
                    success, message = self.close_entire_trade(symbol, "Turned against profit after 1.5 hours")
                    if success:
                        self.performance_stats['timeout_trades'] += 1
                        self.send_timeout_notification(trade, current_price, current_pnl_pct, "Turned against profit")
                        return True
            
            # Final check after 2 hours
            elif (trade['final_extension_used'] and 
                  current_time >= trade['trade_expiry']):
                
                logger.warning(f"⏰ Final time expiry for {symbol} - forced closure")
                success, message = self.close_entire_trade(symbol, "Final time expiry (2 hours)")
                if success:
                    self.performance_stats['timeout_trades'] += 1
                    self.send_timeout_notification(trade, current_price, current_pnl_pct, "Final time expiry")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking trade extension for {symbol}: {e}")
            return False
    
    def check_stop_loss(self, symbol, current_price):
        """Check stop loss"""
        trade = self.managed_trades[symbol]
        
        for phase, config in trade['stop_loss_levels'].items():
            if phase in trade['closed_stop_levels']:
                continue
            
            should_close = False
            reason = f"Stop loss {phase}"
            
            if trade['direction'] == 'LONG' and current_price <= config['price']:
                should_close = True
            elif trade['direction'] == 'SHORT' and current_price >= config['price']:
                should_close = True
            
            if should_close:
                logger.info(f"🚨 Should close part of {symbol} due to {reason}")
                success = self.close_partial_position(symbol, config['quantity'], trade['direction'], reason)
                if success:
                    trade['closed_stop_levels'].append(phase)
                    logger.info(f"✅ Closed stop loss {phase} for {symbol}")
                    
                    self.send_stop_loss_notification(trade, phase, current_price, config)
                    
                    if len(trade['closed_stop_levels']) == len(trade['stop_loss_levels']):
                        self.close_entire_trade(symbol, "All stop loss levels triggered")
                        self.performance_stats['stopped_trades'] += 1
                    return True
        
        return False
    
    def check_take_profits(self, symbol, current_price):
        """Check take profit levels"""
        trade = self.managed_trades[symbol]
        
        for level, config in trade['take_profit_levels'].items():
            if level in trade['closed_tp_levels']:
                continue
            
            should_close = False
            if trade['direction'] == 'LONG' and current_price >= config['price']:
                should_close = True
            elif trade['direction'] == 'SHORT' and current_price <= config['price']:
                should_close = True
            
            if should_close:
                success = self.close_partial_position(symbol, config['quantity'], trade['direction'], f"Take profit {level}")
                if success:
                    trade['closed_tp_levels'].append(level)
                    self.performance_stats['take_profit_hits'] += 1
                    
                    self.send_take_profit_notification(trade, level, current_price)
                    
                    if len(trade['closed_tp_levels']) == len(trade['take_profit_levels']):
                        self.ensure_complete_closure(symbol, "All take profit levels achieved")
                        self.performance_stats['profitable_trades'] += 1
    
    def close_partial_position(self, symbol, quantity, direction, reason):
        """Partial position closure"""
        try:
            side = 'SELL' if direction == 'LONG' else 'BUY'
            
            adjusted_quantity = self.precision_manager.adjust_quantity(symbol, quantity)
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=adjusted_quantity,
                reduceOnly=True
            )
            
            if order:
                logger.info(f"✅ Partial close for {symbol}: {adjusted_quantity:.6f} - Reason: {reason}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"❌ Error in partial close for {symbol}: {e}")
            return False
    
    def close_entire_trade(self, symbol, reason):
        """Full trade closure"""
        try:
            if symbol not in self.managed_trades:
                return False, "Trade not found in management"
        
            trade = self.managed_trades[symbol]
            quantity = trade['quantity']
            direction = trade['direction']
            
            side = 'SELL' if direction == 'LONG' else 'BUY'
            
            adjusted_quantity = self.precision_manager.adjust_quantity(symbol, quantity)
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=adjusted_quantity,
                reduceOnly=True
            )
            
            if order:
                current_price = self.get_current_price(symbol) or trade['entry_price']
                pnl_pct = self.calculate_pnl_percentage(trade, current_price)
                
                logger.info(f"✅ Successful full close for {symbol}: {reason} - PnL: {pnl_pct:+.2f}%")
                
                # Update statistics
                self.performance_stats['total_pnl'] += pnl_pct
                
                # Send closure notification
                if self.notifier:
                    pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
                    message = (
                        f"🔒 <b>Trade Closed</b>\n"
                        f"Symbol: {symbol}\n"
                        f"Direction: {direction}\n"
                        f"Entry Price: ${trade['entry_price']:.4f}\n"
                        f"Exit Price: ${current_price:.4f}\n"
                        f"PnL: {pnl_emoji} {pnl_pct:+.2f}%\n"
                        f"Reason: {reason}\n"
                        f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
                    )
                    self.notifier.send_message(message, title="Trade Closed", 
                                              message_type='success' if pnl_pct > 0 else 'warning')
                
                # Remove trade from management
                if symbol in self.managed_trades:
                    del self.managed_trades[symbol]
                
                return True, "Closed successfully"
            
            return False, "Failed to create order"
        
        except Exception as e:
            logger.error(f"❌ Error in full close for {symbol}: {e}")
            return False, str(e)
    
    def ensure_complete_closure(self, symbol, reason):
        """Ensure complete position closure"""
        try:
            positions = self.client.futures_account()['positions']
            for position in positions:
                if position['symbol'] == symbol:
                    position_amt = float(position['positionAmt'])
                    if position_amt != 0:
                        logger.warning(f"⚠️ Remaining unclosed position for {symbol}: {position_amt}")
                        success, message = self.close_entire_trade(symbol, f"{reason} - cleaning remaining quantity")
                        return success
            
            logger.info(f"✅ Position fully closed for {symbol}")
            if symbol in self.managed_trades:
                del self.managed_trades[symbol]
            return True
            
        except Exception as e:
            logger.error(f"❌ Error ensuring complete closure for {symbol}: {e}")
            return False
    
    def update_dynamic_levels(self, symbol):
        """Update stop loss and take profit levels"""
        if symbol not in self.managed_trades:
            return
        
        trade = self.managed_trades[symbol]
        
        try:
            klines = self.client.futures_klines(symbol=symbol, interval='15m', limit=20)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            
            df = self.stop_loss_manager.calculate_support_resistance(df)
            new_stop_loss_levels = self.stop_loss_manager.calculate_dynamic_stop_loss(
                symbol, trade['entry_price'], trade['direction'], df
            )
            
            for phase, new_level in new_stop_loss_levels.items():
                if phase in trade['stop_loss_levels']:
                    current_level = trade['stop_loss_levels'][phase]
                    
                    if (trade['direction'] == 'LONG' and new_level['price'] > current_level['price']) or \
                       (trade['direction'] == 'SHORT' and new_level['price'] < current_level['price']):
                        self.managed_trades[symbol]['stop_loss_levels'][phase] = new_level
            
            self.managed_trades[symbol]['last_update'] = datetime.now(damascus_tz)
            logger.info(f"🔄 Updated levels for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error updating levels for {symbol}: {e}")
    
    def monitor_margin_risk(self):
        """Monitor margin risk"""
        try:
            time.sleep(np.random.uniform(1, 3))
        
            margin_health = self.margin_monitor.check_margin_health(self.client)
        
            if margin_health and margin_health['is_risk_high']:
                logger.warning(f"🚨 High risk level: {margin_health['margin_ratio']:.2%}")
                self.send_margin_warning(margin_health)
                return True
            return False
        
        except Exception as e:
            logger.error(f"❌ Error checking margin: {e}")
            return False
    
    def send_heartbeat(self):
        """Send heartbeat"""
        try:
            current_time = datetime.now(damascus_tz)
            
            if (current_time - self.last_heartbeat).seconds >= 7200:
                message = (
                    f"💓 <b>Heartbeat - Integrated Bot Working Successfully</b>\n"
                    f"Status: Active and Stable ✅\n"
                    f"Managed Trades: {len(self.managed_trades)}\n"
                    f"Total Trades: {self.performance_stats['total_trades_managed']}\n"
                    f"Last Update: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Timezone: Damascus"
                )
                
                success = self.notifier.send_message(message, title="Bot Heartbeat")
                if success:
                    self.last_heartbeat = current_time
                    logger.info("💓 Sent heartbeat")
                return success
            return False
            
        except Exception as e:
            logger.error(f"❌ Error sending heartbeat: {e}")
            return False
    
    def send_extension_notification(self, trade, current_price, pnl_pct, extension_type, reason):
        """Send trade extension notification"""
        try:
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            
            message = (
                f"⏰ <b>Trade Extension</b>\n"
                f"Symbol: {trade['symbol']}\n"
                f"Direction: {trade['direction']}\n"
                f"Entry Price: ${trade['entry_price']:.4f}\n"
                f"Market Price: ${current_price:.4f}\n"
                f"PnL: {pnl_emoji} {pnl_pct:+.2f}%\n"
                f"Extension: {extension_type}\n"
                f"Reason: {reason}\n"
                f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message, title="Trade Extended")
            
        except Exception as e:
            logger.error(f"❌ Error sending extension notification: {e}")
            return False
    
    def send_stop_loss_notification(self, trade, phase, current_price, config):
        """Send partial stop loss notification"""
        try:
            pnl_pct = self.calculate_pnl_percentage(trade, current_price)
            pnl_emoji = "🟡"
            
            message = (
                f"🛑 <b>Partial Stop Loss</b>\n"
                f"Symbol: {trade['symbol']}\n"
                f"Phase: {phase}\n"
                f"Entry Price: ${trade['entry_price']:.4f}\n"
                f"Stop Price: ${config['price']:.4f}\n"
                f"Market Price: ${current_price:.4f}\n"
                f"PnL: {pnl_emoji} {pnl_pct:+.2f}%\n"
                f"Quantity: {config['quantity']:.6f}\n"
                f"Remaining Levels: {len(trade['stop_loss_levels']) - len(trade['closed_stop_levels'])}\n"
                f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message, title="Stop Loss Triggered", message_type='warning')
            
        except Exception as e:
            logger.error(f"❌ Error sending stop loss notification: {e}")
            return False
    
    def send_take_profit_notification(self, trade, level, current_price):
        """Send take profit notification"""
        try:
            config = trade['take_profit_levels'][level]
            pnl_pct = self.calculate_pnl_percentage(trade, current_price)
            
            message = (
                f"🎯 <b>Partial Take Profit</b>\n"
                f"Symbol: {trade['symbol']}\n"
                f"Level: {level}\n"
                f"Entry Price: ${trade['entry_price']:.4f}\n"
                f"Take Price: ${current_price:.4f}\n"
                f"Current Profit: {pnl_pct:+.2f}%\n"
                f"Target Profit: {config['target_percent']:.2f}%\n"
                f"Quantity: {config['quantity']:.6f}\n"
                f"Remaining Levels: {len(trade['take_profit_levels']) - len(trade['closed_tp_levels'])}\n"
                f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message, title="Take Profit Achieved", message_type='success')
            
        except Exception as e:
            logger.error(f"❌ Error sending take profit notification: {e}")
            return False
    
    def send_timeout_notification(self, trade, current_price, pnl_pct, reason):
        """Send trade timeout notification"""
        try:
            pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
            
            management_duration = datetime.now(damascus_tz) - trade['management_start']
            hours = management_duration.seconds // 3600
            minutes = (management_duration.seconds % 3600) // 60
            
            message = (
                f"⏰ <b>Trade Closed - {reason}</b>\n"
                f"Symbol: {trade['symbol']}\n"
                f"Direction: {trade['direction']}\n"
                f"Entry Price: ${trade['entry_price']:.4f}\n"
                f"Exit Price: ${current_price:.4f}\n"
                f"PnL: {pnl_emoji} {pnl_pct:+.2f}%\n"
                f"Management Duration: {hours}h {minutes}m\n"
                f"Reason: {reason}\n"
                f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
        
            return self.notifier.send_message(message, title="Trade Timeout", 
                                            message_type='warning')
        
        except Exception as e:
            logger.error(f"❌ Error sending timeout notification: {e}")
            return False
    
    def send_margin_warning(self, margin_health):
        """Send margin warning"""
        try:
            message = (
                f"⚠️ <b>Warning: High Risk Level</b>\n"
                f"Margin Used Ratio: {margin_health['margin_ratio']:.2%}\n"
                f"Available Balance: ${margin_health['available_balance']:.2f}\n"
                f"Total Balance: ${margin_health['total_wallet_balance']:.2f}\n"
                f"Status: Continuous Monitoring ⚠️\n"
                f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message, title="Margin Warning", message_type='error')
            
        except Exception as e:
            logger.error(f"❌ Error sending margin warning: {e}")
            return False

class SimpleSignalReceiver:
    """Simple Signal Receiver"""
    
    def __init__(self, trade_manager):
        self.trade_manager = trade_manager
        self.received_signals = []
    
    def process_signal(self, signal_data):
        """Process signal from external bot"""
        try:
            logger.info(f"📨 Receiving new signal: {signal_data}")
        
            # Validate signal
            if not self._validate_signal(signal_data):
                return False, "Invalid signal"
        
            # Save signal
            signal_data['received_time'] = datetime.now(damascus_tz)
            signal_data['processed'] = False
            self.received_signals.append(signal_data)
        
            signal_type = signal_data.get('signal_type', 'UNKNOWN')
        
            if signal_type == 'OPEN_TRADE':
                symbol = signal_data['symbol']
            
                # Check execution feasibility
                can_execute, message = self.trade_manager.can_execute_trade(symbol, signal_data['direction'])
            
                if not can_execute:
                    signal_data['result'] = 'FAILED'
                    signal_data['error_reason'] = message
                    return False, message
            
                # Execute trade
                success, message = self.trade_manager.execute_trade(signal_data)
                if success:
                    signal_data['processed'] = True
                    signal_data['result'] = 'SUCCESS'
                else:
                    signal_data['result'] = 'FAILED'
                    signal_data['error_reason'] = message
                return success, message
        
            elif signal_type == 'CLOSE_TRADE':
                symbol = signal_data.get('symbol')
                reason = signal_data.get('reason', 'External signal closure')
                if symbol and symbol in self.trade_manager.managed_trades:
                    success, message = self.trade_manager.close_entire_trade(symbol, reason)
                    if success:
                        signal_data['processed'] = True
                        signal_data['result'] = 'SUCCESS'
                    else:
                        signal_data['result'] = 'FAILED'
                        signal_data['error_reason'] = message
                    return success, message
                else:
                    signal_data['result'] = 'FAILED'
                    return False, f"No open trades for {symbol}"
        
            else:
                signal_data['result'] = 'FAILED'
                return False, f"Unknown signal type: {signal_type}"
            
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
            if 'signal_data' in locals():
                signal_data['result'] = 'ERROR'
                signal_data['error'] = str(e)
            return False, f"Processing error: {str(e)}"
    
    def _validate_signal(self, signal_data):
        """Validate signal"""
        required_fields = ['symbol', 'direction', 'signal_type', 'confidence_score']
        
        for field in required_fields:
            if field not in signal_data:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        symbol = signal_data['symbol']
        if symbol not in TRADING_SETTINGS['symbols']:
            logger.error(f"❌ Unsupported currency: {symbol}")
            return False
        
        if signal_data['direction'] not in ['LONG', 'SHORT']:
            logger.error(f"❌ Invalid direction: {signal_data['direction']}")
            return False
        
        confidence_score = signal_data['confidence_score']
        if confidence_score < 25:
            logger.error(f"❌ Insufficient confidence: {confidence_score}%")
            return False
        
        return True
    
    def get_recent_signals(self, limit=10):
        """Get recently received signals"""
        return self.received_signals[-limit:]

def convert_signal_format(signal_data):
    """Convert signal format"""
    try:
        if 'symbol' not in signal_data or 'action' not in signal_data:
            logger.error("❌ Signal missing basic fields")
            return None
        
        confidence_score = signal_data.get('confidence_score', 0)
        if confidence_score < 25:
            logger.error(f"❌ Insufficient confidence: {confidence_score}%")
            return None
        
        symbol = signal_data['symbol']
        action = signal_data['action'].upper()
        
        if action == 'BUY':
            direction = 'LONG'
            signal_type = 'OPEN_TRADE'
        elif action == 'SELL':
            direction = 'SHORT' 
            signal_type = 'OPEN_TRADE'
        else:
            logger.error(f"❌ Unknown action: {action}")
            return None
        
        converted_signal = {
            'symbol': symbol,
            'direction': direction,
            'signal_type': signal_type,
            'confidence_score': signal_data.get('confidence_score', 0),
            'original_signal': signal_data,
            'reason': signal_data.get('reason', 'Signal from sender bot'),
            'source': 'top_bottom_scanner'
        }
        
        if 'coin' in signal_data:
            converted_signal['coin'] = signal_data['coin']
        if 'timeframe' in signal_data:
            converted_signal['timeframe'] = signal_data['timeframe']
        if 'analysis' in signal_data:
            converted_signal['analysis'] = signal_data['analysis']
        
        logger.info(f"✅ Signal converted: {action} -> {direction}")
        return converted_signal
        
    except Exception as e:
        logger.error(f"❌ Error converting signal format: {e}")
        return None

class UnifiedTradeBot:
    """Main Integrated Bot"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if UnifiedTradeBot._instance is not None:
            raise Exception("This class uses Singleton pattern")
        
        # Get API keys
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.ntfy_topic = os.environ.get('NTFY_TOPIC')
        
        if not all([self.api_key, self.api_secret]):
            raise ValueError("Binance API keys required")
        
        # Initialize client
        try:
            self.client = Client(self.api_key, self.api_secret)
            self.test_connection()
        except Exception as e:
            logger.error(f"❌ Failed to initialize client: {e}")
            raise
        
        # Initialize components
        self.notifier = NtfyNotifier(self.ntfy_topic)
        self.trade_manager = UnifiedTradeManager(self.client, self.notifier)
        self.signal_receiver = SimpleSignalReceiver(self.trade_manager)
        
        UnifiedTradeBot._instance = self
        logger.info("✅ Integrated Bot initialized successfully")
    
    def test_connection(self):
        """Test connection"""
        try:
            self.client.futures_time()
            logger.info("✅ Binance API connection active")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Binance API: {e}")
            raise
    
    def get_status(self):
        """Get bot status"""
        return {
            'status': 'running',
            'managed_trades': len(self.trade_manager.managed_trades),
            'active_trades': len(self.trade_manager.active_trades),
            'max_simultaneous_trades': TRADING_SETTINGS['max_simultaneous_trades'],
            'total_signals_received': len(self.signal_receiver.received_signals),
            'performance_stats': self.trade_manager.performance_stats,
            'timestamp': datetime.now(damascus_tz).isoformat()
        }

    def management_loop(self):
        """Main management loop"""
        last_report_time = datetime.now(damascus_tz)
        last_sync_time = datetime.now(damascus_tz)
        last_margin_check = datetime.now(damascus_tz)
        last_heartbeat_time = datetime.now(damascus_tz)
    
        while True:
            try:
                current_time = datetime.now(damascus_tz)
            
                # Send heartbeat every 2 hours
                if (current_time - last_heartbeat_time).seconds >= 7200:
                    self.trade_manager.send_heartbeat()
                    last_heartbeat_time = current_time
            
                # Check managed trades every 30 seconds
                self.trade_manager.check_managed_trades()
            
                # Monitor margin every 5 minutes
                if (current_time - last_margin_check).seconds >= 300:
                    self.trade_manager.monitor_margin_risk()
                    last_margin_check = current_time
            
                # Sync trades every 10 minutes
                if (current_time - last_sync_time).seconds >= 600:
                    self.trade_manager.sync_with_binance_positions()
                    last_sync_time = current_time
            
                # Performance report every 6 hours
                if (current_time - last_report_time).seconds >= 21600:
                    self.send_performance_report()
                    last_report_time = current_time
            
                time.sleep(30)
            
            except KeyboardInterrupt:
                logger.info("⏹️ Manual bot stop...")
                break
            except Exception as e:
                logger.error(f"❌ Error in management loop: {e}")
                time.sleep(60)
    
    def send_performance_report(self):
        """Send performance report"""
        try:
            stats = self.trade_manager.performance_stats
            
            if stats['total_trades_managed'] > 0:
                win_rate = (stats['profitable_trades'] / stats['total_trades_managed']) * 100
            else:
                win_rate = 0
            
            message = (
                f"📊 <b>Integrated Bot Performance Report</b>\n"
                f"Total Trades: {stats['total_trades_managed']}\n"
                f"Profitable Trades: {stats['profitable_trades']}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Take Profit Hits: {stats['take_profit_hits']}\n"
                f"Stop Loss Trades: {stats['stopped_trades']}\n"
                f"Timeout Trades: {stats['timeout_trades']}\n"
                f"Active Trades: {len(self.trade_manager.managed_trades)}\n"
                f"Total PnL: {stats['total_pnl']:.2f}%\n"
                f"Time: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message, title="Performance Report")
            
        except Exception as e:
            logger.error(f"❌ Error sending performance report: {e}")
            return False

# ========== Flask Interface ==========

@app.route('/')
def health_check():
    """Bot health check"""
    try:
        bot = UnifiedTradeBot.get_instance()
        status = bot.get_status()
        
        status.update({
            'api_status': 'active',
            'supported_symbols': TRADING_SETTINGS['symbols'],
            'bot_version': 'unified-v1.0',
            'role': 'Integrated System - Execution and Management',
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now(damascus_tz).isoformat()
        }), 500

@app.route('/api/trade/signal', methods=['POST'])
@require_api_key
def receive_trade_signal():
    """Receive trading signals from external bot"""
    try:
        bot = UnifiedTradeBot.get_instance()
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data'})
        
        signal_data = data.get('signal', {})
        if not signal_data:
            return jsonify({'success': False, 'message': 'Signal data missing'})
        
        logger.info(f"📨 Receiving new signal from sender bot: {signal_data}")
        
        converted_signal = convert_signal_format(signal_data)
        if not converted_signal:
            return jsonify({'success': False, 'message': 'Invalid signal format'})
        
        success, message = bot.signal_receiver.process_signal(converted_signal)
        
        response_data = {
            'success': success,
            'message': message,
            'signal_received': signal_data,
            'signal_processed': converted_signal,
            'timestamp': datetime.now(damascus_tz).isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Error receiving trading signal: {e}")
        return jsonify({'success': False, 'message': f'Processing error: {str(e)}'})

@app.route('/api/heartbeat', methods=['POST'])
@require_api_key
def receive_heartbeat():
    """Receive heartbeats from sender bot"""
    try:
        data = request.get_json()
        
        if not data or not data.get('heartbeat'):
            return jsonify({'success': False, 'message': 'Invalid heartbeat data'})
        
        source = data.get('source', 'unknown')
        syria_time = data.get('syria_time')
        system_stats = data.get('system_stats', {})
        
        logger.info(f"💓 Receiving heartbeat from {source}")
        
        bot = UnifiedTradeBot.get_instance()
        
        response_data = {
            'success': True,
            'message': 'Heartbeat received successfully',
            'bot_status': 'active',
            'managed_trades': len(bot.trade_manager.managed_trades),
            'bot_version': 'unified-v1.0',
            'timestamp': datetime.now(damascus_tz).isoformat(),
            'received_heartbeat': {
                'source': source,
                'syria_time': syria_time,
                'scanner_stats': system_stats
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Error receiving heartbeat: {e}")
        return jsonify({'success': False, 'message': f'Heartbeat receive error: {str(e)}'})

@app.route('/health')
def health_check_endpoint():
    """Bot health check"""
    try:
        bot = UnifiedTradeBot.get_instance()
        status = bot.get_status()
        
        status.update({
            'api_status': 'active',
            'supported_symbols': TRADING_SETTINGS['symbols'],
            'bot_version': 'unified-v1.0',
            'role': 'Integrated System - Execution and Management',
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now(damascus_tz).isoformat()
        }), 500

@app.route('/active_trades')
def get_active_trades():
    """Get active trades"""
    try:
        bot = UnifiedTradeBot.get_instance()
        return jsonify(bot.trade_manager.managed_trades)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/performance')
def get_performance():
    """Get performance statistics"""
    try:
        bot = UnifiedTradeBot.get_instance()
        return jsonify(bot.trade_manager.performance_stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recent_signals')
def get_recent_signals():
    """Get recently received signals"""
    try:
        bot = UnifiedTradeBot.get_instance()
        limit = request.args.get('limit', 10, type=int)
        signals = bot.signal_receiver.get_recent_signals(limit)
        
        for signal in signals:
            if 'received_time' in signal:
                signal['received_time'] = signal['received_time'].isoformat()
        
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/close_trade/<symbol>', methods=['POST'])
def close_trade(symbol):
    """Manually close a trade"""
    try:
        bot = UnifiedTradeBot.get_instance()
        data = request.get_json() or {}
        reason = data.get('reason', 'Manual closure')
        
        if symbol in bot.trade_manager.managed_trades:
            success, message = bot.trade_manager.close_entire_trade(symbol, reason)
            return jsonify({'success': success, 'message': message})
        else:
            return jsonify({'success': False, 'message': 'No managed trade with this symbol'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

def run_flask_app():
    """Run Flask application"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

def main():
    """Main function"""
    try:
        # Initialize bot
        bot = UnifiedTradeBot.get_instance()
        
        # Start Flask in separate thread
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        
        logger.info("🚀 Starting Integrated Bot...")
        
        # Send startup message
        if bot.notifier:
            message = (
                "🚀 <b>Starting Integrated Bot v1.0</b>\n"
                f"📋 <b>Role:</b> Integrated System for Execution and Management\n"
                f"✅ <b>Features:</b>\n"
                f"• Receive external trading signals\n"
                f"• Execute trades based on confidence levels\n"
                f"• Automatic management of open trades\n"
                f"• Dynamic stop loss (two phases)\n"
                f"• Dynamic take profit (two levels)\n"
                f"• Smart extension system based on performance\n"
                f"• Advanced margin and risk monitoring\n"
                f"• Automatic cleanup and sync with Binance\n"
                f"• Performance reports and statistics\n"
                f"Supported Symbols: {', '.join(TRADING_SETTINGS['symbols'])}\n"
                f"Trade Size: ${TRADING_SETTINGS['base_trade_amount']} × {TRADING_SETTINGS['leverage']} leverage\n"
                f"Max Trades: {TRADING_SETTINGS['max_simultaneous_trades']} trades simultaneously\n"
                f"Port: {os.environ.get('PORT', 10000)}\n"
                f"Status: Ready to receive signals ✅\n"
                f"Time: {datetime.now(damascus_tz).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            bot.notifier.send_message(message, title="Bot Started", message_type='success')
        
        # Start main management loop
        bot.management_loop()
                
    except Exception as e:
        logger.error(f"❌ Failed to run bot: {e}")

if __name__ == "__main__":
    main()
