"""
Crypto Signal Analyzer Bot - Enhanced for Buy Signals
Version 4.0.0 - Specialized in buy signals only, improved sensitivity and accuracy
All NTFY messages in English (no emojis)
"""

import os
import json
import time
import math
import logging
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock

from flask import Flask, render_template, jsonify, request
import ccxt

# ======================
# إعدادات التسجيل
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crypto_signal.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ======================
# هياكل البيانات الأساسية
# ======================
class SignalType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"

class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    STRUCTURE = "structure"
    SUPPORT_RESISTANCE = "support_resistance"
    # إضافة مؤشر جديد للأنماط السعرية
    CANDLE_PATTERN = "candle_pattern"
    # إضافة مؤشر للقوة الشرائية
    BUYING_PRESSURE = "buying_pressure"

@dataclass
class CoinConfig:
    symbol: str
    name: str
    base_asset: str
    quote_asset: str
    enabled: bool = True

@dataclass
class IndicatorScore:
    name: str
    raw_score: float
    weighted_score: float
    percentage: float
    weight: float
    description: str
    color: str

@dataclass
class CoinSignal:
    symbol: str
    name: str
    current_price: float
    price_change_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    total_percentage: float
    signal_type: SignalType
    signal_strength: str
    signal_color: str
    indicator_scores: Dict[str, IndicatorScore]
    last_updated: datetime
    fear_greed_value: int
    is_valid: bool = True
    error_message: Optional[str] = None

@dataclass
class Notification:
    id: str
    timestamp: datetime
    coin_symbol: str
    coin_name: str
    message: str
    notification_type: str
    signal_strength: float
    price: float

# ======================
# إعدادات التطبيق
# ======================
class AppConfig:
    @staticmethod
    def get_top_coins(limit=15):
        """جلب أفضل العملات من حيث حجم التداول"""
        try:
            exchange = ccxt.binance()
            tickers = exchange.fetch_tickers()
            
            usdt_pairs = {k: v for k, v in tickers.items() 
                         if k.endswith('/USDT') and v.get('quoteVolume')}
            
            sorted_pairs = sorted(usdt_pairs.items(), 
                                key=lambda x: x[1]['quoteVolume'] or 0, 
                                reverse=True)
            
            coins = []
            EXCLUDED_COINS = ['LUNA', 'UST', 'FTT', 'TERRA']
            
            for symbol, ticker in sorted_pairs[:limit]:
                base = symbol.replace('/USDT', '')
                if base not in EXCLUDED_COINS:
                    coins.append(CoinConfig(symbol, base, base, 'USDT'))
            
            if coins:
                logger.info(f"✅ تم جلب {len(coins)} عملة من Binance")
                return coins
            else:
                logger.warning("⚠️ لم نجد عملات، نستخدم القائمة الافتراضية")
                return AppConfig._get_default_coins()
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب العملات: {e}")
            return AppConfig._get_default_coins()
    
    @staticmethod
    def _get_default_coins():
        """القائمة الافتراضية في حالة فشل الاتصال"""
        return [
            CoinConfig("BTC/USDT", "Bitcoin", "BTC", "USDT"),
            CoinConfig("ETH/USDT", "Ethereum", "ETH", "USDT"),
            CoinConfig("BNB/USDT", "Binance Coin", "BNB", "USDT"),
            CoinConfig("SOL/USDT", "Solana", "SOL", "USDT"),
            CoinConfig("XRP/USDT", "Ripple", "XRP", "USDT"),
            CoinConfig("ADA/USDT", "Cardano", "ADA", "USDT"),
            CoinConfig("DOGE/USDT", "Dogecoin", "DOGE", "USDT"),
            CoinConfig("AVAX/USDT", "Avalanche", "AVAX", "USDT"),
            CoinConfig("DOT/USDT", "Polkadot", "DOT", "USDT"),
            CoinConfig("MATIC/USDT", "Polygon", "MATIC", "USDT"),
            CoinConfig("LINK/USDT", "Chainlink", "LINK", "USDT"),
            CoinConfig("TRX/USDT", "TRON", "TRX", "USDT"),
            CoinConfig("ZEC/USDT", "Zcash", "ZEC", "USDT"),
            CoinConfig("LTC/USDT", "Litecoin", "LTC", "USDT"),
            CoinConfig("BCH/USDT", "Bitcoin Cash", "BCH", "USDT"),
        ]

    COINS = get_top_coins(15)

    # أوزان محسنة لصالح إشارات الشراء (زيادة وزن الزخم والحجم والأنماط السعرية)
    INDICATOR_WEIGHTS = {
        IndicatorType.TREND.value: 0.20,           # الاتجاه
        IndicatorType.MOMENTUM.value: 0.25,        # الزخم (أعلى وزن)
        IndicatorType.VOLUME.value: 0.20,           # الحجم
        IndicatorType.VOLATILITY.value: 0.05,       # التقلب (منخفض)
        IndicatorType.SENTIMENT.value: 0.05,        # المشاعر (منخفض)
        IndicatorType.STRUCTURE.value: 0.10,        # الهيكل السعري
        IndicatorType.SUPPORT_RESISTANCE.value: 0.05, # دعم/مقاومة
        IndicatorType.CANDLE_PATTERN.value: 0.05,   # نمط الشموع (جديد)
        IndicatorType.BUYING_PRESSURE.value: 0.05   # ضغط شرائي (جديد)
    }

    # تعديل العتبات لجعل إشارات الشراء أكثر حساسية (خفض عتبات الشراء)
    SIGNAL_THRESHOLDS = {
        SignalType.STRONG_BUY: 75,    # كانت 80
        SignalType.BUY: 65,           # كانت 70
        SignalType.NEUTRAL: 45,       # كانت 50
        SignalType.SELL: 35,           # كانت 40
        SignalType.STRONG_SELL: 25     # كانت 30
    }

    UPDATE_INTERVAL = 120  # 2 دقائق
    MAX_CANDLES = 200

    # ألوان المؤشرات
    INDICATOR_COLORS = {
        IndicatorType.TREND.value: '#2E86AB',
        IndicatorType.MOMENTUM.value: '#A23B72',
        IndicatorType.VOLUME.value: '#3BB273',
        IndicatorType.VOLATILITY.value: '#F18F01',
        IndicatorType.SENTIMENT.value: '#6C757D',
        IndicatorType.STRUCTURE.value: '#8F2D56',
        IndicatorType.SUPPORT_RESISTANCE.value: '#6A4C93',
        IndicatorType.CANDLE_PATTERN.value: '#E67E22',   # برتقالي
        IndicatorType.BUYING_PRESSURE.value: '#27AE60'   # أخضر
    }

    INDICATOR_DISPLAY_NAMES = {
        IndicatorType.TREND.value: 'Trend Strength',
        IndicatorType.MOMENTUM.value: 'Momentum (RSI/MACD)',
        IndicatorType.VOLUME.value: 'Volume Analysis',
        IndicatorType.VOLATILITY.value: 'Volatility',
        IndicatorType.SENTIMENT.value: 'Market Sentiment',
        IndicatorType.STRUCTURE.value: 'Price Structure',
        IndicatorType.SUPPORT_RESISTANCE.value: 'Support/Resistance',
        IndicatorType.CANDLE_PATTERN.value: 'Candle Patterns',
        IndicatorType.BUYING_PRESSURE.value: 'Buying Pressure'
    }

    INDICATOR_DESCRIPTIONS = {
        IndicatorType.TREND.value: 'Trend direction and strength using EMAs',
        IndicatorType.MOMENTUM.value: 'RSI, MACD, and ROC combined',
        IndicatorType.VOLUME.value: 'Volume surge and accumulation detection',
        IndicatorType.VOLATILITY.value: 'Bollinger Bands position',
        IndicatorType.SENTIMENT.value: 'Fear & Greed Index',
        IndicatorType.STRUCTURE.value: 'Higher highs / lower lows',
        IndicatorType.SUPPORT_RESISTANCE.value: 'Nearby support/resistance levels',
        IndicatorType.CANDLE_PATTERN.value: 'Bullish reversal patterns (hammer, engulfing, morning star)',
        IndicatorType.BUYING_PRESSURE.value: 'Price strength relative to volume'
    }

# ======================
# إعدادات APIs الخارجية
# ======================
class ExternalAPIConfig:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'crypto_signals_alerts')
    NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
    FGI_API_URL = "https://api.alternative.me/fng/"
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 2

# ======================
# Binance Client
# ======================
class BinanceClient:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': ExternalAPIConfig.BINANCE_API_KEY,
            'secret': ExternalAPIConfig.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 200) -> Optional[List]:
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Binance OHLCV error {symbol}: {e}")
            return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Binance ticker error {symbol}: {e}")
            return None

    def fetch_24h_stats(self, symbol: str) -> Dict:
        ticker = self.fetch_ticker(symbol)
        if ticker:
            return {
                'change': ticker.get('percentage', 0.0),
                'high': ticker.get('high', 0.0),
                'low': ticker.get('low', 0.0),
                'volume': ticker.get('quoteVolume', 0.0)
            }
        return {'change': 0.0, 'high': 0.0, 'low': 0.0, 'volume': 0.0}

    def get_current_price(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        return ticker['last'] if ticker else 0.0

# ======================
# Fear & Greed Index Fetcher
# ======================
class FearGreedFetcher:
    def __init__(self):
        self.last_value = 50
        self.last_update = None
        self.cache_ttl = 300

    def get(self) -> Tuple[float, int]:
        now = datetime.now()
        if self.last_update and (now - self.last_update).total_seconds() < self.cache_ttl:
            return self._to_score(self.last_value), self.last_value

        try:
            resp = requests.get(ExternalAPIConfig.FGI_API_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and data['data']:
                    value = int(data['data'][0]['value'])
                    self.last_value = value
                    self.last_update = now
                    return self._to_score(value), value
        except Exception as e:
            logger.error(f"FGI fetch error: {e}")

        return self._to_score(self.last_value), self.last_value

    def _to_score(self, value: int) -> float:
        # تحويل مؤشر الخوف والطمع إلى درجة شراء
        if value >= 80:   # طمع شديد -> خطر شراء
            return 0.20
        if value >= 60:   # طمع
            return 0.40
        if value >= 40:   # محايد
            return 0.60
        if value >= 20:   # خوف
            return 0.80
        return 0.95       # خوف شديد (فرصة شراء)

# ======================
# Indicator Calculators (محسّنة لصالح إشارات الشراء)
# ======================
class IndicatorCalculator:
    @staticmethod
    def sma(prices: List[float], period: int) -> List[Optional[float]]:
        result = []
        for i in range(len(prices)):
            if i < period - 1:
                result.append(None)
            else:
                result.append(sum(prices[i - period + 1:i + 1]) / period)
        return result

    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        if not prices:
            return []
        k = 2 / (period + 1)
        ema_values = [prices[0]]
        for i in range(1, len(prices)):
            ema_values.append(prices[i] * k + ema_values[-1] * (1 - k))
        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
        if len(prices) < period + 1:
            return [None] * len(prices)
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        rsi_values = [None] * period

        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
            if i < len(prices) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        return rsi_values

    @staticmethod
    def macd(prices: List[float], fast=12, slow=26, signal=9) -> Dict:
        """حساب MACD وإشارة التقاطع"""
        if len(prices) < slow + signal:
            return {'histogram': 0, 'signal': 0, 'macd': 0}
        ema_fast = IndicatorCalculator.ema(prices, fast)
        ema_slow = IndicatorCalculator.ema(prices, slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = IndicatorCalculator.ema(macd_line, signal)
        histogram = macd_line[-1] - signal_line[-1]
        return {
            'histogram': histogram,
            'signal': signal_line[-1],
            'macd': macd_line[-1]
        }

    # ================== المؤشرات المحسّنة ==================

    @staticmethod
    def trend_strength(close_prices: List[float]) -> float:
        """تحسين اتجاه الشراء باستخدام EMAs متعددة وتقاطعاتها"""
        if len(close_prices) < 50:
            return 0.5
        ema_9 = IndicatorCalculator.ema(close_prices, 9)[-1]
        ema_21 = IndicatorCalculator.ema(close_prices, 21)[-1]
        ema_50 = IndicatorCalculator.ema(close_prices, 50)[-1]
        ema_200 = IndicatorCalculator.ema(close_prices, 200) if len(close_prices) >= 200 else None

        current = close_prices[-1]
        score = 0.0

        # السعر فوق المتوسطات يعطي نقاط إيجابية
        if current > ema_9: score += 0.15
        if current > ema_21: score += 0.15
        if current > ema_50: score += 0.15
        if ema_200 and current > ema_200: score += 0.15

        # ترتيب المتوسطات (صعودي)
        if ema_9 > ema_21 > ema_50:
            score += 0.3
        elif ema_9 < ema_21 < ema_50:
            score -= 0.2

        # المسافة بين السعر والمتوسطات
        dist_to_ema9 = (current - ema_9) / current
        if dist_to_ema9 > 0.02:  # زخم قوي
            score += 0.1

        return max(0.0, min(1.0, (score + 1) / 2.5))  # تطبيع

    @staticmethod
    def momentum(close_prices: List[float]) -> float:
        """دمج RSI, MACD, ROC لقياس الزخم الشرائي"""
        if len(close_prices) < 30:
            return 0.5

        # RSI
        rsi_vals = IndicatorCalculator.rsi(close_prices, 14)
        last_rsi = rsi_vals[-1] if rsi_vals[-1] is not None else 50
        if last_rsi < 30:
            rsi_score = 0.9   # ذروة بيع -> شراء
        elif last_rsi > 70:
            rsi_score = 0.2   # ذروة شراء -> بيع (لكننا نركز على الشراء)
        else:
            rsi_score = 1.0 - (last_rsi / 100)  # كلما قل RSI زادت فرصة الشراء

        # MACD
        macd_data = IndicatorCalculator.macd(close_prices)
        hist = macd_data['histogram']
        if hist > 0 and hist > macd_data.get('prev_hist', 0):
            macd_score = 0.8   # تقاطع إيجابي
        elif hist > 0:
            macd_score = 0.6
        elif hist < 0 and hist < macd_data.get('prev_hist', 0):
            macd_score = 0.3   # زخم سلبي
        else:
            macd_score = 0.5

        # ROC (معدل التغير)
        roc_14 = (close_prices[-1] - close_prices[-14]) / close_prices[-14] * 100
        roc_score = max(0.0, min(1.0, (roc_14 + 5) / 15))  # تطبيع بين -5% و +10%

        # الجمع مع أوزان
        return rsi_score * 0.5 + macd_score * 0.3 + roc_score * 0.2

    @staticmethod
    def volume_analysis(volumes: List[float], close_prices: List[float]) -> float:
        """تحليل الحجم مع التركيز على التراكم واختراق الحجم"""
        if len(volumes) < 20:
            return 0.5
        current_vol = volumes[-1]
        avg_vol = sum(volumes[-20:]) / 20
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # حجم مرتفع -> احتمال قوي
        if ratio > 2.0:
            score = 0.8
        elif ratio > 1.5:
            score = 0.7
        elif ratio > 1.2:
            score = 0.65
        elif ratio > 1.0:
            score = 0.6
        elif ratio > 0.8:
            score = 0.5
        else:
            score = 0.4

        # تحقق من اتجاه السعر مع الحجم
        price_change = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
        if price_change > 0.01 and ratio > 1.2:
            score += 0.2   # اختراق سعري بحجم كبير
        elif price_change > 0 and ratio > 1:
            score += 0.1
        elif price_change < -0.01 and ratio > 1.2:
            score -= 0.2   # بيع بحجم كبير (لا يشجع الشراء)

        # تراكم تدريجي (حجم متزايد)
        if volumes[-1] > volumes[-2] > volumes[-3] > volumes[-4]:
            score += 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def volatility(high: List[float], low: List[float], close: List[float]) -> float:
        """تقلب السعر وموقعه من Bollinger Bands (الشراء عند القاع)"""
        if len(close) < 20:
            return 0.5
        sma_20 = IndicatorCalculator.sma(close, 20)[-1]
        std_dev = 0
        for i in range(-20, 0):
            std_dev += (close[i] - sma_20) ** 2
        std_dev = math.sqrt(std_dev / 20)
        upper = sma_20 + 2 * std_dev
        lower = sma_20 - 2 * std_dev

        if upper == lower:
            return 0.5

        position = (close[-1] - lower) / (upper - lower)
        # كلما اقترب من القاع (position صغير) كانت فرصة الشراء أفضل
        if position < 0.2:
            return 0.8   # قاع البولينجر -> ارتداد محتمل
        if position > 0.8:
            return 0.3   # قمة -> احتمال تصحيح
        if position < 0.4:
            return 0.6
        return 0.5

    @staticmethod
    def price_structure(high: List[float], low: List[float], close: List[float]) -> float:
        """هيكل سعري: قيعان صاعدة / قمم صاعدة"""
        if len(high) < 30:
            return 0.5
        # قمم وقيعان آخر 30 شمعة
        highs_30 = high[-30:]
        lows_30 = low[-30:]
        recent_high = max(highs_30)
        recent_low = min(lows_30)

        # تحديد الاتجاه
        higher_highs = highs_30[-1] > highs_30[-5] > highs_30[-10]
        higher_lows = lows_30[-1] > lows_30[-5] > lows_30[-10]

        if higher_highs and higher_lows:
            return 0.8   # هيكل صعودي قوي
        if higher_lows:
            return 0.7   # قيعان صاعدة (إشارة شراء)
        if not higher_highs and not higher_lows:
            return 0.3   # هيكل هابط
        return 0.5

    @staticmethod
    def support_resistance(high: List[float], low: List[float], close: List[float]) -> float:
        """تحديد القرب من مستويات دعم رئيسية (للبحث عن ارتداد)"""
        if len(high) < 40:
            return 0.5
        highs = high[-40:]
        lows = low[-40:]
        resistance_candidates = []
        support_candidates = []

        # قمم محلية
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_candidates.append(highs[i])
        # قيعان محلية
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_candidates.append(lows[i])

        if not support_candidates and not resistance_candidates:
            return 0.5

        current = close[-1]
        # أقرب دعم أقل من السعر
        closest_support = max([s for s in support_candidates if s < current], default=None)
        # أقرب مقاومة أعلى من السعر
        closest_resistance = min([r for r in resistance_candidates if r > current], default=None)

        if closest_support:
            dist_to_support = (current - closest_support) / current
            if dist_to_support < 0.01:
                return 0.9   # قريب جداً من الدعم
            if dist_to_support < 0.03:
                return 0.7
        if closest_resistance:
            dist_to_resistance = (closest_resistance - current) / current
            if dist_to_resistance < 0.01:
                return 0.2   # قريب من مقاومة
            if dist_to_resistance < 0.03:
                return 0.3
        return 0.5

    @staticmethod
    def candle_pattern(open_prices: List[float], high: List[float], low: List[float], close: List[float]) -> float:
        """الكشف عن أنماط الشموع اليابانية التي تشير إلى انعكاس صعودي"""
        if len(close) < 5:
            return 0.5

        # الشموع الثلاث الأخيرة
        o1, o2, o3 = open_prices[-1], open_prices[-2], open_prices[-3]
        h1, h2, h3 = high[-1], high[-2], high[-3]
        l1, l2, l3 = low[-1], low[-2], low[-3]
        c1, c2, c3 = close[-1], close[-2], close[-3]

        score = 0.5

        # 1. نمط المطرقة (Hammer)
        body1 = abs(c1 - o1)
        lower_shadow1 = min(c1, o1) - l1
        upper_shadow1 = h1 - max(c1, o1)
        if lower_shadow1 > 2 * body1 and upper_shadow1 < 0.3 * body1 and c1 > o1:
            score += 0.3  # مطرقة صاعدة

        # 2. نمط الابتلاع الصاعد (Bullish Engulfing)
        if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:
            score += 0.4

        # 3. نمط نجمة الصباح (Morning Star) - ثلاث شموع
        if c2 < o2 and abs(c2 - o2) > 0.02 * c2:  # شمعة هابطة كبيرة
            if abs(c1 - o1) < 0.01 * c1:  # شمعة دوجي صغيرة
                if c3 > o3 and c3 > (c2 + o2)/2:  # شمعة صاعدة تخترق منتصف الأولى
                    score += 0.5

        # 4. نمط الارتطام (Piercing Line)
        if c2 < o2 and c1 > o1 and o1 < c2 and c1 > (c2 + o2)/2 and c1 < o2:
            score += 0.3

        return min(1.0, score)

    @staticmethod
    def buying_pressure(close: List[float], volume: List[float]) -> float:
        """قياس الضغط الشرائي من خلال حجم التداول بالنسبة لحركة السعر"""
        if len(close) < 10:
            return 0.5

        # مؤشر تدفق الأموال (Money Flow Index) مبسط
        typical_price = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]  # لكننا لا نملك high/low هنا
        # نستخدم close كبديل تقريبي
        money_flow = [c * v for c, v in zip(close, volume)]

        # اتجاه الضغط الشرائي
        positive_flow = 0
        negative_flow = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                positive_flow += money_flow[i]
            else:
                negative_flow += money_flow[i]

        if negative_flow == 0:
            return 1.0
        ratio = positive_flow / negative_flow
        # تطبيع: كلما زادت النسبة زاد الضغط الشرائي
        score = min(1.0, ratio / 3)  # نسبة 3 تعطي 1
        return score

# ======================
# Signal Processor
# ======================
class SignalProcessor:
    @staticmethod
    def calculate_weighted_score(indicator_scores: Dict[str, float]) -> Dict:
        total_weighted = 0.0
        weighted_scores = {}

        for indicator, score in indicator_scores.items():
            weight = AppConfig.INDICATOR_WEIGHTS.get(indicator, 0.1)
            weighted = score * weight
            total_weighted += weighted

            weighted_scores[indicator] = IndicatorScore(
                name=indicator,
                raw_score=score,
                weighted_score=weighted,
                percentage=score * 100,
                weight=weight,
                description=AppConfig.INDICATOR_DESCRIPTIONS.get(indicator, ''),
                color=AppConfig.INDICATOR_COLORS.get(indicator, '#2E86AB')
            )

        total_percentage = total_weighted * 100
        signal_type = SignalProcessor.get_signal_type(total_percentage)
        signal_strength = SignalProcessor.get_signal_strength(total_percentage)
        signal_color = SignalProcessor.get_signal_color(signal_type)

        return {
            'total_percentage': total_percentage,
            'weighted_scores': weighted_scores,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'signal_color': signal_color
        }

    @staticmethod
    def get_signal_type(percentage: float) -> SignalType:
        if percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.STRONG_BUY]:
            return SignalType.STRONG_BUY
        if percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.BUY]:
            return SignalType.BUY
        if percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.NEUTRAL]:
            return SignalType.NEUTRAL
        if percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.SELL]:
            return SignalType.SELL
        return SignalType.STRONG_SELL

    @staticmethod
    def get_signal_strength(percentage: float) -> str:
        if percentage >= 85:
            return "Very Strong"
        if percentage >= 70:
            return "Strong"
        if percentage >= 55:
            return "Moderate"
        if percentage >= 40:
            return "Weak"
        return "Very Weak"

    @staticmethod
    def get_signal_color(signal_type: SignalType) -> str:
        mapping = {
            SignalType.STRONG_BUY: "success",
            SignalType.BUY: "primary",
            SignalType.NEUTRAL: "secondary",
            SignalType.SELL: "warning",
            SignalType.STRONG_SELL: "danger"
        }
        return mapping.get(signal_type, "secondary")

# ======================
# Notification Manager (English only, no emojis, only buy signals)
# ======================
class NotificationManager:
    def __init__(self):
        self.history: List[Notification] = []
        self.max_history = 50
        self.last_notification_time = {}
        self.min_interval = 300  # 5 minutes

    def add(self, notification: Notification):
        self.history.append(notification)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, limit: int = 10) -> List[Notification]:
        return self.history[-limit:] if self.history else []

    def should_send(self, coin_symbol: str, percentage: float, signal_type: SignalType) -> bool:
        """إرسال فقط لإشارات الشراء"""
        if signal_type not in [SignalType.BUY, SignalType.STRONG_BUY]:
            return False

        now = datetime.now()
        if coin_symbol in self.last_notification_time:
            delta = now - self.last_notification_time[coin_symbol]
            if delta.total_seconds() < self.min_interval:
                return False

        return True

    def send_ntfy(self, message: str, title: str = "Crypto Signal", priority: str = "3", tags: str = "chart") -> bool:
        try:
            headers = {
                "Title": title,
                "Priority": priority,
                "Tags": tags,
                "Content-Type": "text/plain; charset=utf-8"
            }
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            resp = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=safe_message.encode('utf-8'),
                headers=headers,
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"NTFY error: {e}")
            return False

    def create_notification(self, coin_signal: CoinSignal) -> Optional[Notification]:
        if not self.should_send(coin_signal.symbol, coin_signal.total_percentage, coin_signal.signal_type):
            return None

        coin = coin_signal
        signal_type = coin.signal_type

        # رسالة إنجليزية بدون إيموجي
        title = f"{signal_type.value} Signal: {coin.name}"
        message = (
            f"{title}\n"
            f"Signal Strength: {coin.total_percentage:.1f}%\n"
            f"Price: ${coin.current_price:,.2f}\n"
            f"24h Change: {coin.price_change_24h:+.2f}%\n"
            f"Time: {coin.last_updated.strftime('%H:%M')}"
        )

        tags_map = {
            SignalType.STRONG_BUY: "heavy_plus_sign",
            SignalType.BUY: "chart_increasing",
            SignalType.STRONG_SELL: "heavy_minus_sign",
            SignalType.SELL: "chart_decreasing"
        }
        tags = tags_map.get(signal_type, "loudspeaker")

        priority_map = {
            SignalType.STRONG_BUY: "4",
            SignalType.BUY: "3",
            SignalType.STRONG_SELL: "4",
            SignalType.SELL: "3"
        }
        priority = priority_map.get(signal_type, "3")

        if self.send_ntfy(message, title, priority, tags):
            notification = Notification(
                id=f"{coin.symbol}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                coin_symbol=coin.symbol,
                coin_name=coin.name,
                message=message,
                notification_type=signal_type.name.lower(),
                signal_strength=coin.total_percentage,
                price=coin.current_price
            )
            self.add(notification)
            self.last_notification_time[coin.symbol] = datetime.now()
            return notification
        return None

# ======================
# Signal Manager
# ======================
class SignalManager:
    def __init__(self):
        self.signals: Dict[str, CoinSignal] = {}
        self.last_coins_update = None
        self.history: List[Dict] = []
        self.last_update: Optional[datetime] = None
        self.fgi_fetcher = FearGreedFetcher()
        self.notification_manager = NotificationManager()
        self.binance = BinanceClient()
        self.lock = Lock()
        self.fear_greed_index = 50
        self.fear_greed_score = 0.5

    def update_coins_list(self):
        """تحديث قائمة العملات كل ساعة"""
        now = datetime.now()
        if not self.last_coins_update or (now - self.last_coins_update).seconds > 3600:
            new_coins = AppConfig.get_top_coins(15)
            if new_coins:
                AppConfig.COINS = new_coins
                self.last_coins_update = now
                logger.info(f"🔄 تم تحديث قائمة العملات: {len(new_coins)} عملة")

    def update_all(self) -> bool:
        with self.lock:
            self.update_coins_list()
            logger.info(f"🔄 Updating {len(AppConfig.COINS)} coins...")
            success_count = 0
            self.fear_greed_score, self.fear_greed_index = self.fgi_fetcher.get()

            for coin in AppConfig.COINS:
                if not coin.enabled:
                    continue
                try:
                    signal = self._process_coin(coin)
                    if signal and signal.is_valid:
                        self.signals[coin.symbol] = signal
                        success_count += 1
                        self.notification_manager.create_notification(signal)
                except Exception as e:
                    logger.error(f"Error on {coin.symbol}: {e}")

            self.last_update = datetime.now()
            self._save_history()
            logger.info(f"✅ Updated {success_count}/{len(AppConfig.COINS)}")
            return success_count > 0

    def _process_coin(self, coin: CoinConfig) -> Optional[CoinSignal]:
        ohlcv = self.binance.fetch_ohlcv(coin.symbol, '15m', AppConfig.MAX_CANDLES)
        if not ohlcv or len(ohlcv) < 50:
            return None

        opens = [c[1] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        ticker = self.binance.fetch_ticker(coin.symbol)
        if not ticker:
            return None

        current_price = ticker['last']
        change_24h = ticker.get('percentage', 0.0)
        high_24h = ticker.get('high', 0.0)
        low_24h = ticker.get('low', 0.0)
        volume_24h = ticker.get('quoteVolume', 0.0)

        # حساب جميع المؤشرات
        scores = {
            IndicatorType.TREND.value: IndicatorCalculator.trend_strength(closes),
            IndicatorType.MOMENTUM.value: IndicatorCalculator.momentum(closes),
            IndicatorType.VOLUME.value: IndicatorCalculator.volume_analysis(volumes, closes),
            IndicatorType.VOLATILITY.value: IndicatorCalculator.volatility(highs, lows, closes),
            IndicatorType.SENTIMENT.value: self.fear_greed_score,
            IndicatorType.STRUCTURE.value: IndicatorCalculator.price_structure(highs, lows, closes),
            IndicatorType.SUPPORT_RESISTANCE.value: IndicatorCalculator.support_resistance(highs, lows, closes),
            IndicatorType.CANDLE_PATTERN.value: IndicatorCalculator.candle_pattern(opens, highs, lows, closes),
            IndicatorType.BUYING_PRESSURE.value: IndicatorCalculator.buying_pressure(closes, volumes)
        }

        result = SignalProcessor.calculate_weighted_score(scores)

        return CoinSignal(
            symbol=coin.symbol,
            name=coin.name,
            current_price=current_price,
            price_change_24h=change_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            volume_24h=volume_24h,
            total_percentage=result['total_percentage'],
            signal_type=result['signal_type'],
            signal_strength=result['signal_strength'],
            signal_color=result['signal_color'],
            indicator_scores=result['weighted_scores'],
            last_updated=datetime.now(),
            fear_greed_value=self.fear_greed_index,
            is_valid=True
        )

    def _save_history(self):
        entry = {
            'timestamp': datetime.now(),
            'signals': {s: self.signals[s].total_percentage for s in self.signals},
            'fgi': self.fear_greed_index
        }
        self.history.append(entry)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def get_coins_data(self) -> List[Dict]:
        data = []
        for coin in AppConfig.COINS:
            signal = self.signals.get(coin.symbol)
            if signal and signal.is_valid:
                data.append(self._format_coin(signal))
            else:
                data.append(self._default_coin(coin))
        data.sort(key=lambda x: x['total_percentage'], reverse=True)
        return data

    def _format_coin(self, s: CoinSignal) -> Dict:
        indicators = []
        for k, v in s.indicator_scores.items():
            indicators.append({
                'name': k,
                'display_name': AppConfig.INDICATOR_DISPLAY_NAMES.get(k, k),
                'description': AppConfig.INDICATOR_DESCRIPTIONS.get(k, ''),
                'raw_score': v.raw_score * 100,
                'percentage': v.percentage,
                'color': v.color,
                'weight': v.weight * 100
            })
        return {
            'symbol': s.symbol,
            'name': s.name,
            'current_price': s.current_price,
            'formatted_price': self._format_number(s.current_price),
            'price_change_24h': s.price_change_24h,
            'formatted_24h_change': self._format_percentage(s.price_change_24h),
            'volume_24h': s.volume_24h,
            'formatted_volume_24h': self._format_number(s.volume_24h),
            'total_percentage': s.total_percentage,
            'signal_type': s.signal_type.value,
            'signal_strength': s.signal_strength,
            'signal_color': s.signal_color,
            'indicators': indicators,
            'last_updated_str': self._format_time_delta(s.last_updated),
            'fear_greed_value': s.fear_greed_value,
            'is_valid': True
        }

    def _default_coin(self, coin: CoinConfig) -> Dict:
        return {
            'symbol': coin.symbol,
            'name': coin.name,
            'current_price': 0,
            'formatted_price': '0',
            'price_change_24h': 0,
            'formatted_24h_change': '0.00%',
            'volume_24h': 0,
            'formatted_volume_24h': '0',
            'total_percentage': 50,
            'signal_type': SignalType.NEUTRAL.value,
            'signal_strength': 'Unavailable',
            'signal_color': 'secondary',
            'indicators': [],
            'last_updated_str': 'Unknown',
            'fear_greed_value': self.fear_greed_index,
            'is_valid': False
        }

    @staticmethod
    def _format_number(v: float) -> str:
        try:
            if v >= 1_000_000:
                return f"{v/1_000_000:.2f}M"
            if v >= 1_000:
                return f"{v/1_000:.2f}K"
            return f"{v:.2f}"
        except:
            return "0"

    @staticmethod
    def _format_percentage(v: float) -> str:
        try:
            return f"{v:+.2f}%" if v else "0.00%"
        except:
            return "0.00%"

    @staticmethod
    def _format_time_delta(dt: datetime) -> str:
        if not dt:
            return "Unknown"
        delta = datetime.now() - dt
        if delta.days > 0:
            return f"{delta.days} day(s) ago"
        if delta.seconds >= 3600:
            return f"{delta.seconds//3600} hour(s) ago"
        if delta.seconds >= 60:
            return f"{delta.seconds//60} minute(s) ago"
        return "Just now"

    def get_stats(self) -> Dict:
        coins = self.get_coins_data()
        valid = [c for c in coins if c['is_valid']]
        percentages = [c['total_percentage'] for c in valid]

        strong_buy = sum(1 for c in valid if c['total_percentage'] >= 70)  # تعديل العتبة حسب الإعدادات الجديدة
        buy = sum(1 for c in valid if 60 <= c['total_percentage'] < 70)
        neutral = sum(1 for c in valid if 35 < c['total_percentage'] < 60)
        sell = sum(1 for c in valid if 20 < c['total_percentage'] <= 35)
        strong_sell = sum(1 for c in valid if c['total_percentage'] <= 20)

        avg = sum(percentages) / len(percentages) if percentages else 50

        return {
            'total_coins': len(AppConfig.COINS),
            'updated_coins': len(valid),
            'avg_signal': avg,
            'strong_buy_signals': strong_buy,
            'buy_signals': buy,
            'neutral_signals': neutral,
            'sell_signals': sell,
            'strong_sell_signals': strong_sell,
            'last_update_str': self._format_time_delta(self.last_update) if self.last_update else 'Unknown',
            'total_notifications': len(self.notification_manager.history),
            'fear_greed_index': self.fear_greed_index,
            'system_status': 'healthy' if len(valid) >= len(AppConfig.COINS) * 0.7 else 'warning'
        }

# ======================
# Background Updater (Thread)
# ======================
def background_updater():
    """Automatic update every 2 minutes"""
    while True:
        try:
            signal_manager.update_all()
            time.sleep(AppConfig.UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Update error: {e}")
            time.sleep(60)

# ======================
# Flask App Initialization
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-signal-secret-2026')
signal_manager = SignalManager()
start_time = time.time()

# Start background updater thread
updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# Initial update
signal_manager.update_all()

# ======================
# Context Processor for Templates
# ======================
@app.context_processor
def utility_processor():
    def signal_color_to_css(color_name):
        mapping = {
            'success': 'var(--success)',
            'primary': 'var(--primary)',
            'secondary': 'var(--gray)',
            'warning': 'var(--warning)',
            'danger': 'var(--danger)'
        }
        return mapping.get(color_name, 'var(--secondary)')
    return dict(
        signal_color_to_css=signal_color_to_css,
        get_indicator_color=lambda k: AppConfig.INDICATOR_COLORS.get(k, '#2E86AB'),
        get_indicator_display_name=lambda k: AppConfig.INDICATOR_DISPLAY_NAMES.get(k, k)
    )

# ======================
# Routes
# ======================
@app.route('/')
def index():
    coins = signal_manager.get_coins_data()
    stats = signal_manager.get_stats()
    notifications = signal_manager.notification_manager.get_recent(10)
    return render_template(
        'index.html',
        coins=coins,
        stats=stats,
        notifications=notifications,
        indicator_weights=AppConfig.INDICATOR_WEIGHTS
    )

@app.route('/api/signals')
def api_signals():
    return jsonify({
        'status': 'success',
        'data': signal_manager.get_coins_data(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/update', methods=['POST'])
def manual_update():
    success = signal_manager.update_all()
    return jsonify({
        'status': 'success' if success else 'warning',
        'message': 'Update completed',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health():
    now = datetime.now()
    last = signal_manager.last_update
    status = 'healthy'
    if last and (now - last).total_seconds() > 600:
        status = 'warning'
    return jsonify({
        'status': status,
        'last_update': last.isoformat() if last else None,
        'coins': len(signal_manager.signals),
        'uptime': time.time() - start_time,
        'fear_greed': signal_manager.fear_greed_index,
        'notifications': len(signal_manager.notification_manager.history)
    })

@app.route('/api/notifications')
def get_notifications():
    limit = request.args.get('limit', 10, type=int)
    nots = signal_manager.notification_manager.get_recent(limit)
    return jsonify({
        'notifications': [asdict(n) for n in nots],
        'total': len(signal_manager.notification_manager.history)
    })

@app.route('/api/test_ntfy')
def test_ntfy():
    msg = "Test notification - System is working properly"
    success = signal_manager.notification_manager.send_ntfy(msg, "Test", "3", "test_tube")
    return jsonify({'success': success})

@app.route('/api/indicator_weights')
def indicator_weights():
    return jsonify({
        'weights': AppConfig.INDICATOR_WEIGHTS,
        'display_names': AppConfig.INDICATOR_DISPLAY_NAMES,
        'colors': AppConfig.INDICATOR_COLORS
    })

# ======================
# Startup notification (English, no emoji)
# ======================
def send_startup_notification():
    try:
        msg = (
            f"Crypto Signal Analyzer Started\n"
            f"Version: 4.0.0 (Enhanced for Buy Signals)\n"
            f"Tracking {len(AppConfig.COINS)} coins\n"
            f"Update interval: {AppConfig.UPDATE_INTERVAL//60} minutes"
        )
        signal_manager.notification_manager.send_ntfy(msg, "System Started", "3", "rocket")
    except Exception as e:
        logger.error(f"Startup notification error: {e}")

# Send startup notification after a short delay
def delayed_startup():
    time.sleep(5)
    send_startup_notification()

threading.Thread(target=delayed_startup, daemon=True).start()

# ======================
# Main
# ======================
if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 Crypto Signal Analyzer v4.0.0 (Enhanced for Buy Signals)")
    logger.info(f"📊 Coins: {len(AppConfig.COINS)}")
    logger.info(f"🔄 Update every {AppConfig.UPDATE_INTERVAL//60} minutes")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
