"""
Crypto Signal Analyzer Bot - النسخة الخفيفة والمحسنة للذاكرة
إصدار 3.5.2 - بدون Pandas/Numpy - تعمل على Render بذاكرة 512MB
"""

import os
import json
import time
import math
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock

from flask import Flask, render_template, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
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
    STRONG_BUY = "شراء قوي"
    BUY = "شراء"
    NEUTRAL = "محايد"
    SELL = "بيع"
    STRONG_SELL = "بيع قوي"

class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    STRUCTURE = "structure"
    SUPPORT_RESISTANCE = "support_resistance"

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
    raw_score: float  # 0-1
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
# إعدادات التطبيق - قائمة مختصرة وخفيفة
# ======================
class AppConfig:
    # العملات المدعومة (أهم 15 عملة لتخفيف الحمل)
    COINS = [
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
        CoinConfig("ATOM/USDT", "Cosmos", "ATOM", "USDT"),
        CoinConfig("LTC/USDT", "Litecoin", "LTC", "USDT"),
        CoinConfig("BCH/USDT", "Bitcoin Cash", "BCH", "USDT"),
    ]

    # أوزان المؤشرات المبسطة
    INDICATOR_WEIGHTS = {
        IndicatorType.TREND.value: 0.25,
        IndicatorType.MOMENTUM.value: 0.20,
        IndicatorType.VOLUME.value: 0.15,
        IndicatorType.VOLATILITY.value: 0.10,
        IndicatorType.SENTIMENT.value: 0.10,
        IndicatorType.STRUCTURE.value: 0.10,
        IndicatorType.SUPPORT_RESISTANCE.value: 0.10
    }

    # عتبات الإشارات
    SIGNAL_THRESHOLDS = {
        SignalType.STRONG_BUY: 75,
        SignalType.BUY: 60,
        SignalType.NEUTRAL: 45,
        SignalType.SELL: 35,
        SignalType.STRONG_SELL: 25
    }

    # إعدادات التحديث
    UPDATE_INTERVAL = 120  # 2 دقيقة
    MAX_CANDLES = 200      # عدد الشموع المحجوبة - أقل بكثير من 500

    # ألوان المؤشرات
    INDICATOR_COLORS = {
        IndicatorType.TREND.value: '#2E86AB',
        IndicatorType.MOMENTUM.value: '#A23B72',
        IndicatorType.VOLUME.value: '#3BB273',
        IndicatorType.VOLATILITY.value: '#F18F01',
        IndicatorType.SENTIMENT.value: '#6C757D',
        IndicatorType.STRUCTURE.value: '#8F2D56',
        IndicatorType.SUPPORT_RESISTANCE.value: '#6A4C93'
    }

    INDICATOR_DISPLAY_NAMES = {
        IndicatorType.TREND.value: 'قوة الاتجاه',
        IndicatorType.MOMENTUM.value: 'الزخم',
        IndicatorType.VOLUME.value: 'الحجم',
        IndicatorType.VOLATILITY.value: 'التقلب',
        IndicatorType.SENTIMENT.value: 'معنويات السوق',
        IndicatorType.STRUCTURE.value: 'هيكل السعر',
        IndicatorType.SUPPORT_RESISTANCE.value: 'دعم/مقاومة'
    }

    INDICATOR_DESCRIPTIONS = {
        IndicatorType.TREND.value: 'يقيس اتجاه السعر بناءً على المتوسطات المتحركة',
        IndicatorType.MOMENTUM.value: 'يقيس سرعة التغير باستخدام RSI ومعدل التغير',
        IndicatorType.VOLUME.value: 'يقيس نشاط التداول مقارنة بالمتوسط',
        IndicatorType.VOLATILITY.value: 'يقيس تقلبات السعر باستخدام Bollinger Bands',
        IndicatorType.SENTIMENT.value: 'مؤشر الخوف والجشع',
        IndicatorType.STRUCTURE.value: 'يحلل القمم والقيعان المحلية',
        IndicatorType.SUPPORT_RESISTANCE.value: 'يحدد مستويات الدعم والمقاومة القريبة'
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
# عميل Binance خفيف باستخدام CCXT
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
        """جلب الشموع كقائمة بسيطة بدلاً من DataFrame"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv  # قائمة من القوائم [timestamp, open, high, low, close, volume]
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
# مؤشر الخوف والجشع (مبسط مع كاش)
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
        if value >= 80:
            return 0.20
        elif value >= 60:
            return 0.40
        elif value >= 40:
            return 0.60
        elif value >= 20:
            return 0.80
        else:
            return 0.95

# ======================
# حاسبات المؤشرات - بدون Pandas/Numpy
# ======================
class IndicatorCalculator:
    @staticmethod
    def sma(prices: List[float], period: int) -> List[Optional[float]]:
        """المتوسط المتحرك البسيط"""
        result = []
        for i in range(len(prices)):
            if i < period - 1:
                result.append(None)
            else:
                result.append(sum(prices[i - period + 1:i + 1]) / period)
        return result

    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """المتوسط المتحرك الأسي"""
        if not prices:
            return []
        k = 2 / (period + 1)
        ema_values = [prices[0]]
        for i in range(1, len(prices)):
            ema_values.append(prices[i] * k + ema_values[-1] * (1 - k))
        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
        """مؤشر القوة النسبية"""
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
    def trend_strength(close_prices: List[float]) -> float:
        """قوة الاتجاه المبسطة"""
        if len(close_prices) < 30:
            return 0.5
        sma_20 = IndicatorCalculator.sma(close_prices, 20)[-1]
        sma_50 = IndicatorCalculator.sma(close_prices, 50)[-1]
        sma_100 = IndicatorCalculator.sma(close_prices, 100)[-1]
        if None in (sma_20, sma_50, sma_100):
            return 0.5

        current = close_prices[-1]
        score = 0.0
        # السعر فوق المتوسطات
        if current > sma_20:
            score += 0.4
        if current > sma_50:
            score += 0.3
        if current > sma_100:
            score += 0.3
        # ترتيب المتوسطات
        if sma_20 > sma_50 > sma_100:
            score += 0.3
        elif sma_20 < sma_50 < sma_100:
            score -= 0.2

        return max(0.0, min(1.0, (score + 1) / 2))

    @staticmethod
    def momentum(close_prices: List[float]) -> float:
        """الزخم المبسط (RSI + ROC)"""
        if len(close_prices) < 20:
            return 0.5

        # RSI
        rsi_vals = IndicatorCalculator.rsi(close_prices, 14)
        last_rsi = rsi_vals[-1] if rsi_vals[-1] is not None else 50
        rsi_score = 1.0 - (last_rsi / 100)  # كلما قل الـ RSI كان أفضل للشراء
        if last_rsi < 30:
            rsi_score = 0.9
        elif last_rsi > 70:
            rsi_score = 0.2

        # ROC (معدل التغير)
        roc_14 = (close_prices[-1] - close_prices[-14]) / close_prices[-14] * 100
        roc_score = max(0.0, min(1.0, (roc_14 + 5) / 10))

        return (rsi_score * 0.6 + roc_score * 0.4)

    @staticmethod
    def volume_analysis(volumes: List[float], close_prices: List[float]) -> float:
        """تحليل الحجم"""
        if len(volumes) < 20:
            return 0.5
        current_vol = volumes[-1]
        avg_vol = sum(volumes[-20:]) / 20
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        if ratio > 2.0:
            score = 0.8
        elif ratio > 1.5:
            score = 0.7
        elif ratio > 1.0:
            score = 0.6
        elif ratio > 0.7:
            score = 0.5
        else:
            score = 0.4

        # تأكيد الاتجاه
        price_change = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
        if price_change > 0.01 and score > 0.6:
            score += 0.1
        elif price_change < -0.01 and score < 0.5:
            score -= 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def volatility(high: List[float], low: List[float], close: List[float]) -> float:
        """التقلب - Bollinger Bands عرض النطاق"""
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
        # القرب من الحدود
        if position > 0.8:
            return 0.2  # مقاومة
        elif position < 0.2:
            return 0.8  # دعم
        else:
            return 0.5

    @staticmethod
    def price_structure(high: List[float], low: List[float], close: List[float]) -> float:
        """هيكل السعر - القمم والقيعان"""
        if len(high) < 30:
            return 0.5

        # آخر 30 شمعة
        recent_high = max(high[-30:])
        recent_low = min(low[-30:])
        current = close[-1]

        if recent_high == recent_low:
            return 0.5

        position = (current - recent_low) / (recent_high - recent_low)
        if position > 0.8:
            return 0.3  # قرب قمة
        elif position < 0.2:
            return 0.7  # قرب قاع
        else:
            return 0.5

    @staticmethod
    def support_resistance(high: List[float], low: List[float], close: List[float]) -> float:
        """دعم ومقاومة مبسط"""
        if len(high) < 40:
            return 0.5

        # قمم وقيعان بسيطة
        highs = high[-40:]
        lows = low[-40:]
        resistance_candidates = []
        support_candidates = []

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_candidates.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_candidates.append(lows[i])

        if not resistance_candidates and not support_candidates:
            return 0.5

        current = close[-1]
        # أقرب مقاومة
        closest_resistance = min([r for r in resistance_candidates if r > current], default=None)
        # أقرب دعم
        closest_support = max([s for s in support_candidates if s < current], default=None)

        if closest_resistance and closest_support:
            distance_to_resistance = (closest_resistance - current) / current
            distance_to_support = (current - closest_support) / current
            if distance_to_support < 0.02:
                return 0.9  # قريب جداً من الدعم
            elif distance_to_resistance < 0.02:
                return 0.1  # قريب جداً من المقاومة
            elif distance_to_support < 0.05:
                return 0.7
            elif distance_to_resistance < 0.05:
                return 0.3
        return 0.5

# ======================
# معالج الإشارات
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
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.BUY]:
            return SignalType.BUY
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.NEUTRAL]:
            return SignalType.NEUTRAL
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.SELL]:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL

    @staticmethod
    def get_signal_strength(percentage: float) -> str:
        if percentage >= 85:
            return "قوية جداً"
        elif percentage >= 70:
            return "قوية"
        elif percentage >= 55:
            return "متوسطة"
        elif percentage >= 40:
            return "ضعيفة"
        else:
            return "ضعيفة جداً"

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
# مدير الإشعارات (مبسط)
# ======================
class NotificationManager:
    def __init__(self):
        self.history: List[Notification] = []
        self.max_history = 50
        self.last_notification_time = {}
        self.min_interval = 300  # 5 دقائق

    def add(self, notification: Notification):
        self.history.append(notification)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, limit: int = 10) -> List[Notification]:
        return self.history[-limit:] if self.history else []

    def should_send(self, coin_symbol: str, percentage: float) -> bool:
        now = datetime.now()
        if coin_symbol in self.last_notification_time:
            delta = now - self.last_notification_time[coin_symbol]
            if delta.total_seconds() < self.min_interval:
                return False

        thresholds = AppConfig.SIGNAL_THRESHOLDS
        if percentage >= thresholds[SignalType.STRONG_BUY] or percentage <= thresholds[SignalType.STRONG_SELL]:
            return True
        if percentage >= thresholds[SignalType.BUY] or percentage <= thresholds[SignalType.SELL]:
            return True
        return False

    def send_ntfy(self, message: str, title: str = "Crypto Signal", priority: str = "3", tags: str = "chart") -> bool:
        try:
            headers = {
                "Title": title,
                "Priority": priority,
                "Tags": tags,
                "Content-Type": "text/plain; charset=utf-8"
            }
            resp = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"NTFY error: {e}")
            return False

    def create_notification(self, coin_signal: CoinSignal) -> Optional[Notification]:
        if not self.should_send(coin_signal.symbol, coin_signal.total_percentage):
            return None

        coin = coin_signal
        signal_type = coin.signal_type
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            emoji = "🚀"
            title = f"{emoji} {signal_type.value}: {coin.name}"
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            emoji = "⚠️"
            title = f"{emoji} {signal_type.value}: {coin.name}"
        else:
            return None

        message = (
            f"{title}\n"
            f"📊 Strength: {coin.total_percentage:.1f}%\n"
            f"💰 Price: ${coin.current_price:,.2f}\n"
            f"📈 24h: {coin.price_change_24h:+.2f}%\n"
            f"⏰ {coin.last_updated.strftime('%H:%M')}"
        )

        tags = "green_circle" if "BUY" in signal_type.value else "red_circle"
        priority = "4" if "قوي" in signal_type.value else "3"

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
# مدير الإشارات الرئيسي
# ======================
class SignalManager:
    def __init__(self):
        self.signals: Dict[str, CoinSignal] = {}
        self.history: List[Dict] = []
        self.last_update: Optional[datetime] = None
        self.fgi_fetcher = FearGreedFetcher()
        self.notification_manager = NotificationManager()
        self.binance = BinanceClient()
        self.lock = Lock()
        self.fear_greed_index = 50
        self.fear_greed_score = 0.5

    def update_all(self) -> bool:
        with self.lock:
            logger.info(f"🔄 بدء تحديث {len(AppConfig.COINS)} عملة...")
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
                    logger.error(f"خطأ في {coin.symbol}: {e}")

            self.last_update = datetime.now()
            self._save_history()
            logger.info(f"✅ تم تحديث {success_count}/{len(AppConfig.COINS)}")
            return success_count > 0

    def _process_coin(self, coin: CoinConfig) -> Optional[CoinSignal]:
        ohlcv = self.binance.fetch_ohlcv(coin.symbol, '15m', AppConfig.MAX_CANDLES)
        if not ohlcv or len(ohlcv) < 50:
            return None

        # استخراج الأعمدة
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

        # حساب المؤشرات
        scores = {
            IndicatorType.TREND.value: IndicatorCalculator.trend_strength(closes),
            IndicatorType.MOMENTUM.value: IndicatorCalculator.momentum(closes),
            IndicatorType.VOLUME.value: IndicatorCalculator.volume_analysis(volumes, closes),
            IndicatorType.VOLATILITY.value: IndicatorCalculator.volatility(highs, lows, closes),
            IndicatorType.SENTIMENT.value: self.fear_greed_score,
            IndicatorType.STRUCTURE.value: IndicatorCalculator.price_structure(highs, lows, closes),
            IndicatorType.SUPPORT_RESISTANCE.value: IndicatorCalculator.support_resistance(highs, lows, closes)
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
            'signal_strength': 'غير متوفر',
            'signal_color': 'secondary',
            'indicators': [],
            'last_updated_str': 'غير معروف',
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
            return "غير معروف"
        delta = datetime.now() - dt
        if delta.days > 0:
            return f"قبل {delta.days} يوم"
        if delta.seconds >= 3600:
            return f"قبل {delta.seconds//3600} ساعة"
        if delta.seconds >= 60:
            return f"قبل {delta.seconds//60} دقيقة"
        return "الآن"

    def get_stats(self) -> Dict:
        coins = self.get_coins_data()
        valid = [c for c in coins if c['is_valid']]
        percentages = [c['total_percentage'] for c in valid]

        strong_buy = sum(1 for c in valid if c['total_percentage'] >= 75)
        buy = sum(1 for c in valid if 60 <= c['total_percentage'] < 75)
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
            'last_update_str': self._format_time_delta(self.last_update) if self.last_update else 'غير معروف',
            'total_notifications': len(self.notification_manager.history),
            'fear_greed_index': self.fear_greed_index,
            'system_status': 'healthy' if len(valid) >= len(AppConfig.COINS) * 0.7 else 'warning'
        }

# ======================
# إنشاء التطبيق
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-signal-secret-2026')
signal_manager = SignalManager()
start_time = time.time()

# ======================
# المجدول الخلفي - يعمل كل 2 دقيقة
# ======================
scheduler = BackgroundScheduler()
scheduler.add_job(func=signal_manager.update_all, trigger="interval", seconds=AppConfig.UPDATE_INTERVAL)
scheduler.start()

# تحديث أولي فور بدء التشغيل
signal_manager.update_all()

# ======================
# المسارات (Routes)
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
        indicator_weights=AppConfig.INDICATOR_WEIGHTS,
        get_indicator_color=lambda k: AppConfig.INDICATOR_COLORS.get(k, '#2E86AB'),
        get_indicator_display_name=lambda k: AppConfig.INDICATOR_DISPLAY_NAMES.get(k, k)
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
        'message': 'تم التحديث',
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
    msg = "🧪 اختبار NTFY - النظام يعمل ✅"
    success = signal_manager.notification_manager.send_ntfy(msg, "اختبار", "4", "test_tube")
    return jsonify({'success': success})

@app.route('/api/indicator_weights')
def indicator_weights():
    return jsonify({
        'weights': AppConfig.INDICATOR_WEIGHTS,
        'display_names': AppConfig.INDICATOR_DISPLAY_NAMES,
        'colors': AppConfig.INDICATOR_COLORS
    })

# ======================
# بدء التشغيل
# ======================
if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 تشغيل Crypto Signal Analyzer v3.5.2 (خفيف)")
    logger.info(f"📊 العملات: {len(AppConfig.COINS)}")
    logger.info(f"🔄 تحديث كل {AppConfig.UPDATE_INTERVAL//60} دقيقة")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
