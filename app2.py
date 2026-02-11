"""
Crypto Signal Analyzer Bot - النسخة المحسنة والمستقرة
نسخة 3.5 - تحسين حساب المؤشرات لدقة أعلى
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import math

from flask import Flask, render_template, jsonify, request, Response
import pandas as pd
import numpy as np
import ccxt
import requests
from requests.exceptions import RequestException, Timeout
import warnings

warnings.filterwarnings('ignore')

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
# هياكل البيانات
# ======================

class SignalType(Enum):
    """أنواع الإشارات"""
    STRONG_BUY = "شراء قوي"
    BUY = "شراء"
    NEUTRAL_HIGH = "محايد موجب"
    NEUTRAL_LOW = "محايد سالب"
    SELL = "بيع"
    STRONG_SELL = "بيع قوي"


class IndicatorType(Enum):
    """أنواع المؤشرات"""
    TREND_STRENGTH = "trend_strength"
    MOMENTUM = "momentum"
    VOLUME_ANALYSIS = "volume_analysis"
    VOLATILITY = "volatility"
    MARKET_SENTIMENT = "market_sentiment"
    PRICE_STRUCTURE = "price_structure"
    SUPPORT_RESISTANCE = "support_resistance"
    MARKET_CYCLE = "market_cycle"


@dataclass
class CoinConfig:
    """إعدادات العملة"""
    symbol: str
    name: str
    base_asset: str
    quote_asset: str
    enabled: bool = True


@dataclass
class IndicatorScore:
    """نتيجة المؤشر"""
    name: str
    raw_score: float  # 0-1
    weighted_score: float  # 0-1
    percentage: float  # 0-100
    weight: float
    description: str
    color: str


@dataclass
class CoinSignal:
    """إشارة العملة"""
    symbol: str
    name: str
    current_price: float
    price_change_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    total_percentage: float  # 0-100
    signal_type: SignalType
    signal_strength: str
    signal_color: str
    indicator_scores: Dict[str, IndicatorScore]
    last_updated: datetime
    fear_greed_value: int
    price_change_since_last: Optional[float] = None
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class Notification:
    """إشعار"""
    id: str
    timestamp: datetime
    coin_symbol: str
    coin_name: str
    message: str
    notification_type: str
    signal_strength: float
    price: float
    priority: str


# ======================
# إعدادات التطبيق
# ======================

class AppConfig:
    """إعدادات التطبيق المركزية"""
    
    # العملات المدعومة
    COINS = [
        # 🥇 Top 10 Cryptocurrencies (February 2026) [citation:5]
        CoinConfig(symbol="BTC/USDT", name="Bitcoin", base_asset="BTC", quote_asset="USDT"),
        CoinConfig(symbol="ETH/USDT", name="Ethereum", base_asset="ETH", quote_asset="USDT"),
        CoinConfig(symbol="BNB/USDT", name="Binance Coin", base_asset="BNB", quote_asset="USDT"),
        CoinConfig(symbol="XRP/USDT", name="Ripple", base_asset="XRP", quote_asset="USDT"),
        CoinConfig(symbol="SOL/USDT", name="Solana", base_asset="SOL", quote_asset="USDT"),
        CoinConfig(symbol="ADA/USDT", name="Cardano", base_asset="ADA", quote_asset="USDT"),
        CoinConfig(symbol="TRX/USDT", name="TRON", base_asset="TRX", quote_asset="USDT"),
        CoinConfig(symbol="DOGE/USDT", name="Dogecoin", base_asset="DOGE", quote_asset="USDT"),
        CoinConfig(symbol="BCH/USDT", name="Bitcoin Cash", base_asset="BCH", quote_asset="USDT"),
        CoinConfig(symbol="LINK/USDT", name="Chainlink", base_asset="LINK", quote_asset="USDT"),
    
        # 🚀 New Binance Listings (January 2026) [citation:1][citation:6][citation:10]
        CoinConfig(symbol="AVAX/USDT", name="Avalanche", base_asset="AVAX", quote_asset="USDT"),
        CoinConfig(symbol="UNI/USDT", name="Uniswap", base_asset="UNI", quote_asset="USDT"),
        CoinConfig(symbol="KGST/USDT", name="KangasToken", base_asset="KGST", quote_asset="USDT"),
    
        # ⭐ Top Layer-1 & Layer-2 Solutions [citation:3][citation:7]
        CoinConfig(symbol="MATIC/USDT", name="Polygon", base_asset="MATIC", quote_asset="USDT"),
        CoinConfig(symbol="DOT/USDT", name="Polkadot", base_asset="DOT", quote_asset="USDT"),
        CoinConfig(symbol="LTC/USDT", name="Litecoin", base_asset="LTC", quote_asset="USDT"),
        CoinConfig(symbol="XLM/USDT", name="Stellar", base_asset="XLM", quote_asset="USDT"),
        CoinConfig(symbol="ATOM/USDT", name="Cosmos", base_asset="ATOM", quote_asset="USDT"),
        CoinConfig(symbol="XMR/USDT", name="Monero", base_asset="XMR", quote_asset="USDT"),
        CoinConfig(symbol="ETC/USDT", name="Ethereum Classic", base_asset="ETC", quote_asset="USDT"),
        CoinConfig(symbol="FIL/USDT", name="Filecoin", base_asset="FIL", quote_asset="USDT"),
        CoinConfig(symbol="APT/USDT", name="Aptos", base_asset="APT", quote_asset="USDT"),
        CoinConfig(symbol="ARB/USDT", name="Arbitrum", base_asset="ARB", quote_asset="USDT"),
        CoinConfig(symbol="OP/USDT", name="Optimism", base_asset="OP", quote_asset="USDT"),
        CoinConfig(symbol="SUI/USDT", name="Sui", base_asset="SUI", quote_asset="USDT"),
    ]
    
    # أوزان المؤشرات المحسنة
    INDICATOR_WEIGHTS = {
        IndicatorType.TREND_STRENGTH.value: 0.25,
        IndicatorType.MOMENTUM.value: 0.20,
        IndicatorType.VOLUME_ANALYSIS.value: 0.15,
        IndicatorType.VOLATILITY.value: 0.10,
        IndicatorType.MARKET_SENTIMENT.value: 0.08,
        IndicatorType.PRICE_STRUCTURE.value: 0.12,
        IndicatorType.SUPPORT_RESISTANCE.value: 0.10
    }
    
    # عتبات الإشارات المحسنة
    SIGNAL_THRESHOLDS = {
        SignalType.STRONG_BUY: 78,
        SignalType.BUY: 62,
        SignalType.NEUTRAL_HIGH: 56,
        SignalType.NEUTRAL_LOW: 44,
        SignalType.SELL: 38,
        SignalType.STRONG_SELL: 22
    }
    
    # عتبات الإشعارات
    NOTIFICATION_THRESHOLDS = {
        'strong_buy': 78,
        'buy': 62,
        'strong_sell': 22,
        'sell': 38,
        'significant_change': 6  # تغير بنسبة 8%
    }
    
    # إعدادات API
    UPDATE_INTERVAL = 120  # 5 دقائق بالثواني
    DATA_FETCH_TIMEOUT = 30  # ثانية
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # ثانية
    
    # ألوان المؤشرات المحسنة
    INDICATOR_COLORS = {
        IndicatorType.TREND_STRENGTH.value: '#2E86AB',
        IndicatorType.MOMENTUM.value: '#A23B72',
        IndicatorType.VOLUME_ANALYSIS.value: '#3BB273',
        IndicatorType.VOLATILITY.value: '#F18F01',
        IndicatorType.MARKET_SENTIMENT.value: '#6C757D',
        IndicatorType.PRICE_STRUCTURE.value: '#8F2D56',
        IndicatorType.SUPPORT_RESISTANCE.value: '#6A4C93'
    }
    
    # أسماء المؤشرات للعرض
    INDICATOR_DISPLAY_NAMES = {
        IndicatorType.TREND_STRENGTH.value: 'قوة الاتجاه',
        IndicatorType.MOMENTUM.value: 'الزخم',
        IndicatorType.VOLUME_ANALYSIS.value: 'تحليل الحجم',
        IndicatorType.VOLATILITY.value: 'التقلب',
        IndicatorType.MARKET_SENTIMENT.value: 'معنويات السوق',
        IndicatorType.PRICE_STRUCTURE.value: 'هيكل السعر',
        IndicatorType.SUPPORT_RESISTANCE.value: 'الدعم والمقاومة'
    }
    
    # أوصاف المؤشرات المحسنة
    INDICATOR_DESCRIPTIONS = {
        IndicatorType.TREND_STRENGTH.value: 'يقيس قوة واتجاه الاتجاه العام باستخدام متعدد المتوسطات المتحركة والانحدار الخطي',
        IndicatorType.MOMENTUM.value: 'يقيس سرعة وقوة حركة السعر باستخدام RSI المتعدد، Stochastic، ومعدل التغير',
        IndicatorType.VOLUME_ANALYSIS.value: 'يحلل نشاط التداول وعلاقة الحجم بحركة السعر مع OBV ومؤشرات الحجم المتقدمة',
        IndicatorType.VOLATILITY.value: 'يقيس مستوى التقلب باستخدام نطاقات بولينجر المتعددة والانحراف المعياري الديناميكي',
        IndicatorType.MARKET_SENTIMENT.value: 'يعكس المشاعر العامة للسوق باستخدام مؤشر الخوف والجشع وتدفق الأموال',
        IndicatorType.PRICE_STRUCTURE.value: 'يحلل هيكل السعر وأنماط الشموع الحديثة مع مؤشرات القوة النسبية',
        IndicatorType.SUPPORT_RESISTANCE.value: 'يحدد مستويات الدعم والمقاومة القريبة ويحسب احتمالية الاختراق'
    }


# ======================
# إعدادات API الخارجية
# ======================

class ExternalAPIConfig:
    """إعدادات APIs الخارجية"""
    
    # Binance
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    
    # NTFY للإشعارات
    NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'crypto_signals_alerts')
    NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
    
    # Fear & Greed Index
    FGI_API_URL = "https://api.alternative.me/fng/"
    
    # الحدود الزمنية للطلبات
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 2


# ======================
# فئات النظام الأساسية
# ======================

class DataValidationError(Exception):
    """خطأ في التحقق من صحة البيانات"""
    pass


class APIFetchError(Exception):
    """خطأ في جلب البيانات من API"""
    pass


class DataFetcher:
    """فئة أساسية لجلب البيانات مع معالجة الأخطاء"""
    
    def __init__(self):
        self.retry_count = 0
        self.max_retries = ExternalAPIConfig.MAX_RETRIES
        self.timeout = ExternalAPIConfig.REQUEST_TIMEOUT
    
    def fetch_with_retry(self, fetch_func, *args, **kwargs):
        """جلب البيانات مع إعادة المحاولة"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return fetch_func(*args, **kwargs)
            except (RequestException, Timeout, ccxt.NetworkError) as e:
                last_error = e
                logger.warning(f"محاولة {attempt + 1}/{self.max_retries + 1} فشلت: {str(e)}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay(attempt))
                else:
                    raise APIFetchError(f"فشل جلب البيانات بعد {self.max_retries + 1} محاولات") from last_error
            except Exception as e:
                raise APIFetchError(f"خطأ غير متوقع: {str(e)}") from e
        
        raise APIFetchError("فشل جلب البيانات")
    
    def retry_delay(self, attempt):
        """تأخير بين المحاولات"""
        return 2 ** attempt  # زيادة أسيّة


class BinanceDataFetcher(DataFetcher):
    """جلب البيانات من Binance مع التحقق"""
    
    def __init__(self):
        super().__init__()
        self.exchange = self._initialize_exchange()
    
    def _initialize_exchange(self):
        """تهيئة اتصال Binance"""
        try:
            exchange = ccxt.binance({
                'apiKey': ExternalAPIConfig.BINANCE_API_KEY,
                'secret': ExternalAPIConfig.BINANCE_SECRET_KEY,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            exchange.load_markets()
            logger.info("تم تهيئة اتصال Binance بنجاح")
            return exchange
        except Exception as e:
            logger.error(f"فشل تهيئة اتصال Binance: {e}")
            raise
    
    def validate_ohlcv_data(self, df: pd.DataFrame, min_rows: int = 100) -> bool:
        """التحقق من صحة بيانات OHLCV"""
        if df is None or df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        if len(df) < min_rows:
            return False
        
        # التحقق من القيم غير الصالحة
        if df[required_columns].isnull().any().any():
            return False
        
        # التحقق من التطابق المنطقي للأسعار
        if (df['high'] < df['low']).any() or (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            return False
        
        # التحقق من الاستقرار الإحصائي
        price_std = df['close'].std()
        if price_std == 0:
            return False
        
        return True
    
    def get_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 500) -> Optional[pd.DataFrame]:
        """جلب بيانات OHLCV مع التحقق (15m كإطار زمني أساسي)"""
        try:
            def fetch():
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                if not self.validate_ohlcv_data(df):
                    raise DataValidationError(f"بيانات OHLCV غير صالحة لـ {symbol}")
                
                return df
            
            return self.fetch_with_retry(fetch)
            
        except (APIFetchError, DataValidationError) as e:
            logger.error(f"خطأ في جلب بيانات OHLCV لـ {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"خطأ غير متوقع في جلب بيانات OHLCV لـ {symbol}: {e}")
            return None
    
    def get_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """جلب بيانات متعددة الإطار الزمني"""
        timeframes = {
            '15m': 500,  # زيادة عدد القضبان لتعويض الإطار الزمني الأقصر
            '1h': 200,
            '4h': 150
        }
        
        data = {}
        for tf, limit in timeframes.items():
            try:
                df = self.get_ohlcv(symbol, tf, limit)
                if df is not None and len(df) > 50:
                    data[tf] = df
            except Exception as e:
                logger.warning(f"خطأ في جلب بيانات {tf} لـ {symbol}: {e}")
        
        return data
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات التاكر مع التحقق"""
        try:
            def fetch():
                ticker = self.exchange.fetch_ticker(symbol)
                
                # التحقق الأساسي للبيانات
                required_fields = ['last', 'percentage', 'high', 'low', 'quoteVolume', 'bid', 'ask']
                if not all(field in ticker for field in required_fields):
                    raise DataValidationError(f"بيانات التاكر غير مكتملة لـ {symbol}")
                
                # حساب سيولة السوق
                spread = (ticker['ask'] - ticker['bid']) / ticker['bid'] * 100
                ticker['spread'] = spread
                
                return ticker
            
            return self.fetch_with_retry(fetch)
            
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات التاكر لـ {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> float:
        """جلب السعر الحالي"""
        ticker = self.get_ticker(symbol)
        return ticker['last'] if ticker else 0.0
    
    def get_24h_stats(self, symbol: str) -> Dict[str, float]:
        """جلب إحصائيات 24 ساعة"""
        ticker = self.get_ticker(symbol)
        if ticker:
            return {
                'change': ticker.get('percentage', 0.0),
                'high': ticker.get('high', 0.0),
                'low': ticker.get('low', 0.0),
                'volume': ticker.get('quoteVolume', 0.0),
                'bid': ticker.get('bid', 0.0),
                'ask': ticker.get('ask', 0.0),
                'spread': ticker.get('spread', 0.0)
            }
        return {'change': 0.0, 'high': 0.0, 'low': 0.0, 'volume': 0.0, 'bid': 0.0, 'ask': 0.0, 'spread': 0.0}


class FearGreedIndexFetcher(DataFetcher):
    """جلب مؤشر الخوف والجشع"""
    
    def __init__(self):
        super().__init__()
        self.last_value = 50
        self.last_update = None
        self.cache_duration = 300  # 5 دقائق بالثواني
        self.history_values = []
        self.max_history = 10
    
    def get_index(self) -> Tuple[float, int, str]:
        """جلب قيمة المؤشر مع التخزين المؤقت والاتجاه"""
        # التحقق من التخزين المؤقت
        if (self.last_update and 
            (datetime.now() - self.last_update).total_seconds() < self.cache_duration):
            return self._convert_to_score(self.last_value), self.last_value, self._get_trend()
        
        try:
            def fetch():
                response = requests.get(
                    ExternalAPIConfig.FGI_API_URL, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    fgi_value = int(data['data'][0]['value'])
                    
                    # التحقق من القيمة
                    if not 0 <= fgi_value <= 100:
                        raise DataValidationError(f"قيمة FGI غير صالحة: {fgi_value}")
                    
                    return fgi_value
                else:
                    raise DataValidationError("بيانات FGI غير مكتملة")
            
            fgi_value = self.fetch_with_retry(fetch)
            
            # تحديث التاريخ
            self.history_values.append(fgi_value)
            if len(self.history_values) > self.max_history:
                self.history_values.pop(0)
            
            # تحديث التخزين المؤقت
            self.last_value = fgi_value
            self.last_update = datetime.now()
            
            return self._convert_to_score(fgi_value), fgi_value, self._get_trend()
            
        except Exception as e:
            logger.error(f"خطأ في جلب مؤشر الخوف والجشع: {e}")
            # استخدام القيمة المخزنة مؤقتاً إذا فشل الجلب
            return self._convert_to_score(self.last_value), self.last_value, self._get_trend()
    
    def _convert_to_score(self, fgi_value: int) -> float:
        """تحويل عكسي - الجشع الشديد = إشارة بيع"""
        if fgi_value >= 80:  # جشع شديد
            return 0.15  # ← إشارة بيع (ليس 0.85!)
        elif fgi_value >= 60:  # جشع
            return 0.35  # ← إشارة بيع خفيفة
        elif fgi_value >= 40:  # محايد
            return 0.55
        elif fgi_value >= 20:  # خوف
            return 0.75  # ← إشارة شراء
        else:  # خوف شديد
            return 0.95  # ← إشارة شراء قوية
    
    def _get_trend(self) -> str:
        """الحصول على اتجاه المؤشر"""
        if len(self.history_values) < 2:
            return "ثابت"
        
        if self.history_values[-1] > self.history_values[-2]:
            return "صاعد"
        elif self.history_values[-1] < self.history_values[-2]:
            return "هابط"
        else:
            return "ثابت"


class IndicatorsCalculator:
    """حساب المؤشرات المحسنة مع تحليل متقدم"""
    
    @staticmethod
    def validate_score(score: float, indicator_name: str) -> float:
        """التحقق من صحة النتيجة وتطبيعها"""
        if score is None or np.isnan(score) or np.isinf(score):
            logger.warning(f"نتيجة {indicator_name} غير صالحة، استخدام القيمة الافتراضية")
            return 0.5
        
        # تطبيع بين 0 و1 مع تدرج سلس
        normalized = max(0.0, min(1.0, float(score)))
        return normalized
    
    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame, multiple_tf_data: Dict[str, pd.DataFrame] = None) -> float:
        """حساب قوة الاتجاه المحسن (معدل للإطار الزمني 15m)"""
        try:
            if len(df) < 100:
                return 0.5
            
            current_price = df['close'].iloc[-1]
            
            # 1. تحليل متعدد المتوسطات المتحركة (معدلة للإطار الزمني 15m)
            ma_periods = [12, 24, 80, 160, 320]  # فترات معادلة للفترات الأصلية في 15m
            ma_scores = []
            ma_weights = []
            
            for i, period in enumerate(ma_periods):
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean().iloc[-1]
                    ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
                    
                    if pd.notna(sma) and sma > 0:
                        # حساب المسافة النسبية مع معايرة دقيقة
                        price_to_sma = current_price / sma
                        price_to_ema = current_price / ema
                        
                        # تسجيل الاتجاه (SMA فوق/تحت)
                        ma_position_score = 1.0 if price_to_sma > 1.0 else 0.0
                        
                        # حساب الزخم النسبي
                        if period >= 20:
                            sma_prev = df['close'].rolling(window=period).mean().iloc[-2]
                            ema_prev = df['close'].ewm(span=period, adjust=False).mean().iloc[-2]
                            sma_momentum = (sma - sma_prev) / sma_prev if sma_prev > 0 else 0
                            ema_momentum = (ema - ema_prev) / ema_prev if ema_prev > 0 else 0
                            momentum_score = (sma_momentum + ema_momentum) * 500  # تكبير للتأثير
                        else:
                            momentum_score = 0
                        
                        # النتيجة المركبة مع وزن حسب الفترة
                        period_weight = 1.0 / (i + 1) ** 0.7
                        score = (0.4 * ma_position_score + 
                                0.3 * min(1.0, max(0.0, price_to_sma - 0.9) * 10) +
                                0.3 * min(1.0, max(0.0, 0.5 + momentum_score)))
                        
                        ma_scores.append(score)
                        ma_weights.append(period_weight)
            
            # 2. تحليل الانحدار الخطي للاتجاه
            if len(df) >= 80:  # زيادة الحد الأدنى للإطار الزمني 15m
                # انحدار قصير المدى (80 فترة = 20 ساعة)
                x_short = np.arange(80)
                y_short = df['close'].tail(80).values
                slope_short, _ = np.polyfit(x_short, y_short, 1)
                slope_pct_short = slope_short / y_short[0] if y_short[0] > 0 else 0
                
                # انحدار متوسط المدى (200 فترة = 50 ساعة)
                x_medium = np.arange(min(200, len(df)))
                y_medium = df['close'].tail(min(200, len(df))).values
                slope_medium, _ = np.polyfit(x_medium, y_medium, 1)
                slope_pct_medium = slope_medium / y_medium[0] if y_medium[0] > 0 else 0
                
                # حساب قوة الاتجاه من الانحدار
                regression_score = (0.6 * min(1.0, max(0.0, 0.5 + slope_pct_short * 100)) +
                                  0.4 * min(1.0, max(0.0, 0.5 + slope_pct_medium * 50)))
            else:
                regression_score = 0.5
            
            # 3. تحليل الاتجاه عبر أطر زمنية متعددة
            multi_tf_score = 0.5
            if multiple_tf_data:
                tf_scores = []
                for tf, tf_df in multiple_tf_data.items():
                    if len(tf_df) > 50:
                        tf_current = tf_df['close'].iloc[-1]
                        tf_sma_20 = tf_df['close'].rolling(window=20).mean().iloc[-1]
                        if tf_sma_20 > 0:
                            tf_score = 1.0 if tf_current > tf_sma_20 else 0.0
                            tf_scores.append(tf_score)
                
                if tf_scores:
                    multi_tf_score = np.mean(tf_scores)
            
            # 4. حساب النهائي المركب
            if ma_scores:
                ma_weighted = np.average(ma_scores, weights=ma_weights)
                
                # دمج جميع المؤشرات
                final_score = (0.45 * ma_weighted + 
                             0.30 * regression_score + 
                             0.25 * multi_tf_score)
            else:
                final_score = 0.5
            
            return IndicatorsCalculator.validate_score(final_score, "قوة الاتجاه")
            
        except Exception as e:
            logger.error(f"خطأ في حساب قوة الاتجاه: {e}")
            return 0.5
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame) -> float:
        """حساب الزخم المحسن (معدل للإطار الزمني 15m)"""
        try:
            if len(df) < 80:  # زيادة الحد الأدنى للإطار الزمني 15m
                return 0.5
            
            # 1. RSI متعدد الفترات (فترات معادلة)
            rsi_scores = []
            rsi_weights = []
            
            for period in [28, 56, 84]:  # معادلة لفترات 7، 14، 21 في 1H
                if len(df) >= period * 2:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_value = rsi.iloc[-1] if not rsi.empty else 50
                    
                    # تحويل RSI إلى درجة غير خطية
                    if rsi_value <= 25:
                        score = 0.95 + (25 - rsi_value) * 0.002
                    elif rsi_value <= 40:
                        score = 0.85 - (40 - rsi_value) * 0.02
                    elif rsi_value <= 60:
                        score = 0.55 - (60 - rsi_value) * 0.02
                    elif rsi_value <= 75:
                        score = 0.25 - (75 - rsi_value) * 0.02
                    else:
                        score = 0.05 - (100 - rsi_value) * 0.002
                    
                    rsi_scores.append(score)
                    rsi_weights.append(1.0 / period)
            
            rsi_final = np.average(rsi_scores, weights=rsi_weights) if rsi_scores else 0.5
            
            # 2. Stochastic RSI محسن
            if len(df) >= 84:  # معادلة لفترة 21 في 1H
                # حساب RSI أولاً
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=56).mean()  # 14 فترة في 1H
                avg_loss = loss.rolling(window=56).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # حساب Stochastic على RSI
                rsi_low = rsi.rolling(window=56).min()
                rsi_high = rsi.rolling(window=56).max()
                stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
                stoch_value = stoch_rsi.iloc[-1] if not stoch_rsi.empty else 50
                
                # تحويل Stochastic RSI
                if stoch_value <= 20:
                    stoch_score = 0.9
                elif stoch_value <= 30:
                    stoch_score = 0.7
                elif stoch_value <= 70:
                    stoch_score = 0.5
                elif stoch_value <= 80:
                    stoch_score = 0.3
                else:
                    stoch_score = 0.1
            else:
                stoch_score = 0.5
            
            # 3. معدل التغير (ROC) متعدد الفترات
            roc_scores = []
            for period in [12, 28, 56, 84]:  # معادلة لفترات 3، 7، 14، 21 في 1H
                if len(df) >= period:
                    roc = ((df['close'].iloc[-1] - df['close'].iloc[-period]) / 
                           df['close'].iloc[-period]) * 100
                    
                    # تحويل ROC إلى درجة مع منحنى سيجمويد
                    roc_normalized = roc / 20  # تقسيم للحصول على نطاق معقول
                    roc_score = 1.0 / (1.0 + math.exp(-roc_normalized))
                    roc_scores.append(roc_score)
            
            roc_final = np.mean(roc_scores) if roc_scores else 0.5
            
            # 4. مؤشر الزخم المطلق
            momentum_periods = [20, 40, 80]  # معادلة لفترات 5، 10، 20 في 1H
            momentum_scores = []
            for period in momentum_periods:
                if len(df) >= period:
                    momentum = df['close'].iloc[-1] - df['close'].iloc[-period]
                    momentum_pct = momentum / df['close'].iloc[-period] if df['close'].iloc[-period] > 0 else 0
                    momentum_score = 1.0 / (1.0 + math.exp(-momentum_pct * 100))
                    momentum_scores.append(momentum_score)
            
            momentum_final = np.mean(momentum_scores) if momentum_scores else 0.5
            
            # 5. حساب النتيجة النهائية المرجحة
            final_score = (0.35 * rsi_final + 
                         0.25 * stoch_score + 
                         0.25 * roc_final + 
                         0.15 * momentum_final)
            
            return IndicatorsCalculator.validate_score(final_score, "الزخم")
            
        except Exception as e:
            logger.error(f"خطأ في حساب الزخم: {e}")
            return 0.5
    
    @staticmethod
    def calculate_volume_analysis(df: pd.DataFrame, price_change_24h: float = 0) -> float:
        """تحليل الحجم المحسن (معدل للإطار الزمني 15m)"""
        try:
            if len(df) < 80:  # زيادة الحد الأدنى للإطار الزمني 15m
                return 0.5
            
            current_volume = df['volume'].iloc[-1]
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # 1. مؤشر OBV (On-Balance Volume) محسن
            obv = 0
            price_changes = df['close'].diff()
            volumes = df['volume']
            
            for i in range(1, min(80, len(df))):  # زيادة الفترة
                if price_changes.iloc[i] > 0:
                    obv += volumes.iloc[i]
                elif price_changes.iloc[i] < 0:
                    obv -= volumes.iloc[i]
            
            # حساب OBV السلس
            obv_sma = pd.Series([obv]).rolling(window=20).mean().iloc[-1] if len(df) >= 20 else obv
            
            # تحويل OBV إلى درجة
            if len(df) >= 40:
                obv_avg = abs(df['volume'].tail(40).mean())
                if obv_avg > 0:
                    obv_ratio = obv_sma / obv_avg
                    if obv_ratio > 1.5:
                        obv_score = 0.9
                    elif obv_ratio > 1.2:
                        obv_score = 0.7
                    elif obv_ratio > 0.8:
                        obv_score = 0.5
                    elif obv_ratio > 0.5:
                        obv_score = 0.3
                    else:
                        obv_score = 0.1
                else:
                    obv_score = 0.5
            else:
                obv_score = 0.5
            
            # 2. نسبة الحجم إلى المتوسطات
            volume_ratios = []
            for period in [28, 56, 84]:  # معادلة لفترات 7، 14، 21 في 1H
                if len(df) >= period:
                    avg_volume = df['volume'].tail(period).mean()
                    if avg_volume > 0:
                        ratio = current_volume / avg_volume
                        volume_ratios.append(ratio)
            
            if volume_ratios:
                avg_ratio = np.mean(volume_ratios)
                
                # تحويل النسبة إلى درجة مع مراعاة اتجاه السعر
                if avg_ratio > 3.0:
                    base_score = 0.9 if price_change > 0 else 0.1
                elif avg_ratio > 2.0:
                    base_score = 0.75 if price_change > 0 else 0.25
                elif avg_ratio > 1.5:
                    base_score = 0.6 if price_change > 0 else 0.4
                elif avg_ratio > 1.0:
                    base_score = 0.55 if price_change > 0 else 0.45
                elif avg_ratio > 0.7:
                    base_score = 0.5
                elif avg_ratio > 0.5:
                    base_score = 0.45 if price_change < 0 else 0.55
                else:
                    base_score = 0.3 if price_change < 0 else 0.7
            else:
                base_score = 0.5
            
            # 3. مؤشر توزيع الحجم (VWAP نسبي)
            if len(df) >= 40:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (typical_price * df['volume']).rolling(window=40).sum() / df['volume'].rolling(window=40).sum()
                current_vwap = vwap.iloc[-1]
                
                if current_vwap > 0:
                    vwap_position = current_price / current_vwap
                    if vwap_position > 1.02:
                        vwap_score = 0.8
                    elif vwap_position > 1.0:
                        vwap_score = 0.6
                    elif vwap_position > 0.98:
                        vwap_score = 0.4
                    else:
                        vwap_score = 0.2
                else:
                    vwap_score = 0.5
            else:
                vwap_score = 0.5
            
            # 4. دمج النتائج مع أوزان
            final_score = (0.40 * base_score + 
                         0.35 * obv_score + 
                         0.25 * vwap_score)
            
            # 5. تعديل بناء على تغير السعر في 24 ساعة
            if abs(price_change_24h) > 10:
                # إذا كان التغير كبير، زيادة تأثير الحجم
                volume_impact = min(1.0, max(0.0, final_score + (price_change_24h / 100)))
                final_score = (0.7 * volume_impact + 0.3 * final_score)
            
            return IndicatorsCalculator.validate_score(final_score, "تحليل الحجم")
            
        except Exception as e:
            logger.error(f"خطأ في حساب تحليل الحجم: {e}")
            return 0.5
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame) -> float:
        """حساب التقلب المحسن (معدل للإطار الزمني 15m)"""
        try:
            if len(df) < 80:  # زيادة الحد الأدنى للإطار الزمني 15m
                return 0.5
            
            current_price = df['close'].iloc[-1]
            
            # 1. بولينجر باند متعدد المستويات
            bb_scores = []
            
            for std_dev in [1.5, 2.0, 2.5]:
                if len(df) >= 40:  # 20 فترة في 1H
                    sma_40 = df['close'].rolling(window=40).mean()
                    std_40 = df['close'].rolling(window=40).std()
                    
                    upper_band = sma_40 + (std_40 * std_dev)
                    lower_band = sma_40 - (std_40 * std_dev)
                    
                    current_sma = sma_40.iloc[-1]
                    current_upper = upper_band.iloc[-1]
                    current_lower = lower_band.iloc[-1]
                    
                    if current_upper - current_lower > 0:
                        position = (current_price - current_lower) / (current_upper - current_lower)
                        
                        # تحليل الموقع في النطاق
                        if position > 0.9:
                            score = 0.1  # قرب المقاومة القوية
                        elif position > 0.75:
                            score = 0.3  # قرب المقاومة
                        elif position > 0.6:
                            score = 0.45  # منطقة مقاومة محتملة
                        elif position > 0.4:
                            score = 0.5  # منطقة محايدة
                        elif position > 0.25:
                            score = 0.55  # منطقة دعم محتملة
                        elif position > 0.1:
                            score = 0.7  # قرب الدعم
                        else:
                            score = 0.9  # قرب الدعم القوي
                        
                        bb_scores.append(score)
            
            bb_final = np.mean(bb_scores) if bb_scores else 0.5
            
            # 2. نسبة التقلب التاريخي
            volatility_scores = []
            for period in [40, 80, 120]:  # معادلة لفترات 10، 20، 30 في 1H
                if len(df) >= period:
                    returns = df['close'].pct_change().tail(period)
                    hist_volatility = returns.std() * math.sqrt(365 * 24)  # سنوي مع تعديل لـ 15m
                    
                    # تحويل التقلب التاريخي إلى درجة
                    if hist_volatility > 1.5:
                        vol_score = 0.2  # تقلب عالي جداً - خطير
                    elif hist_volatility > 1.0:
                        vol_score = 0.3
                    elif hist_volatility > 0.6:
                        vol_score = 0.5
                    elif hist_volatility > 0.3:
                        vol_score = 0.7
                    else:
                        vol_score = 0.9  # تقلب منخفض - فرصة جيدة
                    
                    volatility_scores.append(vol_score)
            
            hist_vol_final = np.mean(volatility_scores) if volatility_scores else 0.5
            
            # 3. مؤشر نطاق التداول (ATR نسبي)
            if len(df) >= 56:  # 14 فترة في 1H
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=56).mean()
                current_atr = atr.iloc[-1]
                
                if current_price > 0:
                    atr_percent = (current_atr / current_price) * 100
                    
                    if atr_percent > 3:
                        atr_score = 0.2
                    elif atr_percent > 2:
                        atr_score = 0.35
                    elif atr_percent > 1:
                        atr_score = 0.5
                    elif atr_percent > 0.5:
                        atr_score = 0.65
                    else:
                        atr_score = 0.8
                else:
                    atr_score = 0.5
            else:
                atr_score = 0.5
            
            # 4. دمج النتائج
            final_score = (0.40 * bb_final + 
                         0.35 * hist_vol_final + 
                         0.25 * atr_score)
            
            return IndicatorsCalculator.validate_score(final_score, "التقلب")
            
        except Exception as e:
            logger.error(f"خطأ في حساب التقلب: {e}")
            return 0.5
    
    @staticmethod
    def calculate_price_structure(df: pd.DataFrame) -> float:
        """تحليل هيكل السعر المحسن (معدل للإطار الزمني 15m)"""
        try:
            if len(df) < 40:  # زيادة الحد الأدنى للإطار الزمني 15m
                return 0.5
            
            # 1. تحليل الشموع المتقدم (آخر 20 شمعة)
            recent_candles = df.tail(20)
            candle_analysis_scores = []
            
            for i in range(len(recent_candles)):
                candle = recent_candles.iloc[i]
                open_price = candle['open']
                close_price = candle['close']
                high_price = candle['high']
                low_price = candle['low']
                
                # حساب قوة الشمعة
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 0:
                    body_ratio = body_size / total_range
                    
                    # تحديد نوع الشمعة
                    if close_price > open_price:  # شمعة صاعدة
                        if body_ratio > 0.7:
                            candle_strength = 0.9  # شمعة صاعدة قوية
                        elif body_ratio > 0.4:
                            candle_strength = 0.7  # شمعة صاعدة متوسطة
                        else:
                            candle_strength = 0.6  # شمعة دوجي صاعدة
                    else:  # شمعة هابطة
                        if body_ratio > 0.7:
                            candle_strength = 0.1  # شمعة هابطة قوية
                        elif body_ratio > 0.4:
                            candle_strength = 0.3  # شمعة هابطة متوسطة
                        else:
                            candle_strength = 0.4  # شمعة دوجي هابطة
                    
                    # حساب ظلال الشمعة
                    upper_shadow = high_price - max(open_price, close_price)
                    lower_shadow = min(open_price, close_price) - low_price
                    
                    if upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
                        candle_strength *= 0.8  # مقاومة قوية
                    elif lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                        candle_strength *= 1.2  # دعم قوي
                    
                    candle_analysis_scores.append(candle_strength)
            
            candle_score = np.mean(candle_analysis_scores) if candle_analysis_scores else 0.5
            
            # 2. تحليل الهيكل السعري (العاليات والمنخفضات)
            if len(df) >= 60:
                # العثور على القمم والقيعان المحلية
                highs = []
                lows = []
                
                for i in range(10, len(df) - 10):  # زيادة النطاق للإطار الزمني 15m
                    if (df['high'].iloc[i] == df['high'].iloc[i-10:i+11].max() and
                        df['high'].iloc[i] > df['high'].iloc[i-1] and
                        df['high'].iloc[i] > df['high'].iloc[i+1]):
                        highs.append(df['high'].iloc[i])
                    
                    if (df['low'].iloc[i] == df['low'].iloc[i-10:i+11].min() and
                        df['low'].iloc[i] < df['low'].iloc[i-1] and
                        df['low'].iloc[i] < df['low'].iloc[i+1]):
                        lows.append(df['low'].iloc[i])
                
                current_price = df['close'].iloc[-1]
                
                # حساب موقع السعر بالنسبة للهيكل
                if highs and lows:
                    recent_high = max(highs[-4:]) if len(highs) >= 4 else max(highs)
                    recent_low = min(lows[-4:]) if len(lows) >= 4 else min(lows)
                    
                    if recent_high - recent_low > 0:
                        structure_position = (current_price - recent_low) / (recent_high - recent_low)
                        
                        if structure_position > 0.8:
                            structure_score = 0.3  # قرب قمة جديدة
                        elif structure_position > 0.6:
                            structure_score = 0.45
                        elif structure_position > 0.4:
                            structure_score = 0.55
                        elif structure_position > 0.2:
                            structure_score = 0.65
                        else:
                            structure_score = 0.8  # قرب قاع جديد
                    else:
                        structure_score = 0.5
                else:
                    structure_score = 0.5
            else:
                structure_score = 0.5
            
            # 3. تحليل تقاطع الأسعار
            if len(df) >= 20:
                price_cross_scores = []
                
                # تحليل تقاطعات قصيرة المدى
                for i in range(1, 10):  # زيادة النطاق للإطار الزمني 15m
                    if len(df) >= i + 10:
                        current = df['close'].iloc[-1]
                        prev = df['close'].iloc[-i-1]
                        
                        if current > prev:
                            cross_score = 0.6 + (0.4 / i)  # كلما كان التقاطع أحدث، كلما زادت النتيجة
                        elif current < prev:
                            cross_score = 0.4 - (0.4 / i)
                        else:
                            cross_score = 0.5
                        
                        price_cross_scores.append(cross_score)
                
                cross_final = np.mean(price_cross_scores) if price_cross_scores else 0.5
            else:
                cross_final = 0.5
            
            # 4. حساب النتيجة النهائية
            final_score = (0.40 * candle_score + 
                         0.35 * structure_score + 
                         0.25 * cross_final)
            
            return IndicatorsCalculator.validate_score(final_score, "هيكل السعر")
            
        except Exception as e:
            logger.error(f"خطأ في حساب هيكل السعر: {e}")
            return 0.5
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> float:
        """تحليل الدعم والمقاومة المحسن (معدل للإطار الزمني 15m)"""
        try:
            if len(df) < 80:  # زيادة الحد الأدنى للإطار الزمني 15m
                return 0.5
            
            current_price = df['close'].iloc[-1]
            
            # 1. تحديد مستويات الدعم والمقاومة من القمم والقيعان
            support_levels = []
            resistance_levels = []
            
            # البحث عن القيعان (الدعم)
            for i in range(40, len(df) - 10):  # زيادة النطاق للإطار الزمني 15m
                if (df['low'].iloc[i] == df['low'].iloc[i-40:i+11].min() and
                    df['low'].iloc[i] < df['low'].iloc[i-1] and
                    df['low'].iloc[i] < df['low'].iloc[i+1]):
                    support_levels.append(df['low'].iloc[i])
            
            # البحث عن القمم (المقاومة)
            for i in range(40, len(df) - 10):
                if (df['high'].iloc[i] == df['high'].iloc[i-40:i+11].max() and
                    df['high'].iloc[i] > df['high'].iloc[i-1] and
                    df['high'].iloc[i] > df['high'].iloc[i+1]):
                    resistance_levels.append(df['high'].iloc[i])
            
            # 2. حساب أقرب مستويات الدعم والمقاومة
            closest_support = None
            closest_resistance = None
            
            for level in sorted(support_levels, reverse=True):
                if level < current_price:
                    closest_support = level
                    break
            
            for level in sorted(resistance_levels):
                if level > current_price:
                    closest_resistance = level
                    break
            
            # 3. حساب نسبة القرب من المستويات
            if closest_support and closest_resistance:
                price_range = closest_resistance - closest_support
                if price_range > 0:
                    position_from_support = (current_price - closest_support) / price_range
                    
                    # تحويل الموقع إلى درجة
                    if position_from_support < 0.2:
                        position_score = 0.85  # قريب جداً من الدعم
                    elif position_from_support < 0.4:
                        position_score = 0.70
                    elif position_from_support < 0.6:
                        position_score = 0.50
                    elif position_from_support < 0.8:
                        position_score = 0.30
                    else:
                        position_score = 0.15  # قريب جداً من المقاومة
                else:
                    position_score = 0.5
            else:
                position_score = 0.5
            
            # 4. تحليل قوة المستويات بناءً على عدد المرات التي تم اختبارها
            support_strength = 0
            resistance_strength = 0
            
            if closest_support and len(df) > 80:
                # حساب عدد المرات التي تم فيها اختبار مستوى الدعم
                support_tests = 0
                for i in range(max(0, len(df) - 160), len(df)):  # زيادة الفترة
                    if abs(df['low'].iloc[i] - closest_support) / closest_support < 0.01:  # ضمن 1%
                        support_tests += 1
                
                support_strength = min(1.0, support_tests / 15)  # قوة تصل إلى 1.0 بعد 15 اختبارات
            
            if closest_resistance and len(df) > 80:
                # حساب عدد المرات التي تم فيها اختبار مستوى المقاومة
                resistance_tests = 0
                for i in range(max(0, len(df) - 160), len(df)):
                    if abs(df['high'].iloc[i] - closest_resistance) / closest_resistance < 0.01:  # ضمن 1%
                        resistance_tests += 1
                
                resistance_strength = min(1.0, resistance_tests / 15)
            
            # 5. حساب النتيجة النهائية
            strength_factor = 1.0
            if position_score > 0.6:  # قرب الدعم
                strength_factor += support_strength * 0.3
            elif position_score < 0.4:  # قرب المقاومة
                strength_factor -= resistance_strength * 0.3
            
            final_score = position_score * strength_factor
            
            return IndicatorsCalculator.validate_score(final_score, "الدعم والمقاومة")
            
        except Exception as e:
            logger.error(f"خطأ في حساب الدعم والمقاومة: {e}")
            return 0.5


class SignalProcessor:
    """معالجة الإشارات"""
    
    @staticmethod
    def calculate_weighted_score(indicator_scores: Dict[str, float]) -> Dict[str, Any]:
        """حساب الإشارة المرجحة"""
        total_weighted = 0.0
        weighted_scores = {}
        
        for indicator, score in indicator_scores.items():
            if indicator in AppConfig.INDICATOR_WEIGHTS:
                weight = AppConfig.INDICATOR_WEIGHTS[indicator]
                weighted = score * weight
                
                weighted_scores[indicator] = IndicatorScore(
                    name=indicator,
                    raw_score=score,
                    weighted_score=weighted,
                    percentage=weighted * 100,
                    weight=weight,
                    description=AppConfig.INDICATOR_DESCRIPTIONS.get(indicator, ''),
                    color=AppConfig.INDICATOR_COLORS.get(indicator, '#2E86AB')
                )
                
                total_weighted += weighted
        
        total_percentage = total_weighted * 100
        
        return {
            'total_percentage': total_percentage,
            'weighted_scores': weighted_scores,
            'signal_type': SignalProcessor.get_signal_type(total_percentage),
            'signal_strength': SignalProcessor.get_signal_strength(total_percentage),
            'signal_color': SignalProcessor.get_signal_color(total_percentage)
        }
    
    @staticmethod
    def get_signal_type(percentage: float) -> SignalType:
        """تحديد نوع الإشارة"""
        if percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.STRONG_BUY]:
            return SignalType.STRONG_BUY
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.BUY]:
            return SignalType.BUY
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.NEUTRAL_HIGH]:
            return SignalType.NEUTRAL_HIGH
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.NEUTRAL_LOW]:
            return SignalType.NEUTRAL_LOW
        elif percentage >= AppConfig.SIGNAL_THRESHOLDS[SignalType.SELL]:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    @staticmethod
    def get_signal_strength(percentage: float) -> str:
        """تحديد قوة الإشارة"""
        if percentage >= 85:
            return "قوية جداً"
        elif percentage >= 70:
            return "قوية"
        elif percentage >= 58:
            return "متوسطة"
        elif percentage >= 42:
            return "ضعيفة"
        else:
            return "ضعيفة جداً"
    
    @staticmethod
    def get_signal_color(percentage: float) -> str:
        """تحديد لون الإشارة"""
        signal_type = SignalProcessor.get_signal_type(percentage)
        
        color_map = {
            SignalType.STRONG_BUY: "success",
            SignalType.BUY: "info",
            SignalType.NEUTRAL_HIGH: "secondary",
            SignalType.NEUTRAL_LOW: "warning",
            SignalType.SELL: "warning",
            SignalType.STRONG_SELL: "danger"
        }
        
        return color_map.get(signal_type, "secondary")


class NotificationManager:
    """إدارة الإشعارات"""
    
    def __init__(self):
        super().__init__()
        self.notification_history: List[Notification] = []
        self.max_history = 100
        self.last_notification_time = {}
        self.last_heartbeat = None
        self.heartbeat_interval = 7200
        
        # ✅ اختبار الاتصال عند الإنشاء
        self.test_ntfy_connection()
        
        # ✅ إرسال إشعار بدء التشغيل بعد 5 ثوانٍ
        threading.Thread(target=self._send_startup_notification, daemon=True).start()

    # في فئة NotificationManager، أضف هذه الدالة:

    def test_executor_connection(self) -> Dict:
        """اختبار اتصال البوت التنفيذ"""
        try:
            EXECUTOR_BOT_URL = os.environ.get('EXECUTOR_BOT_URL')
        
            if not EXECUTOR_BOT_URL:
                logger.warning("⚠️ EXECUTOR_BOT_URL غير معين")
                return {
                    'success': False,
                    'message': 'لم يتم تعيين EXECUTOR_BOT_URL',
                    'steps': ['❌ خطأ في الإعدادات']
                }
        
            # الخطوة 1: محاولة الوصول إلى صفحة الصحة
            logger.info(f"🔍 اختبار اتصال البوت التنفيذ: {EXECUTOR_BOT_URL}")
        
            health_url = f'{EXECUTOR_BOT_URL}/health'
            steps = []
        
            try:
                response = requests.get(health_url, timeout=5)
                steps.append(f"✅ الوصول إلى صفحة الصحة: {response.status_code}")
            
                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        steps.append(f"📊 حالة النظام: {health_data.get('status', 'غير معروف')}")
                    except:
                        steps.append("⚠️ الاستجابة ليست JSON")
                else:
                    steps.append(f"❌ صفحة الصحة غير متوفرة: {response.status_code}")
                
            except requests.exceptions.ConnectionError:
                steps.append("❌ فشل الاتصال بالبوت التنفيذ")
            except requests.exceptions.Timeout:
                steps.append("⏰ انتهت المهلة")
        
            # الخطوة 2: اختبار إرسال إشعار بسيط
            test_signal = {
                'test': True,
                'timestamp': datetime.now().isoformat(),
                'source': 'signal_bot_tester'
            }
        
            try:
                headers = {
                    'Authorization': f'Bearer {os.environ.get("EXECUTOR_API_KEY", "test")}',
                    'Content-Type': 'application/json'
                }
            
                test_url = f'{EXECUTOR_BOT_URL}/api/test'
            
                response = requests.post(
                    test_url,
                    json=test_signal,
                    headers=headers,
                    timeout=5
                )
            
                steps.append(f"📤 اختبار API: {response.status_code}")
            
                if response.status_code in [200, 201]:
                    steps.append("✅ اتصال API ناجح")
                    return {
                        'success': True,
                        'message': 'الاتصال بالبوت التنفيذ ناجح',
                        'steps': steps,
                        'url': EXECUTOR_BOT_URL
                    }
                else:
                    steps.append(f"⚠️ استجابة غير متوقعة: {response.text[:100]}")
                
            except Exception as e:
                steps.append(f"❌ خطأ في اختبار API: {str(e)}")
        
            return {
                'success': False,
                'message': 'فشل اختبار الاتصال',
                'steps': steps,
                'url': EXECUTOR_BOT_URL
            }
        
        except Exception as e:
            logger.error(f"❌ خطأ في اختبار الاتصال: {e}")
            return {
                'success': False,
                'message': f'خطأ في اختبار الاتصال: {str(e)}',
                'steps': [f'❌ خطأ غير متوقع: {str(e)}']
            }
    
    def _send_startup_notification(self):
        """إرسال إشعار بدء التشغيل بعد فترة قصيرة"""
        time.sleep(5)  # انتظار 5 ثوانٍ لتهيئة النظام
        
        try:
            startup_message = (
                f"🚀 بدء تشغيل نظام الإشارات\n"
                f"📊 الإصدار: 3.5.1 (مُحدث)\n"
                f"⏰ الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📍 الموقع: سوريا\n"
                f"✅ النظام يعمل بشكل طبيعي"
            )
            
            success = self.send_ntfy_notification(startup_message, "system", "low")
            
            if success:
                logger.info("✅ تم إرسال إشعار بدء التشغيل")
            else:
                logger.warning("⚠️ فشل إرسال إشعار بدء التشغيل")
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار البدء: {e}")


    
    def send_test_notification(self):
        """إرسال إشعار اختبار فوري"""
        try:
            test_message = (
                f"🔔 TEST: Notification System Working\n"
                f"📊 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📍 From: Syria\n"
                f"✅ System: Crypto Bot v3.5.1"
            )
            
            success = self.send_ntfy_notification(test_message, "test", "high")
            
            if success:
                logger.info("✅ تم إرسال إشعار الاختبار بنجاح!")
            else:
                logger.error("❌ فشل إرسال إشعار الاختبار")
            
            return success
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار الاختبار: {e}")
            return False

    def monitor_notification_status(self):
        """مراقبة وتقرير حالة الإشعارات"""
        try:
            now = datetime.now()
            report = []
            
            for coin_config in AppConfig.COINS:
                symbol = coin_config.symbol
                last_notif = self.last_notification_time.get(symbol)
                
                if last_notif:
                    hours_since = (now - last_notif).total_seconds() / 3600
                    report.append(f"{coin_config.name}: آخر إشعار قبل {hours_since:.1f} ساعة")
                else:
                    report.append(f"{coin_config.name}: لا توجد إشعارات سابقة")
            
            logger.info("📊 تقرير حالة الإشعارات:")
            for line in report:
                logger.info(f"   {line}")
            
            return report
        except Exception as e:
            logger.error(f"❌ خطأ في مراقبة حالة الإشعارات: {e}")
            return []

    def test_notification_system(self):
        """اختبار نظام الإشعارات"""
        logger.info("🧪 Starting notification system test...")
    
        # اختبار 1: اتصال NTFY
        test1 = self.test_ntfy_connection()
    
        # اختبار 2: إرسال رسالة اختبار بسيطة
        test_message = "Test notification from Crypto Bot\nTime: " + datetime.now().strftime('%H:%M:%S')
        test2 = self.send_ntfy_notification(test_message, "test", "low")
    
        # اختبار 3: إرسال رسالة مع إيموجيات
        emoji_message = "🚀 Test with emojis\n📈 Chart\n💰 Money\n⏰ Time"
        test3 = self.send_ntfy_notification(emoji_message, "test", "low")
    
        logger.info(f"Test Results: Connection={test1}, Simple={test2}, Emoji={test3}")
        return all([test1, test2, test3])

    def check_and_send_heartbeat(self):
        """إرسال نبضات النظام"""
        try:
            now = datetime.now()
        
            # التحقق من إرسال النبضات كل ساعتين
            if (self.last_heartbeat is None or 
                (now - self.last_heartbeat).total_seconds() >= self.heartbeat_interval):
            
                heartbeat_message = (
                    f"💓 System Heartbeat\n"
                    f"📊 Crypto Bot v3.5.1\n"
                    f"🔄 Running normally\n"
                    f"⏰ Last update: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"🪙 Tracking {len(AppConfig.COINS)} coins"
                )
            
                success = self.send_ntfy_notification(heartbeat_message, "heartbeat", "low")
            
                if success:
                    self.last_heartbeat = now
                    logger.info("✅ تم إرسال نبضة النظام")
            
                return success
            return True
        
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال نبضة النظام: {e}")
            return False
    
    def test_ntfy_connection(self):  # ← هذا السطر 1278
        """اختبار اتصال NTFY عند بدء التشغيل"""
        try:
            # استخدام نص إنجليزي فقط للاختبار
            test_message = "NTFY Connection Test - Crypto Bot is working!"
            headers = {
                "Title": "Connection Test",
                "Priority": "low",
                "Tags": "green_circle"
            }
        
            logger.info(f"🔍 Testing NTFY connection to: {ExternalAPIConfig.NTFY_URL}")
        
            response = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=test_message.encode('utf-8'),
                headers=headers,
                timeout=10
            )
        
            if response.status_code == 200:
                logger.info("✅ NTFY connection successful!")
            
                # اختبار ثان مع نص عربي
                arabic_test = self._send_simple_arabic_test()
                if arabic_test:
                    logger.info("✅ Arabic text encoding works!")
                else:
                    logger.warning("⚠️ Arabic text might have encoding issues")
                
                return True
            else:
                logger.warning(f"⚠️ Unexpected NTFY response: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to NTFY: {e}")
            return False

    def _send_simple_arabic_test(self):
        """اختبار بسيط للنصوص العربية"""
        try:
            test_msg = "اختبار النصوص العربية"
            response = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=test_msg.encode('utf-8'),
                headers={"Title": "Test"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
  
    def check_and_send(self, coin_signal: CoinSignal, previous_signal: Optional[CoinSignal]) -> bool:
        """التحقق وإرسال الإشعارات مع تسجيل مفصل"""
        try:
            current_percentage = coin_signal.total_percentage
            coin_symbol = coin_signal.symbol
            coin_name = coin_signal.name
    
            # تسجيل تفصيلي للتحقق
            logger.info(f"🔔 التحقق من إشعارات {coin_name} ({current_percentage:.1f}%)")
        
            # تسجيل حالة الإشارة السابقة
            if previous_signal:
                logger.info(f"   📊 الإشارة السابقة: {previous_signal.total_percentage:.1f}%")
                logger.info(f"   📈 الفرق: {current_percentage - previous_signal.total_percentage:+.1f}%")
            else:
                logger.info(f"   📊 الإشارة السابقة: غير موجودة (التحديث الأول)")
    
            logger.info(f"   ⚡ العتبات: شراء قوي({AppConfig.NOTIFICATION_THRESHOLDS['strong_buy']}) | شراء({AppConfig.NOTIFICATION_THRESHOLDS['buy']})")
            logger.info(f"   ⚡ العتبات: بيع({AppConfig.NOTIFICATION_THRESHOLDS['sell']}) | بيع قوي({AppConfig.NOTIFICATION_THRESHOLDS['strong_sell']})")
    
            # تقليل وقت الانتظار بين الإشعارات من 30 دقيقة إلى 5 دقائق للاختبار
            MIN_NOTIFICATION_INTERVAL = 300  # 5 دقائق بدلاً من 30
            
            if coin_symbol in self.last_notification_time:
                time_since_last = datetime.now() - self.last_notification_time[coin_symbol]
                minutes_since_last = int(time_since_last.total_seconds() / 60)
            
                if time_since_last.total_seconds() < MIN_NOTIFICATION_INTERVAL:
                    logger.info(f"   ⏰ آخر إشعار كان قبل {minutes_since_last} دقيقة - تخطي (5 دقائق كحد أدنى)")
                    return False
                else:
                    logger.info(f"   ⏰ آخر إشعار كان قبل {minutes_since_last} دقيقة - مؤهل للإشعار")
            else:
                logger.info(f"   ⏰ لا يوجد إشعارات سابقة لهذه العملة")
    
            message = None
            notification_type = None
            priority = "default"
    
            # إشعارات بناء على مستوى الإشارة
            if current_percentage >= AppConfig.NOTIFICATION_THRESHOLDS['strong_buy']:
                if not previous_signal or previous_signal.total_percentage < AppConfig.NOTIFICATION_THRESHOLDS['strong_buy']:
                    message = self._create_buy_message(coin_signal, "قوية")
                    notification_type = "strong_buy"
                    priority = "high"
                    logger.info(f"   🟢 مؤهل للإشعار: شراء قوي ({current_percentage:.1f}% ≥ {AppConfig.NOTIFICATION_THRESHOLDS['strong_buy']})")
    
            elif current_percentage <= AppConfig.NOTIFICATION_THRESHOLDS['strong_sell']:
                if not previous_signal or previous_signal.total_percentage > AppConfig.NOTIFICATION_THRESHOLDS['strong_sell']:
                    message = self._create_sell_message(coin_signal, "قوية")
                    notification_type = "strong_sell"
                    priority = "high"
                    logger.info(f"   🔴 مؤهل للإشعار: بيع قوي ({current_percentage:.1f}% ≤ {AppConfig.NOTIFICATION_THRESHOLDS['strong_sell']})")
    
            elif current_percentage >= AppConfig.NOTIFICATION_THRESHOLDS['buy']:
                if not previous_signal or previous_signal.total_percentage < AppConfig.NOTIFICATION_THRESHOLDS['buy']:
                    message = self._create_buy_message(coin_signal, "عادية")
                    notification_type = "buy"
                    priority = "normal"
                    logger.info(f"   🟢 مؤهل للإشعار: شراء ({current_percentage:.1f}% ≥ {AppConfig.NOTIFICATION_THRESHOLDS['buy']})")
    
            elif current_percentage <= AppConfig.NOTIFICATION_THRESHOLDS['sell']:
                if not previous_signal or previous_signal.total_percentage > AppConfig.NOTIFICATION_THRESHOLDS['sell']:
                    message = self._create_sell_message(coin_signal, "عادية")
                    notification_type = "sell"
                    priority = "normal"
                    logger.info(f"   🔴 مؤهل للإشعار: بيع ({current_percentage:.1f}% ≤ {AppConfig.NOTIFICATION_THRESHOLDS['sell']})")
    
            # إشعارات التغير الكبير
            elif previous_signal and abs(current_percentage - previous_signal.total_percentage) >= AppConfig.NOTIFICATION_THRESHOLDS['significant_change']:
                change = current_percentage - previous_signal.total_percentage
                direction = "UP" if change > 0 else "DOWN"
                logger.info(f"   🔄 مؤهل للإشعار: تغير كبير ({direction} {abs(change):.1f}% ≥ {AppConfig.NOTIFICATION_THRESHOLDS['significant_change']}%)")
                
                message = f"🔄 BIG CHANGE: {coin_name}\n"
                message += f"From {previous_signal.total_percentage:.1f}% to {current_percentage:.1f}% ({direction})\n"
                message += f"📊 Current Signal: {coin_signal.signal_type.value}\n"
                message += f"💰 Price: ${coin_signal.current_price:,.2f}\n"
                message += f"⏰ {datetime.now().strftime('%H:%M')}"
                
                notification_type = "significant_change"
                priority = "low"
    
            else:
                logger.info(f"   ⚪ غير مؤهل لأي إشعار (لا يفي بالشروط)")
                return False
    
            # إرسال الإشعار
            if message:
                logger.info(f"📤 محاولة إرسال إشعار {notification_type} لـ {coin_name}")
                success = self.send_ntfy_notification(message, notification_type, priority)
        
                if success:
                    notification_id = f"{coin_symbol}_{datetime.now().timestamp()}"
                    notification = Notification(
                        id=notification_id,
                        timestamp=datetime.now(),
                        coin_symbol=coin_symbol,
                        coin_name=coin_name,
                        message=message,
                        notification_type=notification_type,
                        signal_strength=current_percentage,
                        price=coin_signal.current_price,
                        priority=priority
                    )
            
                    self.add_notification(notification)
                    self.last_notification_time[coin_symbol] = datetime.now()
            
                    logger.info(f"✅ تم إرسال إشعار {notification_type} لـ {coin_name}")
                    return True
                else:
                    logger.warning(f"⚠️ فشل إرسال إشعار {notification_type} لـ {coin_name}")
    
            return False
    
        except Exception as e:
            logger.error(f"❌ خطأ في التحقق من الإشعارات: {e}")
            import traceback
            logger.error(f"تفاصيل الخطأ:\n{traceback.format_exc()}")
            return False
    
    def _create_buy_message(self, coin_signal: CoinSignal, strength: str) -> str:
        """إنشاء رسالة شراء مع نص إنجليزي فقط"""
        coin_name = coin_signal.name
        symbol = coin_signal.symbol

        # استخدام إيموجيات مع نص إنجليزي
        if strength == "قوية":
            strength_emoji = "🚀"
            strength_text = "STRONG"
        else:
            strength_emoji = "📈"
            strength_text = "REGULAR"

        return (
            f"{strength_emoji} {strength_text} BUY: {coin_name} ({symbol})\n"
            f"📊 Strength: {coin_signal.total_percentage:.1f}%\n"
            f"💰 Price: ${coin_signal.current_price:,.2f}\n"
            f"📈 24h Change: {coin_signal.price_change_24h:+.2f}%\n"
            f"📊 Fear/Greed: {coin_signal.fear_greed_value}\n"
            f"⏰ {datetime.now().strftime('%H:%M')}"
        )

    def _create_sell_message(self, coin_signal: CoinSignal, strength: str) -> str:
        """إنشاء رسالة بيع مع نص إنجليزي فقط"""
        coin_name = coin_signal.name
        symbol = coin_signal.symbol

        if strength == "قوية":
            strength_emoji = "⚠️"
            strength_text = "STRONG"
        else:
            strength_emoji = "📉"
            strength_text = "REGULAR"

        return (
            f"{strength_emoji} {strength_text} SELL: {coin_name} ({symbol})\n"
            f"📊 Strength: {coin_signal.total_percentage:.1f}%\n"
            f"💰 Price: ${coin_signal.current_price:,.2f}\n"
            f"📈 24h Change: {coin_signal.price_change_24h:+.2f}%\n"
            f"📊 Fear/Greed: {coin_signal.fear_greed_value}\n"
            f"⏰ {datetime.now().strftime('%H:%M')}"
        )
    
    def send_ntfy_notification(self, message: str, notification_type: str, priority: str) -> bool:
        """إرسال إشعار عبر NTFY مع معالجة ترميز UTF-8 فقط"""
        try:
            # استخدام إيموجيات فقط في Tags (لا نصوص عربية)
            tags = {
                'strong_buy': 'heavy_plus_sign,green_circle',
                'buy': 'chart_increasing,blue_circle',
                'strong_sell': 'heavy_minus_sign,red_circle',
                'sell': 'chart_decreasing,orange_circle',
                'significant_change': 'arrows_counterclockwise,yellow_circle',
                'heartbeat': 'heart,blue_circle',
                'test': 'test_tube,white_circle'
            }
    
            # استخدام عنوان إنجليزي فقط لتجنب مشاكل الترميز
            title_map = {
                'strong_buy': 'Strong Buy Signal',
                'buy': 'Buy Signal',
                'strong_sell': 'Strong Sell Signal',
                'sell': 'Sell Signal',
                'significant_change': 'Significant Change',
                'heartbeat': 'System Heartbeat',
                'test': 'Test Notification'
            }
         
            # ✅ تصحيح قيم Priority حسب توثيق NTFY
            # القيم المسموحة: 1 (min), 2 (low), 3 (default), 4 (high), 5 (max)
            priority_map = {
                'high': '4',    # أو "high"
                'normal': '3',  # أو "default" 
                'low': '2',     # أو "low"
                'default': '3'  # القيمة الافتراضية
            }
           
            priority_value = priority_map.get(priority, '3')
      
            headers = {
                "Title": title_map.get(notification_type, "Crypto Signal"),
                "Priority": priority_value,  # ✅ استخدام القيمة الصحيحة
                "Tags": tags.get(notification_type, 'loudspeaker'),
                "Content-Type": "text/plain; charset=utf-8"
            }
     
            logger.info(f"📤 Sending {notification_type} notification (Priority: {priority_value})")
            logger.info(f"   URL: {ExternalAPIConfig.NTFY_URL}")
            logger.debug(f"   Headers: {headers}")
    
            # إرسال مع ضبط ترميز UTF-8 صراحة
            response = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=15
            )
    
            logger.info(f"📥 NTFY Response: {response.status_code}")
    
            if response.status_code == 200:
                logger.info("✅ Notification sent successfully")
                return True
            else:
                logger.error(f"❌ Failed to send: {response.status_code} - {response.text}")
                return False
        
        except requests.exceptions.Timeout:
            logger.error("⏰ NTFY timeout (15 seconds)")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("🔌 Connection error - check internet")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False
        
    def _send_with_ascii_fallback(self, original_message: str, notification_type: str, priority: str) -> bool:
        """إرسال بإسقاط النصوص العربية إذا فشل الترميز"""
        try:
            # تحويل الرسالة إلى نص ASCII آمن
            safe_message = original_message
        
            # استبدال النصوص العربية بنصوص إنجليزية مع إيموجيات
            replacements = {
                "شراء قوي": "🚀 STRONG BUY",
                "شراء": "📈 BUY",
                "بيع قوي": "⚠️ STRONG SELL",
                "بيع": "📉 SELL",
                "محايد": "⚪ NEUTRAL",
                "التغير": "Change",
                "السعر": "Price",
                "القوة": "Strength",
                "الخوف والجشع": "Fear/Greed",
                "إشارة": "Signal",
                "تغير كبير": "🔄 BIG CHANGE"
            }
        
            for arabic, english in replacements.items():
                safe_message = safe_message.replace(arabic, english)
        
            # إزالة أي أحرف غير ASCII متبقية
            safe_message = ''.join(char if ord(char) < 128 else '?' for char in safe_message)
        
            headers = {
                "Title": "Crypto Signal",
                "Priority": priority,
                "Tags": "warning",
                "Content-Type": "text/plain; charset=ascii"
            }
        
            response = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=safe_message.encode('ascii', 'ignore'),
                headers=headers,
                timeout=10
            )
        
            return response.status_code == 200
        
        except Exception as e:
            logger.error(f"❌ Fallback also failed: {e}")
            return False
    
    def add_notification(self, notification: Notification):
        """إضافة إشعار إلى السجل"""
        self.notification_history.append(notification)
        
        # الحفاظ على الحد الأقصى
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history:]
    
    def get_recent_notifications(self, limit: int = 10) -> List[Notification]:
        """الحصول على الإشعارات الأخيرة"""
        return self.notification_history[-limit:] if self.notification_history else []


class SignalManager:
    """مدير الإشارات الرئيسي"""
    
    def __init__(self):
        self.signals: Dict[str, CoinSignal] = {}
        self.signal_history: List[Dict] = []
        self.last_update: Optional[datetime] = None
        self.fear_greed_index = 50
        self.fear_greed_score = 0.5
        
        self.data_fetcher = BinanceDataFetcher()
        self.fgi_fetcher = FearGreedIndexFetcher()
        self.notification_manager = NotificationManager()
        self.calculator = IndicatorsCalculator()
        
        self.max_history = 100
        self.update_lock = threading.Lock()
    

    def update_all_signals(self) -> bool:
        """تحديث جميع الإشارات مع تحسين الإشعارات"""
        with self.update_lock:
            logger.info("=" * 60)
            logger.info(f"🚀 بدء تحديث الإشارات - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"📊 العملات: {len([c for c in AppConfig.COINS if c.enabled])} مفعلة")
            logger.info("=" * 60)
        
            try:
                # تحديث مؤشر الخوف والجشع
                logger.info("🔄 تحديث مؤشر الخوف والجشع...")
                self._update_fear_greed_index()
            
                success_count = 0
                failed_coins = []
                notifications_sent = 0
            
                for coin_config in AppConfig.COINS:
                    if not coin_config.enabled:
                        continue
                
                    logger.info(f"🔍 معالجة {coin_config.name} ({coin_config.symbol})...")
                    previous_signal = self.signals.get(coin_config.symbol)
                
                    try:
                        coin_signal = self._process_coin_signal(coin_config)
                    
                        if coin_signal.is_valid:
                            logger.info(f"   ✅ إشارة صالحة: {coin_signal.total_percentage:.1f}%")
                        
                            # حفظ الإشارة أولاً
                            self.signals[coin_config.symbol] = coin_signal
                            success_count += 1
                        
                            # التحقق من الإشعارات بعد الحفظ
                            notification_sent = self.notification_manager.check_and_send(coin_signal, previous_signal)
                            if notification_sent:
                                notifications_sent += 1
                                logger.info(f"   📢 تم إرسال إشعار لـ {coin_config.name}")
                            else:
                                logger.info(f"   📭 لم يتم إرسال إشعار لـ {coin_config.name} (غير مؤهل)")
                        
                        else:
                            error_msg = f"{coin_config.name}: {coin_signal.error_message}"
                            logger.warning(f"   ❌ {error_msg}")
                            failed_coins.append(error_msg)
                        
                    except Exception as e:
                        error_msg = f"خطأ في معالجة {coin_config.name}: {str(e)}"
                        logger.error(f"   ❌ {error_msg}")
                        failed_coins.append(error_msg)
                        continue
            
                # تحديث وقت التحديث الأخير
                self.last_update = datetime.now()
            
                # حفظ في السجل
                self._save_to_history()
            
                logger.info("=" * 60)
                logger.info(f"✅ تم تحديث {success_count}/{len([c for c in AppConfig.COINS if c.enabled])} إشارات بنجاح")
                logger.info(f"📤 تم إرسال {notifications_sent} إشعارات خلال هذه الدورة")
            
                if failed_coins:
                    logger.warning(f"⚠️ العملات التي فشلت: {', '.join(failed_coins)}")
            
                return success_count > 0
            
            except Exception as e:
                logger.error(f"❌ خطأ في تحديث الإشارات: {e}")
                import traceback
                logger.error(f"تفاصيل الخطأ:\n{traceback.format_exc()}")
                return False
    
    def _update_fear_greed_index(self):
        """تحديث مؤشر الخوف والجشع"""
        try:
            self.fear_greed_score, self.fear_greed_index, _ = self.fgi_fetcher.get_index()
            logger.info(f"مؤشر الخوف والجشع: {self.fear_greed_index} (النتيجة: {self.fear_greed_score:.2f})")
        except Exception as e:
            logger.error(f"خطأ في تحديث مؤشر الخوف والجشع: {e}")

    def send_strong_signals_to_executor(self):
        """إرسال الإشارات القوية تلقائياً إلى بوت التنفيذ"""
        try:
            strong_signals = []
            
            for symbol, signal in self.signals.items():
                if signal.is_valid and signal.total_percentage >= 63:  # إشارات شراء قوية
                    action = 'BUY' if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY] else 'SELL'
                    
                    signal_data = {
                        'symbol': symbol.replace('/', ''),  # تحويل BTC/USDT إلى BTCUSDT
                        'action': action,
                        'confidence_score': signal.total_percentage,
                        'reason': f'{signal.signal_type.value} - {signal.signal_strength}',
                        'coin_name': signal.name,
                        'signal_strength': signal.signal_strength
                    }
                    
                    strong_signals.append(signal_data)
            
            if strong_signals:
                EXECUTOR_BOT_URL = os.environ.get('EXECUTOR_BOT_URL')
                if EXECUTOR_BOT_URL:
                    for signal in strong_signals:
                        try:
                            headers = {
                                'Authorization': f'Bearer {os.environ.get("EXECUTOR_API_KEY", "default_key_here")}',
                                'Content-Type': 'application/json'
                            }
                            
                            response = requests.post(
                                f'{EXECUTOR_BOT_URL}/api/trade/signal',
                                json={'signal': signal},
                                headers=headers,
                                timeout=10
                            )
                            
                            if response.status_code == 200:
                                logger.info(f"✅ أرسلت إشارة {signal['symbol']} ({signal['confidence_score']:.1f}%) إلى بوت التنفيذ")
                        except Exception as e:
                            logger.error(f"❌ خطأ في إرسال إشارة {signal['symbol']}: {e}")
            
            return len(strong_signals)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الإشارات القوية: {e}")
            return 0
    
    def _process_coin_signal(self, coin_config: CoinConfig) -> CoinSignal:
        """معالجة إشارة عملة واحدة (باستخدام 15m كإطار زمني أساسي)"""
        try:
            # جلب البيانات من الإطار الزمني 15m (الإطار الزمني الأساسي الجديد)
            df_15m = self.data_fetcher.get_ohlcv(coin_config.symbol, timeframe='15m', limit=500)
            if df_15m is None or df_15m.empty:
                return CoinSignal(
                    symbol=coin_config.symbol,
                    name=coin_config.name,
                    current_price=0,
                    price_change_24h=0,
                    high_24h=0,
                    low_24h=0,
                    volume_24h=0,
                    total_percentage=50,
                    signal_type=SignalType.NEUTRAL_HIGH,
                    signal_strength="غير معروف",
                    signal_color="secondary",
                    indicator_scores={},
                    last_updated=datetime.now(),
                    fear_greed_value=self.fear_greed_index,
                    is_valid=False,
                    error_message="فشل جلب بيانات OHLCV للإطار الزمني 15m"
                )
            
            # جلب بيانات متعددة الإطار الزمني
            multiple_tf_data = self.data_fetcher.get_multiple_timeframes(coin_config.symbol)
            
            # جلب الإحصائيات
            stats_24h = self.data_fetcher.get_24h_stats(coin_config.symbol)
            current_price = self.data_fetcher.get_current_price(coin_config.symbol)
            
            # حساب المؤشرات المحسنة (باستخدام df_15m كبيانات أساسية)
            trend_score = self.calculator.calculate_trend_strength(df_15m, multiple_tf_data)
            momentum_score = self.calculator.calculate_momentum(df_15m)
            volume_score = self.calculator.calculate_volume_analysis(df_15m, stats_24h['change'])
            volatility_score = self.calculator.calculate_volatility(df_15m)
            price_structure_score = self.calculator.calculate_price_structure(df_15m)
            support_resistance_score = self.calculator.calculate_support_resistance(df_15m)
            
            # جمع المؤشرات
            indicator_scores = {
                IndicatorType.TREND_STRENGTH.value: trend_score,
                IndicatorType.MOMENTUM.value: momentum_score,
                IndicatorType.VOLUME_ANALYSIS.value: volume_score,
                IndicatorType.VOLATILITY.value: volatility_score,
                IndicatorType.MARKET_SENTIMENT.value: self.fear_greed_score,
                IndicatorType.PRICE_STRUCTURE.value: price_structure_score,
                IndicatorType.SUPPORT_RESISTANCE.value: support_resistance_score
            }
            
            # حساب الإشارة المرجحة
            signal_result = SignalProcessor.calculate_weighted_score(indicator_scores)
            
            # حساب تغير السعر منذ التحديث الأخير
            price_change_since_last = None
            previous_signal = self.signals.get(coin_config.symbol)
            if previous_signal and previous_signal.current_price > 0 and current_price > 0:
                price_change_since_last = ((current_price - previous_signal.current_price) / 
                                          previous_signal.current_price) * 100
            
            # إنشاء إشارة العملة
            coin_signal = CoinSignal(
                symbol=coin_config.symbol,
                name=coin_config.name,
                current_price=current_price,
                price_change_24h=stats_24h['change'],
                high_24h=stats_24h['high'],
                low_24h=stats_24h['low'],
                volume_24h=stats_24h['volume'],
                total_percentage=signal_result['total_percentage'],
                signal_type=signal_result['signal_type'],
                signal_strength=signal_result['signal_strength'],
                signal_color=signal_result['signal_color'],
                indicator_scores=signal_result['weighted_scores'],
                last_updated=datetime.now(),
                fear_greed_value=self.fear_greed_index,
                price_change_since_last=price_change_since_last,
                is_valid=True
            )
            
            return coin_signal
            
        except Exception as e:
            logger.error(f"خطأ في معالجة {coin_config.name}: {e}")
            return CoinSignal(
                symbol=coin_config.symbol,
                name=coin_config.name,
                current_price=0,
                price_change_24h=0,
                high_24h=0,
                low_24h=0,
                volume_24h=0,
                total_percentage=50,
                signal_type=SignalType.NEUTRAL_HIGH,
                signal_strength="خطأ",
                signal_color="secondary",
                indicator_scores={},
                last_updated=datetime.now(),
                fear_greed_value=self.fear_greed_index,
                is_valid=False,
                error_message=str(e)
            )
    
    def _save_to_history(self):
        """حفظ البيانات في السجل"""
        history_entry = {
            'timestamp': datetime.now(),
            'signals': {symbol: signal.total_percentage for symbol, signal in self.signals.items()},
            'fear_greed_index': self.fear_greed_index
        }
        
        self.signal_history.append(history_entry)
        
        # الحفاظ على الحد الأقصى
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]
    
    def _cleanup_old_data(self):
        """تنظيف البيانات القديمة"""
        # تنظيف الإشارات القديمة (أقدم من 3 ساعات)
        cutoff_time = datetime.now() - timedelta(hours=3)
        self.signals = {
            symbol: signal for symbol, signal in self.signals.items()
            if signal.last_updated > cutoff_time
        }
    
    def get_coins_data(self) -> List[Dict]:
        """الحصول على بيانات العملات للتنسيق"""
        coins_data = []
        
        for coin_config in AppConfig.COINS:
            if not coin_config.enabled:
                continue
            
            symbol = coin_config.symbol
            if symbol in self.signals:
                signal = self.signals[symbol]
                coins_data.append(self._format_coin_data(signal))
            else:
                # بيانات افتراضية
                coins_data.append(self._get_default_coin_data(coin_config))
        
        # ترتيب حسب قوة الإشارة
        coins_data.sort(key=lambda x: x['total_percentage'], reverse=True)
        
        return coins_data
    
    def _format_coin_data(self, signal: CoinSignal) -> Dict:
        """تنسيق بيانات العملة للعرض"""
        indicators_list = []
        
        for ind_name, ind_data in signal.indicator_scores.items():
            indicators_list.append({
                'name': ind_name,
                'display_name': AppConfig.INDICATOR_DISPLAY_NAMES.get(ind_name, ind_name),
                'description': AppConfig.INDICATOR_DESCRIPTIONS.get(ind_name, ''),
                'raw_score': ind_data.raw_score * 100,
                'weighted_score': ind_data.weighted_score * 100,
                'percentage': ind_data.percentage,
                'color': ind_data.color,
                'weight': ind_data.weight * 100
            })
        
        return {
            'symbol': signal.symbol,
            'name': signal.name,
            'current_price': signal.current_price,
            'formatted_price': self._format_number(signal.current_price),
            'price_change_24h': signal.price_change_24h,
            'formatted_24h_change': self._format_percentage(signal.price_change_24h),
            'high_24h': signal.high_24h,
            'low_24h': signal.low_24h,
            'volume_24h': signal.volume_24h,
            'formatted_volume_24h': self._format_number(signal.volume_24h),
            'total_percentage': signal.total_percentage,
            'signal_type': signal.signal_type.value,
            'signal_strength': signal.signal_strength,
            'signal_color': signal.signal_color,
            'indicators': indicators_list,
            'last_updated': signal.last_updated,
            'last_updated_str': self._format_time_delta(signal.last_updated),
            'fear_greed_value': signal.fear_greed_value,
            'price_change_since_last': signal.price_change_since_last,
            'formatted_price_change': self._format_percentage(signal.price_change_since_last) if signal.price_change_since_last else '0.00%',
            'is_valid': signal.is_valid
        }
    
    def _get_default_coin_data(self, coin_config: CoinConfig) -> Dict:
        """الحصول على بيانات افتراضية للعملة"""
        return {
            'symbol': coin_config.symbol,
            'name': coin_config.name,
            'current_price': 0,
            'formatted_price': '0',
            'price_change_24h': 0,
            'formatted_24h_change': '0.00%',
            'high_24h': 0,
            'low_24h': 0,
            'volume_24h': 0,
            'formatted_volume_24h': '0',
            'total_percentage': 50,
            'signal_type': SignalType.NEUTRAL_HIGH.value,
            'signal_strength': 'غير متوفر',
            'signal_color': 'secondary',
            'indicators': [],
            'last_updated': None,
            'last_updated_str': 'غير معروف',
            'fear_greed_value': self.fear_greed_index,
            'price_change_since_last': 0,
            'formatted_price_change': '0.00%',
            'is_valid': False
        }
    
    @staticmethod
    def _format_number(value: float) -> str:
        """تنسيق الأرقام"""
        try:
            if value is None or np.isnan(value):
                return "0"
            
            value = float(value)
            
            if abs(value) >= 1_000_000_000:
                return f"{value/1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                return f"{value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.2f}K"
            elif abs(value) >= 1:
                return f"{value:,.2f}"
            elif abs(value) >= 0.01:
                return f"{value:.4f}"
            else:
                return f"{value:.6f}"
        except:
            return "0"
    
    @staticmethod
    def _format_percentage(value: float) -> str:
        """تنسيق النسب المئوية"""
        try:
            if value is None or np.isnan(value):
                return "0.00%"
            
            value = float(value)
            prefix = "+" if value > 0 else ""
            return f"{prefix}{value:.2f}%"
        except:
            return "0.00%"
    
    @staticmethod
    def _format_time_delta(dt: datetime) -> str:
        """تنسيق الفرق الزمني"""
        if not dt:
            return "غير معروف"
        
        now = datetime.now()
        delta = now - dt
        
        if delta.days > 0:
            return f"قبل {delta.days} يوم"
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            return f"قبل {hours} ساعة و{minutes} دقيقة"
        elif delta.seconds >= 60:
            minutes = delta.seconds // 60
            seconds = delta.seconds % 60
            return f"قبل {minutes} دقيقة و{seconds} ثانية"
        else:
            return f"قبل {delta.seconds} ثانية"
    
    def get_stats(self) -> Dict:
        """الحصول على الإحصائيات"""
        coins_data = self.get_coins_data()
        valid_signals = [c for c in coins_data if c['is_valid']]
        
        if not valid_signals:
            return {
                'total_coins': len(AppConfig.COINS),
                'updated_coins': 0,
                'avg_signal': 50,
                'strong_buy_signals': 0,
                'buy_signals': 0,
                'neutral_signals': len(AppConfig.COINS),
                'sell_signals': 0,
                'strong_sell_signals': 0,
                'last_update': self.last_update,
                'last_update_str': self._format_time_delta(self.last_update) if self.last_update else 'غير معروف',
                'total_notifications': len(self.notification_manager.notification_history),
                'fear_greed_index': self.fear_greed_index,
                'system_status': 'warning'
            }
        
        signal_percentages = [c['total_percentage'] for c in valid_signals]
        
        # عدّ الإشارات حسب النوع
        signal_counts = {stype: 0 for stype in SignalType}
        
        for coin in valid_signals:
            for signal_type, threshold in AppConfig.SIGNAL_THRESHOLDS.items():
                if coin['total_percentage'] >= threshold:
                    signal_counts[signal_type] += 1
                    break
        
        return {
            'total_coins': len(AppConfig.COINS),
            'updated_coins': len(valid_signals),
            'avg_signal': np.mean(signal_percentages) if signal_percentages else 50,
            'strong_buy_signals': signal_counts[SignalType.STRONG_BUY],
            'buy_signals': signal_counts[SignalType.BUY],
            'neutral_signals': signal_counts[SignalType.NEUTRAL_HIGH] + signal_counts[SignalType.NEUTRAL_LOW],
            'sell_signals': signal_counts[SignalType.SELL],
            'strong_sell_signals': signal_counts[SignalType.STRONG_SELL],
            'last_update': self.last_update,
            'last_update_str': self._format_time_delta(self.last_update) if self.last_update else 'غير معروف',
            'total_notifications': len(self.notification_manager.notification_history),
            'fear_greed_index': self.fear_greed_index,
            'system_status': 'healthy' if len(valid_signals) >= len(AppConfig.COINS) * 0.7 else 'warning'
        }


# ======================
# تهيئة التطبيق
# ======================

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-signal-secret-2024')

signal_manager = SignalManager()
start_time = time.time()


# ======================
# Routes
# ======================

# ======================
# Routes لإرسال الإشارات إلى البوت التنفيذي
# ======================

@app.route('/test-executor')
def test_executor_page():
    """صفحة اختبار البوت التنفيذ"""
    return render_template('test_executor.html')


@app.route('/api/check_executor_connection', methods=['GET'])
def check_executor_connection():
    """فحص اتصال البوت التنفيذ"""
    try:
        result = signal_manager.notification_manager.test_executor_connection()
        
        # تسجيل النتيجة في السجل
        logger.info("=" * 50)
        logger.info("🔍 نتيجة فحص اتصال البوت التنفيذ:")
        for step in result.get('steps', []):
            logger.info(f"   {step}")
        logger.info(f"📊 النتيجة النهائية: {'✅ ناجح' if result.get('success') else '❌ فشل'}")
        logger.info("=" * 50)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ خطأ في فحص الاتصال: {e}")
        return jsonify({
            'success': False,
            'message': f'خطأ في فحص الاتصال: {str(e)}'
        }), 500


@app.route('/api/test_executor_notification', methods=['POST'])
def test_executor_notification():
    """إرسال إشعار تجريبي إلى البوت التنفيذ"""
    try:
        logger.info("🧪 بدء اختبار إشعار البوت التنفيذ...")
        
        # عنوان بوت التنفيذ
        EXECUTOR_BOT_URL = os.environ.get('EXECUTOR_BOT_URL')
        
        if not EXECUTOR_BOT_URL:
            logger.error("❌ لم يتم تعيين EXECUTOR_BOT_URL في متغيرات البيئة")
            return jsonify({
                'success': False,
                'message': 'لم يتم تعيين عنوان البوت التنفيذ'
            })
        
        # إعداد بيانات الاختبار
        test_signal = {
            'signal': {
                'symbol': 'BTCUSDT',
                'action': 'BUY',
                'confidence_score': 85.5,
                'reason': 'إختبار نظام الإشعارات - إشارة تجريبية',
                'coin_name': 'Bitcoin',
                'timeframe': '15m',
                'analysis': 'شراء قوي',
                'signal_strength': 'قوية جداً',
                'test_mode': True,
                'test_timestamp': datetime.now().isoformat()
            }
        }
        
        # تسجيل تفاصيل الاختبار
        logger.info(f"📤 إرسال إشعار اختبار إلى: {EXECUTOR_BOT_URL}")
        logger.info(f"📊 بيانات الإشعار: {json.dumps(test_signal, ensure_ascii=False)}")
        
        # إعداد الهيدرات
        headers = {
            'Authorization': f'Bearer {os.environ.get("EXECUTOR_API_KEY", "test_key_123")}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"🔑 استخدام مفتاح API: {os.environ.get('EXECUTOR_API_KEY', 'مفتاح افتراضي')[:10]}...")
        
        # إرسال الطلب مع مهلة زمنية
        start_time = time.time()
        response = requests.post(
            f'{EXECUTOR_BOT_URL}/api/trade/signal',
            json=test_signal,
            headers=headers,
            timeout=15
        )
        request_time = time.time() - start_time
        
        # تحليل الاستجابة
        logger.info(f"📥 استجابة البوت التنفيذ:")
        logger.info(f"   ⏱️  وقت الاستجابة: {request_time:.2f} ثانية")
        logger.info(f"   📊 كود الحالة: {response.status_code}")
        logger.info(f"   📝 نص الاستجابة: {response.text[:200]}...")
        
        # تسجيل النتيجة
        if response.status_code == 200:
            logger.info("✅ تم إرسال الإشعار التجريبي بنجاح إلى البوت التنفيذ")
            
            try:
                response_data = response.json()
                logger.info(f"   📈 بيانات الاستجابة: {json.dumps(response_data, ensure_ascii=False)}")
                
                return jsonify({
                    'success': True,
                    'message': 'تم إرسال الإشعار التجريبي بنجاح',
                    'response_time': f"{request_time:.2f} ثانية",
                    'status_code': response.status_code,
                    'executor_response': response_data,
                    'test_data_sent': test_signal
                })
            except json.JSONDecodeError:
                logger.warning("⚠️ لم يتمكن البوت التنفيذ من إرجاع JSON")
                return jsonify({
                    'success': True,
                    'message': 'تم إرسال الإشعار ولكن الاستجابة ليست JSON صحيح',
                    'response_time': f"{request_time:.2f} ثانية",
                    'status_code': response.status_code,
                    'raw_response': response.text[:500]
                })
                
        elif response.status_code == 401:
            logger.error("❌ فشل المصادقة (401 Unauthorized)")
            return jsonify({
                'success': False,
                'message': 'فشل المصادقة - تأكد من صحة مفتاح API',
                'status_code': 401,
                'details': 'EXECUTOR_API_KEY غير صحيح أو منتهي الصلاحية'
            }), 401
            
        elif response.status_code == 404:
            logger.error("❌ العنوان غير موجود (404 Not Found)")
            return jsonify({
                'success': False,
                'message': 'العنوان غير موجود - تأكد من صحة EXECUTOR_BOT_URL',
                'status_code': 404,
                'details': f'تم الوصول إلى: {EXECUTOR_BOT_URL}/api/trade/signal'
            }), 404
            
        elif response.status_code == 500:
            logger.error("❌ خطأ داخلي في البوت التنفيذ (500 Internal Error)")
            return jsonify({
                'success': False,
                'message': 'خطأ داخلي في البوت التنفيذ',
                'status_code': 500,
                'details': response.text[:500]
            }), 500
            
        else:
            logger.error(f"❌ استجابة غير متوقعة: {response.status_code}")
            return jsonify({
                'success': False,
                'message': f'استجابة غير متوقعة من البوت التنفيذ: {response.status_code}',
                'status_code': response.status_code,
                'details': response.text[:500]
            }), 400
            
    except requests.exceptions.Timeout:
        logger.error("⏰ انتهت المهلة - البوت التنفيذ لم يستجب خلال 15 ثانية")
        return jsonify({
            'success': False,
            'message': 'انتهت المهلة - البوت التنفيذ لم يستجب',
            'details': 'تحقق من اتصال الشبكة وتوافر البوت التنفيذ'
        }), 408
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"🔌 خطأ في الاتصال: {e}")
        return jsonify({
            'success': False,
            'message': 'فشل الاتصال بالبوت التنفيذ',
            'details': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع: {str(e)}")
        import traceback
        logger.error(f"📋 تفاصيل الخطأ:\n{traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'message': f'خطأ غير متوقع: {str(e)}',
            'details': traceback.format_exc()[:1000]
        }), 500

@app.route('/api/send_signal_to_executor', methods=['POST'])
def send_signal_to_executor():
    """إرسال إشارة إلى بوت التنفيذ"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'لا توجد بيانات'})
        
        # عنوان بوت التنفيذ (افتراضي على Render)
        EXECUTOR_BOT_URL = os.environ.get('EXECUTOR_BOT_URL', 'http://localhost:10000')
        
        # تحويل تنسيق الإشارة
        signal_to_send = {
            'signal': {
                'symbol': data.get('symbol'),
                'action': data.get('action'),  # 'BUY' أو 'SELL'
                'confidence_score': data.get('confidence_score', 50),
                'reason': data.get('reason', 'إشارة من محلل الإشارات'),
                'coin': data.get('coin_name'),
                'timeframe': '15m',
                'analysis': data.get('signal_strength', 'متوسطة')
            }
        }
        
        # إرسال الإشارة إلى بوت التنفيذ
        headers = {
            'Authorization': f'Bearer {os.environ.get("EXECUTOR_API_KEY", "default_key_here")}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{EXECUTOR_BOT_URL}/api/trade/signal',
            json=signal_to_send,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"✅ تم إرسال الإشارة إلى بوت التنفيذ: {data.get('symbol')}")
            return jsonify({'success': True, 'message': 'تم إرسال الإشارة'})
        else:
            logger.error(f"❌ فشل إرسال الإشارة: {response.status_code}")
            return jsonify({'success': False, 'message': f'فشل الإرسال: {response.status_code}'})
            
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال الإشارة: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/send_heartbeat_to_executor', methods=['POST'])
def send_heartbeat_to_executor():
    """إرسال نبضة حياة إلى بوت التنفيذ"""
    try:
        EXECUTOR_BOT_URL = os.environ.get('EXECUTOR_BOT_URL', 'http://localhost:10000')
        
        heartbeat_data = {
            'heartbeat': True,
            'source': 'signal_analyzer_bot',
            'syria_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_stats': {
                'total_coins': len(AppConfig.COINS),
                'updated_coins': len(signal_manager.signals),
                'avg_signal': signal_manager.get_stats().get('avg_signal', 50),
                'last_update': signal_manager.last_update.isoformat() if signal_manager.last_update else None
            }
        }
        
        headers = {
            'Authorization': f'Bearer {os.environ.get("EXECUTOR_API_KEY", "default_key_here")}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{EXECUTOR_BOT_URL}/api/heartbeat',
            json=heartbeat_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("💓 تم إرسال نبضة حياة إلى بوت التنفيذ")
            return jsonify({'success': True})
        else:
            logger.warning(f"⚠️ فشل إرسال نبضة الحياة: {response.status_code}")
            return jsonify({'success': False})
            
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال نبضة الحياة: {e}")
        return jsonify({'success': False})

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    coins_data = signal_manager.get_coins_data()
    stats = signal_manager.get_stats()
    
    # الإشعارات الأخيرة
    recent_notifications = signal_manager.notification_manager.get_recent_notifications(10)
    
    # وقت التحديث التالي
    next_update_time = None
    if signal_manager.last_update:
        next_update_time = signal_manager.last_update + timedelta(seconds=AppConfig.UPDATE_INTERVAL)
    
    return render_template(
        'index.html',
        coins=coins_data,
        stats=stats,
        next_update_time=next_update_time,
        notifications=recent_notifications,
        indicator_weights=AppConfig.INDICATOR_WEIGHTS,
        get_indicator_color=lambda key: AppConfig.INDICATOR_COLORS.get(key, '#2E86AB'),
        get_indicator_display_name=lambda key: AppConfig.INDICATOR_DISPLAY_NAMES.get(key, key),
        format_number=signal_manager._format_number,
        format_percentage=signal_manager._format_percentage
    )


@app.route('/api/signals')
def api_signals():
    """API للإشارات"""
    coins_data = signal_manager.get_coins_data()
    return jsonify({
        'status': 'success',
        'data': coins_data,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/update', methods=['POST'])
def manual_update():
    """تحديث يدوي"""
    try:
        success = signal_manager.update_all_signals()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'تم تحديث الإشارات بنجاح',
                'timestamp': datetime.now().isoformat(),
                'updated_coins': len(signal_manager.signals)
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'تم تحديث بعض الإشارات فقط',
                'timestamp': datetime.now().isoformat(),
                'updated_coins': len(signal_manager.signals)
            }), 200
    except Exception as e:
        logger.error(f"خطأ في التحديث اليدوي: {e}")
        return jsonify({
            'status': 'error',
            'message': f'فشل التحديث: {str(e)}'
        }), 500


@app.route('/api/health')
def health_check():
    """فحص صحة النظام"""
    now = datetime.now()
    last_update = signal_manager.last_update
    
    status = 'healthy'
    if last_update:
        time_since_update = (now - last_update).total_seconds()
        if time_since_update > 600:  # أكثر من 10 دقائق
            status = 'warning'
        elif time_since_update > 1800:  # أكثر من 30 دقيقة
            status = 'unhealthy'
    else:
        status = 'unknown'
    
    return jsonify({
        'status': status,
        'last_update': last_update.isoformat() if last_update else None,
        'time_since_update': (now - last_update).total_seconds() if last_update else None,
        'coins_available': len(signal_manager.signals),
        'coins_total': len(AppConfig.COINS),
        'uptime': time.time() - start_time,
        'version': '3.5.0',
        'fear_greed_index': signal_manager.fear_greed_index,
        'notification_count': len(signal_manager.notification_manager.notification_history)
    })


@app.route('/api/notifications')
def get_notifications():
    """الحصول على الإشعارات"""
    limit = request.args.get('limit', 10, type=int)
    notifications = signal_manager.notification_manager.get_recent_notifications(limit)
    
    formatted_notifications = []
    for notification in notifications:
        formatted_notifications.append({
            'id': notification.id,
            'timestamp': notification.timestamp.isoformat(),
            'coin_symbol': notification.coin_symbol,
            'coin_name': notification.coin_name,
            'message': notification.message,
            'type': notification.notification_type,
            'priority': notification.priority
        })
    
    return jsonify({
        'notifications': formatted_notifications,
        'total': len(signal_manager.notification_manager.notification_history)
    })


@app.route('/api/coins')
def get_coins():
    """الحصول على قائمة العملات"""
    coins_list = []
    for coin in AppConfig.COINS:
        coins_list.append({
            'symbol': coin.symbol,
            'name': coin.name,
            'base_asset': coin.base_asset,
            'quote_asset': coin.quote_asset,
            'enabled': coin.enabled
        })
    
    return jsonify({'coins': coins_list})


@app.route('/api/indicators')
def get_indicators():
    """الحصول على معلومات المؤشرات"""
    indicators_info = {}
    for key in AppConfig.INDICATOR_WEIGHTS.keys():
        indicators_info[key] = {
            'display_name': AppConfig.INDICATOR_DISPLAY_NAMES.get(key, key),
            'description': AppConfig.INDICATOR_DESCRIPTIONS.get(key, ''),
            'color': AppConfig.INDICATOR_COLORS.get(key, '#2E86AB'),
            'weight': AppConfig.INDICATOR_WEIGHTS[key] * 100,
            'weight_raw': AppConfig.INDICATOR_WEIGHTS[key]
        }
    return jsonify({'indicators': indicators_info})


@app.route('/api/history')
def get_history():
    """الحصول على السجل التاريخي"""
    limit = request.args.get('limit', 50, type=int)
    history = signal_manager.signal_history[-limit:] if signal_manager.signal_history else []
    
    formatted_history = []
    for entry in history:
        formatted_history.append({
            'timestamp': entry['timestamp'].isoformat(),
            'signals': entry['signals'],
            'fear_greed_index': entry.get('fear_greed_index', 50)
        })
    
    return jsonify({
        'history': formatted_history,
        'total': len(signal_manager.signal_history)
    })

def background_monitor():
    """مراقبة النظام في الخلفية وإرسال تقارير دورية"""
    monitor_interval = 3600  # كل ساعة
    
    # الانتظار 30 ثانية قبل البدء
    time.sleep(30)
    
    logger.info("👁️  بدأ خيط مراقبة النظام")
    
    while True:
        try:
            # الانتظار قبل كل دورة مراقبة
            logger.info(f"⏳ مراقبة النظام: الانتظار {monitor_interval//60} دقائق للدورة القادمة")
            time.sleep(monitor_interval)
            
            now = datetime.now()
            logger.info(f"📊 بدء دورة مراقبة النظام الساعة {now.strftime('%H:%M:%S')}")
            
            # مراقبة حالة الإشعارات
            try:
                report = signal_manager.notification_manager.monitor_notification_status()
                
                # إعداد تقرير المراقبة
                active_coins = len([c for c in AppConfig.COINS if c.enabled])
                updated_coins = len(signal_manager.signals)
                total_notifications = len(signal_manager.notification_manager.notification_history)
                
                report_message = (
                    f"📊 تقرير مراقبة النظام\n"
                    f"⏰ الوقت: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"💰 العملات النشطة: {active_coins}\n"
                    f"🔄 العملات المحدثة: {updated_coins}\n"
                    f"📈 نسبة التحديث: {(updated_coins/active_coins*100):.1f}%\n"
                    f"🔔 إجمالي الإشعارات: {total_notifications}\n"
                )
                
                # إضافة حالة آخر تحديث
                if signal_manager.last_update:
                    hours_since = (now - signal_manager.last_update).total_seconds() / 3600
                    report_message += f"⏰ آخر تحديث: قبل {hours_since:.1f} ساعة\n"
                else:
                    report_message += f"⏰ آخر تحديث: غير متوفر\n"
                
                # إرسال تقرير المراقبة
                success = signal_manager.notification_manager.send_ntfy_notification(
                    report_message, 
                    "heartbeat", 
                    "low"
                )
                
                if success:
                    logger.info("✅ تم إرسال تقرير المراقبة الدورية")
                else:
                    logger.warning("⚠️ فشل إرسال تقرير المراقبة")
                    
            except Exception as e:
                logger.error(f"❌ خطأ في مراقبة الإشعارات: {e}")
            
        except Exception as e:
            logger.error(f"❌ خطأ في مراقبة الخلفية: {e}")
            time.sleep(300)  # انتظار 5 دقائق ثم المحاولة مرة أخرى

# قم بتعديل دالة background_updater لتسجيل المزيد من التفاصيل:

def background_updater():
    """تحديث تلقائي مع إرسال الإشارات"""
    logger.info("🔧 بدأ التحديث التلقائي مع إرسال الإشارات")
    
    while True:
        try:
            print(f"\n{'='*50}")
            print(f"🔄 التحديث التلقائي في: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*50}")
            
            # تحديث الإشارات
            signal_manager.update_all_signals()
            
            # إرسال الإشارات القوية إلى بوت التنفيذ
            logger.info("📤 التحقق من الإشارات القوية للإرسال...")
            strong_signals_sent = signal_manager.send_strong_signals_to_executor()
            
            if strong_signals_sent > 0:
                logger.info(f"✅ تم إرسال {strong_signals_sent} إشارة قوية إلى بوت التنفيذ")
            else:
                logger.info("📭 لا توجد إشارات قوية للإرسال حالياً")
            
            # تسجيل حالة الإرسال
            active_signals = len([s for s in signal_manager.signals.values() 
                                if s.is_valid and s.total_percentage >= 63])
            logger.info(f"📊 الإشارات القوية المتاحة: {active_signals}")
            
            # إرسال نبضة حياة
            try:
                EXECUTOR_BOT_URL = os.environ.get('EXECUTOR_BOT_URL')
                if EXECUTOR_BOT_URL:
                    logger.info(f"💓 إرسال نبضة حياة إلى: {EXECUTOR_BOT_URL}")
                    
                    heartbeat_data = {
                        'heartbeat': True,
                        'source': 'signal_analyzer_bot',
                        'syria_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'system_stats': signal_manager.get_stats()
                    }
                    
                    headers = {
                        'Authorization': f'Bearer {os.environ.get("EXECUTOR_API_KEY", "default_key_here")}',
                        'Content-Type': 'application/json'
                    }
                    
                    response = requests.post(
                        f'{EXECUTOR_BOT_URL}/api/heartbeat',
                        json=heartbeat_data,
                        headers=headers,
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        logger.info("✅ تم إرسال نبضة الحياة بنجاح")
                    else:
                        logger.warning(f"⚠️ فشل إرسال نبضة الحياة: {response.status_code}")
                        
            except Exception as e:
                logger.error(f"❌ خطأ في إرسال نبضة الحياة: {e}")
            
            # انتظار
            wait_time = 120  # دقيقتين
            logger.info(f"⏳ الانتظار {wait_time} ثانية للتحديث التالي...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            logger.info("🛑 توقف التحديث التلقائي بواسطة المستخدم")
            break
        except Exception as e:
            logger.error(f"❌ خطأ في التحديث التلقائي: {e}")
            import traceback
            logger.error(f"📋 التفاصيل:\n{traceback.format_exc()}")
            time.sleep(60)
# ======================
# تشغيل التطبيق
# ======================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 بدء تشغيل Crypto Signal Analyzer - الإصدار 3.5.1")
    print("📊 الإطار الزمني الأساسي: 15 دقيقة (15M)")
    print("=" * 60)
    print(f"📢 إعدادات الإشعارات:")
    print(f"   Topic: {ExternalAPIConfig.NTFY_TOPIC}")
    print(f"   URL: {ExternalAPIConfig.NTFY_URL}")
    print(f"   رابط الاشتراك: https://ntfy.sh/{ExternalAPIConfig.NTFY_TOPIC}")
    print("=" * 60)
    print(f"📊 مراقبة العملات: {[coin.name for coin in AppConfig.COINS]}")
    print(f"📈 نظام المؤشرات المتقدم المحسن مع {len(AppConfig.INDICATOR_WEIGHTS)} مؤشرات")
    print(f"⚡ التحديث التلقائي كل {AppConfig.UPDATE_INTERVAL//60} دقائق")
    print(f"🔔 نظام إشعارات متقدم مع تحسين الدقة")
    print(f"🔧 وضع التطوير: {os.environ.get('DEBUG', 'False')}")
    print("=" * 60)

    # بدء خيط التحديث التلقائي
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()

    # بدء خيط المراقبة
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()

    logger.info("👁️  بدء خيط مراقبة النظام")
    
    # إرسال إشعار بدء التشغيل إلى NTFY
    def send_startup_notification():
        try:
            startup_message = (
                f"🚀 بدء تشغيل Crypto Signal Analyzer\n"
                f"📊 الإصدار: 3.5.1 (15M timeframe)\n"
                f"📈 مراقبة {len(AppConfig.COINS)} عملة\n"
                f"⏰ وقت البدء: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔄 التحديث التلقائي: كل {AppConfig.UPDATE_INTERVAL//60} دقائق"
            )
            
            headers = {
                "Title": "🚀 بدء تشغيل نظام الإشارات",
                "Priority": "low",
                "Tags": "rocket,green_circle"
            }
            
            response = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=startup_message.encode('utf-8'),
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ تم إرسال إشعار بدء التشغيل إلى NTFY")
            else:
                logger.warning(f"⚠️ فشل إرسال إشعار بدء التشغيل: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار بدء التشغيل: {e}")
    
    # إرسال إشعار البدء
    send_startup_notification()
    
    # تحديث أولي
    try:
        logger.info("بدء التحديث الأولي...")
        success = signal_manager.update_all_signals()
        if success:
            logger.info("✅ التحديث الأولي تم بنجاح")
        else:
            logger.warning("⚠️ التحديث الأولي واجه مشاكل")
    except Exception as e:
        logger.error(f"❌ خطأ في التحديث الأولي: {e}")
    
    # بدء خيط التحديث التلقائي
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
