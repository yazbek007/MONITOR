"""
Crypto Signal Analyzer Bot - النسخة المحسنة والمستقرة
نسخة 3.0 - تم إعادة الكتابة بالكامل لحل مشاكل المنطق والتحقق
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
        CoinConfig(symbol="BTC/USDT", name="Bitcoin", base_asset="BTC", quote_asset="USDT"),
        CoinConfig(symbol="ETH/USDT", name="Ethereum", base_asset="ETH", quote_asset="USDT"),
        CoinConfig(symbol="BNB/USDT", name="Binance Coin", base_asset="BNB", quote_asset="USDT")
    ]
    
    # أوزان المؤشرات
    INDICATOR_WEIGHTS = {
        IndicatorType.TREND_STRENGTH.value: 0.20,
        IndicatorType.MOMENTUM.value: 0.20,
        IndicatorType.VOLUME_ANALYSIS.value: 0.15,
        IndicatorType.VOLATILITY.value: 0.15,
        IndicatorType.MARKET_SENTIMENT.value: 0.15,
        IndicatorType.PRICE_STRUCTURE.value: 0.15
    }
    
    # عتبات الإشارات
    SIGNAL_THRESHOLDS = {
        SignalType.STRONG_BUY: 75,
        SignalType.BUY: 60,
        SignalType.NEUTRAL_HIGH: 55,
        SignalType.NEUTRAL_LOW: 45,
        SignalType.SELL: 40,
        SignalType.STRONG_SELL: 25
    }
    
    # عتبات الإشعارات
    NOTIFICATION_THRESHOLDS = {
        'strong_buy': 75,
        'buy': 60,
        'strong_sell': 25,
        'sell': 40,
        'significant_change': 10  # تغير بنسبة 10%
    }
    
    # إعدادات API
    UPDATE_INTERVAL = 300  # 5 دقائق بالثواني
    DATA_FETCH_TIMEOUT = 30  # ثانية
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # ثانية
    
    # ألوان المؤشرات
    INDICATOR_COLORS = {
        IndicatorType.TREND_STRENGTH.value: '#2E86AB',
        IndicatorType.MOMENTUM.value: '#A23B72',
        IndicatorType.VOLUME_ANALYSIS.value: '#3BB273',
        IndicatorType.VOLATILITY.value: '#F18F01',
        IndicatorType.MARKET_SENTIMENT.value: '#6C757D',
        IndicatorType.PRICE_STRUCTURE.value: '#8F2D56'
    }
    
    # أسماء المؤشرات للعرض
    INDICATOR_DISPLAY_NAMES = {
        IndicatorType.TREND_STRENGTH.value: 'قوة الاتجاه',
        IndicatorType.MOMENTUM.value: 'الزخم',
        IndicatorType.VOLUME_ANALYSIS.value: 'تحليل الحجم',
        IndicatorType.VOLATILITY.value: 'التقلب',
        IndicatorType.MARKET_SENTIMENT.value: 'معنويات السوق',
        IndicatorType.PRICE_STRUCTURE.value: 'هيكل السعر'
    }
    
    # أوصاف المؤشرات
    INDICATOR_DESCRIPTIONS = {
        IndicatorType.TREND_STRENGTH.value: 'يقيس قوة واتجاه الاتجاه العام بناءً على المتوسطات المتحركة',
        IndicatorType.MOMENTUM.value: 'يقيس سرعة وقوة حركة السعر باستخدام RSI ومعدل التغير',
        IndicatorType.VOLUME_ANALYSIS.value: 'يحلل نشاط التداول وعلاقة الحجم بحركة السعر',
        IndicatorType.VOLATILITY.value: 'يقيس مستوى التقلب باستخدام نطاقات بولينجر',
        IndicatorType.MARKET_SENTIMENT.value: 'يعكس المشاعر العامة للسوق باستخدام مؤشر الخوف والجشع',
        IndicatorType.PRICE_STRUCTURE.value: 'يحلل هيكل السعر وأنماط الشموع الحديثة'
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
    
    def validate_ohlcv_data(self, df: pd.DataFrame, min_rows: int = 50) -> bool:
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
        
        return True
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """جلب بيانات OHLCV مع التحقق"""
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
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات التاكر مع التحقق"""
        try:
            def fetch():
                ticker = self.exchange.fetch_ticker(symbol)
                
                # التحقق الأساسي للبيانات
                required_fields = ['last', 'percentage', 'high', 'low', 'quoteVolume']
                if not all(field in ticker for field in required_fields):
                    raise DataValidationError(f"بيانات التاكر غير مكتملة لـ {symbol}")
                
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
                'volume': ticker.get('quoteVolume', 0.0)
            }
        return {'change': 0.0, 'high': 0.0, 'low': 0.0, 'volume': 0.0}


class FearGreedIndexFetcher(DataFetcher):
    """جلب مؤشر الخوف والجشع"""
    
    def __init__(self):
        super().__init__()
        self.last_value = 50
        self.last_update = None
        self.cache_duration = 300  # 5 دقائق بالثواني
    
    def get_index(self) -> Tuple[float, int]:
        """جلب قيمة المؤشر مع التخزين المؤقت"""
        # التحقق من التخزين المؤقت
        if (self.last_update and 
            (datetime.now() - self.last_update).total_seconds() < self.cache_duration):
            return self._convert_to_score(self.last_value), self.last_value
        
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
            
            # تحديث التخزين المؤقت
            self.last_value = fgi_value
            self.last_update = datetime.now()
            
            return self._convert_to_score(fgi_value), fgi_value
            
        except Exception as e:
            logger.error(f"خطأ في جلب مؤشر الخوف والجشع: {e}")
            # استخدام القيمة المخزنة مؤقتاً إذا فشل الجلب
            return self._convert_to_score(self.last_value), self.last_value
    
    def _convert_to_score(self, fgi_value: int) -> float:
        """تحويل قيمة FGI إلى درجة 0-1"""
        # 0 = خوف شديد (إشارة شراء) = 1.0
        # 50 = محايد = 0.5
        # 100 = جشع شديد (إشارة بيع) = 0.0
        return 1.0 - (fgi_value / 100)


class IndicatorsCalculator:
    """حساب المؤشرات مع التحقق من الصحة"""
    
    @staticmethod
    def validate_score(score: float, indicator_name: str) -> float:
        """التحقق من صحة النتيجة وتطبيعها"""
        if score is None or np.isnan(score):
            logger.warning(f"نتيجة {indicator_name} غير صالحة، استخدام القيمة الافتراضية")
            return 0.5
        
        # تطبيع بين 0 و1
        normalized = max(0.0, min(1.0, float(score)))
        return normalized
    
    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame, periods: List[int] = None) -> float:
        """حساب قوة الاتجاه"""
        if periods is None:
            periods = [20, 50, 200]
        
        try:
            if len(df) < max(periods):
                return 0.5
            
            current_price = df['close'].iloc[-1]
            scores = []
            weights = []
            
            for i, period in enumerate(periods):
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean().iloc[-1]
                    
                    if pd.notna(sma) and sma > 0:
                        # حساب المسافة النسبية
                        distance_pct = ((current_price - sma) / sma) * 100
                        
                        # تحويل المسافة إلى درجة
                        if distance_pct > 15:
                            score = 1.0  # فوق المتوسط بكثير
                        elif distance_pct > 8:
                            score = 0.8
                        elif distance_pct > 3:
                            score = 0.6
                        elif distance_pct > -3:
                            score = 0.5
                        elif distance_pct > -8:
                            score = 0.4
                        elif distance_pct > -15:
                            score = 0.2
                        else:
                            score = 0.0  # تحت المتوسط بكثير
                        
                        # وزن أقل للفترات الأطول
                        weight = 1.0 / (i + 1)
                        
                        scores.append(score)
                        weights.append(weight)
            
            if not scores:
                return 0.5
            
            # حساب المتوسط المرجح
            weighted_avg = np.average(scores, weights=weights)
            return IndicatorsCalculator.validate_score(weighted_avg, "قوة الاتجاه")
            
        except Exception as e:
            logger.error(f"خطأ في حساب قوة الاتجاه: {e}")
            return 0.5
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame) -> float:
        """حساب الزخم"""
        try:
            if len(df) < 30:
                return 0.5
            
            # حساب RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
            
            # تطبيع RSI (30=1.0, 70=0.0, خطي بينهما)
            if rsi_value <= 30:
                rsi_score = 1.0
            elif rsi_value >= 70:
                rsi_score = 0.0
            else:
                rsi_score = 1.0 - ((rsi_value - 30) / 40)
            
            # حساب معدل التغير
            roc_scores = []
            for period in [7, 14, 21]:
                if len(df) >= period:
                    roc = ((df['close'].iloc[-1] - df['close'].iloc[-period]) / 
                           df['close'].iloc[-period]) * 100
                    
                    # تحويل ROC إلى درجة
                    if roc > 10:
                        roc_score = 1.0
                    elif roc > 5:
                        roc_score = 0.8
                    elif roc > 2:
                        roc_score = 0.6
                    elif roc > -2:
                        roc_score = 0.5
                    elif roc > -5:
                        roc_score = 0.4
                    elif roc > -10:
                        roc_score = 0.2
                    else:
                        roc_score = 0.0
                    
                    roc_scores.append(roc_score)
            
            roc_avg = np.mean(roc_scores) if roc_scores else 0.5
            
            # دمج RSI وROC
            momentum_score = (rsi_score * 0.6) + (roc_avg * 0.4)
            
            return IndicatorsCalculator.validate_score(momentum_score, "الزخم")
            
        except Exception as e:
            logger.error(f"خطأ في حساب الزخم: {e}")
            return 0.5
    
    @staticmethod
    def calculate_volume_analysis(df: pd.DataFrame, price_change_24h: float = 0) -> float:
        """تحليل الحجم"""
        try:
            if len(df) < 20:
                return 0.5
            
            current_volume = df['volume'].iloc[-1]
            
            # متوسطات الحجم
            avg_volume_7 = df['volume'].tail(7).mean()
            avg_volume_20 = df['volume'].tail(20).mean()
            
            if avg_volume_20 == 0:
                return 0.5
            
            # نسبة الحجم
            volume_ratio_20 = current_volume / avg_volume_20
            
            # تحليل علاقة الحجم بالسعر
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / 
                           df['close'].iloc[-2]) * 100
            
            # حساب درجة الحجم
            if volume_ratio_20 > 2.5:
                # حجم عالي جداً
                if price_change > 2:
                    volume_score = 1.0  # حجم شرائي قوي
                elif price_change < -2:
                    volume_score = 0.0  # حجم بيعي قوي
                else:
                    volume_score = 0.7
            elif volume_ratio_20 > 1.8:
                if price_change > 1:
                    volume_score = 0.8
                elif price_change < -1:
                    volume_score = 0.2
                else:
                    volume_score = 0.6
            elif volume_ratio_20 > 1.3:
                volume_score = 0.55
            elif volume_ratio_20 > 0.7:
                volume_score = 0.5
            elif volume_ratio_20 > 0.4:
                volume_score = 0.45
            else:
                volume_score = 0.3
            
            # تعديل بناء على تغير السعر في 24 ساعة
            if price_change_24h > 5 and volume_score > 0.5:
                volume_score = min(1.0, volume_score + 0.1)
            elif price_change_24h < -5 and volume_score < 0.5:
                volume_score = max(0.0, volume_score - 0.1)
            
            return IndicatorsCalculator.validate_score(volume_score, "تحليل الحجم")
            
        except Exception as e:
            logger.error(f"خطأ في حساب تحليل الحجم: {e}")
            return 0.5
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame) -> float:
        """حساب التقلب"""
        try:
            if len(df) < 20:
                return 0.5
            
            # حساب بولينجر باند
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            current_price = df['close'].iloc[-1]
            current_sma = sma_20.iloc[-1]
            
            if pd.isna(current_sma) or current_sma == 0:
                return 0.5
            
            # حساب موقع السعر في النطاق
            bandwidth = upper_band.iloc[-1] - lower_band.iloc[-1]
            
            if bandwidth > 0:
                position = (current_price - lower_band.iloc[-1]) / bandwidth
            else:
                position = 0.5
            
            # حساب درجة التقلب
            if position > 0.85:
                # قرب النطاق العلوي - احتمال تصحيح
                score = 0.2
            elif position > 0.7:
                score = 0.35
            elif position > 0.3:
                score = 0.5
            elif position > 0.15:
                score = 0.65
            else:
                # قرب النطاق السفلي - احتمال ارتداد
                score = 0.8
            
            # تعديل بناء على عرض النطاق (مستوى التقلب)
            volatility_ratio = std_20.iloc[-1] / current_sma
            
            if volatility_ratio > 0.04:
                # تقلب عالي جداً - مخاطرة عالية
                score = score * 0.8
            elif volatility_ratio > 0.02:
                # تقلب متوسط
                score = score * 0.9
            
            return IndicatorsCalculator.validate_score(score, "التقلب")
            
        except Exception as e:
            logger.error(f"خطأ في حساب التقلب: {e}")
            return 0.5
    
    @staticmethod
    def calculate_price_structure(df: pd.DataFrame) -> float:
        """تحليل هيكل السعر"""
        try:
            if len(df) < 10:
                return 0.5
            
            # تحليل آخر 5 شموع
            recent_candles = df.tail(5)
            
            # حساب قوة الشموع
            candle_strengths = []
            for _, row in recent_candles.iterrows():
                body_size = abs(row['close'] - row['open'])
                total_range = row['high'] - row['low']
                
                if total_range > 0:
                    strength = body_size / total_range
                    # شمعة صاعدة موجبة، هابطة سالبة
                    if row['close'] > row['open']:
                        candle_strengths.append(strength)
                    else:
                        candle_strengths.append(-strength)
            
            avg_candle_strength = np.mean(candle_strengths) if candle_strengths else 0
            
            # تحليل القمم والقيعان
            recent_high = recent_candles['high'].max()
            recent_low = recent_candles['low'].min()
            current_price = df['close'].iloc[-1]
            
            if (recent_high - recent_low) > 0:
                price_position = (current_price - recent_low) / (recent_high - recent_low)
            else:
                price_position = 0.5
            
            # حساب النتيجة النهائية
            base_score = 0.5
            
            # تأثير قوة الشموع
            if avg_candle_strength > 0.3:
                base_score += 0.15
            elif avg_candle_strength > 0.1:
                base_score += 0.08
            elif avg_candle_strength < -0.3:
                base_score -= 0.15
            elif avg_candle_strength < -0.1:
                base_score -= 0.08
            
            # تأثير موقع السعر
            if price_position > 0.8:
                base_score -= 0.1  # قرب المقاومة
            elif price_position < 0.2:
                base_score += 0.1  # قرب الدعم
            
            return IndicatorsCalculator.validate_score(base_score, "هيكل السعر")
            
        except Exception as e:
            logger.error(f"خطأ في حساب هيكل السعر: {e}")
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
        if percentage >= 80:
            return "قوية جداً"
        elif percentage >= 65:
            return "قوية"
        elif percentage >= 55:
            return "متوسطة"
        elif percentage >= 45:
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
        self.notification_history: List[Notification] = []
        self.max_history = 100
        self.last_notification_time = {}
    
    def check_and_send(self, coin_signal: CoinSignal, previous_signal: Optional[CoinSignal]) -> bool:
        """التحقق وإرسال الإشعارات"""
        try:
            current_percentage = coin_signal.total_percentage
            coin_symbol = coin_signal.symbol
            coin_name = coin_signal.name
            
            # التحقق من التكرار
            notification_id = f"{coin_symbol}_{coin_signal.last_updated.timestamp()}"
            if notification_id in self.last_notification_time:
                return False
            
            message = None
            notification_type = None
            priority = "default"
            
            # إشعارات بناء على مستوى الإشارة
            if current_percentage >= AppConfig.NOTIFICATION_THRESHOLDS['strong_buy']:
                message = self._create_buy_message(coin_signal, "قوية")
                notification_type = "strong_buy"
                priority = "high"
            
            elif current_percentage <= AppConfig.NOTIFICATION_THRESHOLDS['strong_sell']:
                message = self._create_sell_message(coin_signal, "قوية")
                notification_type = "strong_sell"
                priority = "high"
            
            elif current_percentage >= AppConfig.NOTIFICATION_THRESHOLDS['buy']:
                if not previous_signal or previous_signal.total_percentage < AppConfig.NOTIFICATION_THRESHOLDS['buy']:
                    message = self._create_buy_message(coin_signal, "عادية")
                    notification_type = "buy"
            
            elif current_percentage <= AppConfig.NOTIFICATION_THRESHOLDS['sell']:
                if not previous_signal or previous_signal.total_percentage > AppConfig.NOTIFICATION_THRESHOLDS['sell']:
                    message = self._create_sell_message(coin_signal, "عادية")
                    notification_type = "sell"
            
            # إشعارات التغير الكبير
            elif previous_signal and abs(current_percentage - previous_signal.total_percentage) >= \
                 AppConfig.NOTIFICATION_THRESHOLDS['significant_change']:
                
                change = current_percentage - previous_signal.total_percentage
                direction = "صاعد" if change > 0 else "هابط"
                signal_type = coin_signal.signal_type.value
                
                message = f"🔄 تغير كبير في {coin_name}\n"
                message += f"من {previous_signal.total_percentage:.1f}% إلى {current_percentage:.1f}% ({direction})\n"
                message += f"📊 الإشارة الحالية: {signal_type}\n"
                message += f"💰 السعر: ${coin_signal.current_price:,.2f}\n"
                message += f"⏰ {datetime.now().strftime('%H:%M')}"
                
                notification_type = "significant_change"
            
            if message:
                success = self.send_ntfy_notification(message, notification_type, priority)
                
                if success:
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
                    self.last_notification_time[notification_id] = datetime.now()
                    
                    logger.info(f"تم إرسال إشعار {notification_type} لـ {coin_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"خطأ في التحقق من الإشعارات: {e}")
            return False
    
    def _create_buy_message(self, coin_signal: CoinSignal, strength: str) -> str:
        """إنشاء رسالة شراء"""
        return (f"🚀 إشارة شراء {strength}: {coin_signal.name} ({coin_signal.symbol})\n"
                f"📊 القوة: {coin_signal.total_percentage:.1f}%\n"
                f"💰 السعر: ${coin_signal.current_price:,.2f}\n"
                f"📈 التغير 24h: {coin_signal.price_change_24h:+.2f}%\n"
                f"⏰ {datetime.now().strftime('%H:%M')}")
    
    def _create_sell_message(self, coin_signal: CoinSignal, strength: str) -> str:
        """إنشاء رسالة بيع"""
        return (f"⚠️ إشارة بيع {strength}: {coin_signal.name} ({coin_signal.symbol})\n"
                f"📊 القوة: {coin_signal.total_percentage:.1f}%\n"
                f"💰 السعر: ${coin_signal.current_price:,.2f}\n"
                f"📈 التغير 24h: {coin_signal.price_change_24h:+.2f}%\n"
                f"⏰ {datetime.now().strftime('%H:%M')}")
    
    def send_ntfy_notification(self, message: str, notification_type: str, priority: str) -> bool:
        """إرسال إشعار عبر NTFY"""
        try:
            tags = {
                'strong_buy': 'heavy_plus_sign,green_circle',
                'buy': 'chart_increasing,blue_circle',
                'strong_sell': 'heavy_minus_sign,red_circle',
                'sell': 'chart_decreasing,orange_circle',
                'significant_change': 'arrows_counterclockwise,yellow_circle'
            }
            
            headers = {
                "Title": "📊 إشعار إشارة التشفير",
                "Priority": priority,
                "Tags": tags.get(notification_type, 'loudspeaker')
            }
            
            response = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"خطأ في إرسال إشعار NTFY: {e}")
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
        """تحديث جميع الإشارات"""
        with self.update_lock:
            logger.info("بدء تحديث جميع الإشارات...")
            
            try:
                # تحديث مؤشر الخوف والجشع
                self._update_fear_greed_index()
                
                success_count = 0
                failed_coins = []
                
                for coin_config in AppConfig.COINS:
                    if not coin_config.enabled:
                        continue
                    
                    try:
                        coin_signal = self._process_coin_signal(coin_config)
                        
                        if coin_signal.is_valid:
                            # التحقق من الإشعارات
                            previous_signal = self.signals.get(coin_config.symbol)
                            self.notification_manager.check_and_send(coin_signal, previous_signal)
                            
                            # حفظ الإشارة
                            self.signals[coin_config.symbol] = coin_signal
                            success_count += 1
                            
                            logger.info(f"تم تحديث {coin_config.name}: {coin_signal.total_percentage:.1f}% ({coin_signal.signal_type.value})")
                        else:
                            failed_coins.append(f"{coin_config.name}: {coin_signal.error_message}")
                            
                    except Exception as e:
                        error_msg = f"خطأ في معالجة {coin_config.name}: {str(e)}"
                        logger.error(error_msg)
                        failed_coins.append(error_msg)
                        continue
                
                # تحديث وقت التحديث الأخير
                self.last_update = datetime.now()
                
                # حفظ في السجل
                self._save_to_history()
                
                # تنظيف الإشارات القديمة
                self._cleanup_old_data()
                
                logger.info(f"تم تحديث {success_count}/{len(AppConfig.COINS)} إشارات بنجاح")
                
                if failed_coins:
                    logger.warning(f"العملات التي فشلت: {', '.join(failed_coins)}")
                
                return success_count > 0
                
            except Exception as e:
                logger.error(f"خطأ في تحديث الإشارات: {e}")
                return False
    
    def _update_fear_greed_index(self):
        """تحديث مؤشر الخوف والجشع"""
        try:
            self.fear_greed_score, self.fear_greed_index = self.fgi_fetcher.get_index()
            logger.info(f"مؤشر الخوف والجشع: {self.fear_greed_index} (النتيجة: {self.fear_greed_score:.2f})")
        except Exception as e:
            logger.error(f"خطأ في تحديث مؤشر الخوف والجشع: {e}")
    
    def _process_coin_signal(self, coin_config: CoinConfig) -> CoinSignal:
        """معالجة إشارة عملة واحدة"""
        try:
            # جلب البيانات
            df = self.data_fetcher.get_ohlcv(coin_config.symbol, timeframe='1h', limit=200)
            if df is None or df.empty:
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
                    error_message="فشل جلب بيانات OHLCV"
                )
            
            # جلب الإحصائيات
            stats_24h = self.data_fetcher.get_24h_stats(coin_config.symbol)
            current_price = self.data_fetcher.get_current_price(coin_config.symbol)
            
            # حساب المؤشرات
            trend_score = self.calculator.calculate_trend_strength(df)
            momentum_score = self.calculator.calculate_momentum(df)
            volume_score = self.calculator.calculate_volume_analysis(df, stats_24h['change'])
            volatility_score = self.calculator.calculate_volatility(df)
            price_structure_score = self.calculator.calculate_price_structure(df)
            
            # جمع المؤشرات
            indicator_scores = {
                IndicatorType.TREND_STRENGTH.value: trend_score,
                IndicatorType.MOMENTUM.value: momentum_score,
                IndicatorType.VOLUME_ANALYSIS.value: volume_score,
                IndicatorType.VOLATILITY.value: volatility_score,
                IndicatorType.MARKET_SENTIMENT.value: self.fear_greed_score,
                IndicatorType.PRICE_STRUCTURE.value: price_structure_score
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
        # تنظيف الإشارات القديمة (أقدم من ساعتين)
        cutoff_time = datetime.now() - timedelta(hours=2)
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
            return f"قبل {hours} ساعة"
        elif delta.seconds >= 60:
            minutes = delta.seconds // 60
            return f"قبل {minutes} دقيقة"
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
        'version': '3.0.0',
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


def background_updater():
    """تحديث البيانات في الخلفية"""
    while True:
        try:
            signal_manager.update_all_signals()
            time.sleep(AppConfig.UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"خطأ في التحديث التلقائي: {e}")
            time.sleep(60)  # انتظار دقيقة ثم إعادة المحاولة


# ======================
# تشغيل التطبيق
# ======================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 بدء تشغيل Crypto Signal Analyzer - الإصدار 3.0")
    print("=" * 60)
    print(f"📊 مراقبة العملات: {[coin.name for coin in AppConfig.COINS]}")
    print(f"📈 نظام المؤشرات المتقدم مع {len(AppConfig.INDICATOR_WEIGHTS)} مؤشرات")
    print(f"⚡ التحديث التلقائي كل {AppConfig.UPDATE_INTERVAL//60} دقائق")
    print(f"🔔 نظام إشعارات متقدم مع التحقق من التكرار")
    print(f"🔧 وضع التطوير: {os.environ.get('DEBUG', 'False')}")
    print("=" * 60)
    
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
    
    # تشغيل Flask
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 تشغيل الخادم على المنفذ {port}")
    print(f"🔧 وضع التصحيح: {'مفعل' if debug_mode else 'معطل'}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=False)
