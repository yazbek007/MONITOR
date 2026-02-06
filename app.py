"""
Crypto Signal Analyzer Bot - نسخة محسنة ومعدلة
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import ccxt
import requests
import warnings
warnings.filterwarnings('ignore')

# تهيئة Flask App
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-signal-secret-2024')

# ======================
# إعدادات التطبيق
# ======================

# العملات المطلوبة
COINS = [
    {"symbol": "BTC/USDT", "name": "Bitcoin"},
    {"symbol": "ETH/USDT", "name": "Ethereum"},
    {"symbol": "BNB/USDT", "name": "Binance Coin"}
]

# أوزان المؤشرات المحسنة
INDICATOR_WEIGHTS = {
    'trend_strength': 0.20,      # قوة الاتجاه
    'momentum': 0.20,            # الزخم (RSI + MACD)
    'volume_analysis': 0.15,     # تحليل الحجم
    'volatility': 0.15,          # التقلب (بولينجر باند)
    'market_sentiment': 0.15,    # معنويات السوق
    'price_structure': 0.15      # هيكل السعر
}

# عتبات الإشعارات المحسنة
NOTIFICATION_THRESHOLDS = {
    'strong_buy': 75,
    'buy': 60,
    'neutral_high': 55,
    'neutral_low': 45,
    'sell': 40,
    'strong_sell': 25,
    'change_threshold': 10
}

# إعدادات Binance API
BINANCE_CONFIG = {
    'apiKey': os.environ.get('BINANCE_API_KEY', ''),
    'secret': os.environ.get('BINANCE_SECRET_KEY', ''),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
}

# إعدادات NTFY
NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'crypto_signals_alerts')
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

# ======================
# تخزين البيانات
# ======================

signals_data = {
    'last_update': None,
    'coins': {},
    'history': [],
    'notifications': []
}

# ======================
# فئات المساعدين
# ======================

class BinanceDataFetcher:
    """فئة لجلب البيانات من Binance"""
    
    def __init__(self):
        self.exchange = ccxt.binance(BINANCE_CONFIG)
        self.exchange.load_markets()
    
    def get_ohlcv(self, symbol, timeframe='1h', limit=500):
        """جلب بيانات OHLCV"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def get_ticker(self, symbol):
        """جلب البيانات الحالية"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Error fetching ticker {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol):
        """جلب السعر الحالي"""
        ticker = self.get_ticker(symbol)
        return ticker['last'] if ticker else 0
    
    def get_24h_stats(self, symbol):
        """جلب إحصائيات 24 ساعة"""
        ticker = self.get_ticker(symbol)
        if ticker:
            return {
                'change': ticker.get('percentage', 0),
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0),
                'volume': ticker.get('quoteVolume', 0)
            }
        return None

class IndicatorsCalculator:
    """فئة لحساب المؤشرات باستخدام pandas/numpy فقط"""
    
    @staticmethod
    def calculate_trend_strength(df, periods=[20, 50, 200]):
        """حساب قوة الاتجاه"""
        try:
            if len(df) < max(periods):
                return 0.5
            
            scores = []
            current_price = df['close'].iloc[-1]
            
            for period in periods:
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean().iloc[-1]
                    if pd.notna(sma):
                        # حساب المسافة من المتوسط
                        distance = ((current_price - sma) / sma) * 100
                        
                        # تقييم قوة الاتجاه
                        if abs(distance) > 10:
                            score = 1.0 if distance > 0 else 0.0
                        elif abs(distance) > 5:
                            score = 0.75 if distance > 0 else 0.25
                        elif abs(distance) > 2:
                            score = 0.6 if distance > 0 else 0.4
                        else:
                            score = 0.5
                        
                        scores.append(score)
            
            if not scores:
                return 0.5
            
            # وزن الفترات الأقرب أكثر
            weights = [1.0, 0.7, 0.3][:len(scores)]
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            print(f"Error calculating trend strength: {e}")
            return 0.5
    
    @staticmethod
    def calculate_momentum(df):
        """حساب الزخم (RSI + معدل التغير)"""
        try:
            if len(df) < 30:
                return 0.5
            
            # حساب RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
            
            if pd.isna(rsi_value):
                rsi_value = 50
            
            # حساب معدل التغير
            roc_7 = ((df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7]) * 100 if len(df) >= 7 else 0
            roc_14 = ((df['close'].iloc[-1] - df['close'].iloc[-14]) / df['close'].iloc[-14]) * 100 if len(df) >= 14 else 0
            
            # تسجيل RSI (0-1)
            if rsi_value <= 30:
                rsi_score = 1.0  # تشبع بيعي قوي
            elif rsi_value >= 70:
                rsi_score = 0.0  # تشبع شرائي قوي
            else:
                # تحويل خطي
                rsi_score = 1.0 - ((rsi_value - 30) / 40)
            
            # تسجيل معدل التغير
            roc_score = 0.5
            if roc_7 > 5 or roc_14 > 10:
                roc_score = 1.0
            elif roc_7 > 2 or roc_14 > 5:
                roc_score = 0.75
            elif roc_7 > 0 or roc_14 > 0:
                roc_score = 0.6
            elif roc_7 < -5 or roc_14 < -10:
                roc_score = 0.0
            elif roc_7 < -2 or roc_14 < -5:
                roc_score = 0.25
            elif roc_7 < 0 or roc_14 < 0:
                roc_score = 0.4
            
            # دمج النتائج
            momentum_score = (rsi_score * 0.6) + (roc_score * 0.4)
            
            return momentum_score
            
        except Exception as e:
            print(f"Error calculating momentum: {e}")
            return 0.5
    
    @staticmethod
    def calculate_volume_analysis(df, ticker_data=None):
        """تحليل الحجم"""
        try:
            if len(df) < 30:
                return 0.5
            
            current_volume = df['volume'].iloc[-1]
            
            # متوسطات الحجم
            avg_volume_7 = df['volume'].tail(7).mean()
            avg_volume_30 = df['volume'].tail(30).mean()
            
            if avg_volume_30 == 0:
                return 0.5
            
            # نسب الحجم
            volume_ratio_7 = current_volume / avg_volume_7 if avg_volume_7 > 0 else 1
            volume_ratio_30 = current_volume / avg_volume_30 if avg_volume_30 > 0 else 1
            
            # تحليل علاقة السعر بالحجم
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            volume_score = 0.5
            
            # حجم قوي مع اتجاه سعري
            if volume_ratio_30 > 2.0:
                if price_change > 1:
                    volume_score = 1.0  # حجم شرائي قوي
                elif price_change < -1:
                    volume_score = 0.0  # حجم بيعي قوي
                else:
                    volume_score = 0.7
            elif volume_ratio_30 > 1.5:
                if price_change > 0.5:
                    volume_score = 0.8
                elif price_change < -0.5:
                    volume_score = 0.2
                else:
                    volume_score = 0.6
            elif volume_ratio_30 > 1.2:
                volume_score = 0.55
            elif volume_ratio_30 > 0.8:
                volume_score = 0.5
            elif volume_ratio_30 > 0.5:
                volume_score = 0.45
            else:
                volume_score = 0.3
            
            return volume_score
            
        except Exception as e:
            print(f"Error calculating volume analysis: {e}")
            return 0.5
    
    @staticmethod
    def calculate_volatility(df):
        """حساب التقلب (بولينجر باند)"""
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
            current_std = std_20.iloc[-1]
            
            if pd.isna(current_sma) or pd.isna(current_std) or current_std == 0:
                return 0.5
            
            # حساب موقع السعر في النطاق
            bandwidth = upper_band.iloc[-1] - lower_band.iloc[-1]
            position = (current_price - lower_band.iloc[-1]) / bandwidth if bandwidth > 0 else 0.5
            
            # تحليل التقلب
            volatility_ratio = current_std / current_sma
            
            # تسجيل التقلب
            if position > 0.8:
                # قرب النطاق العلوي - تشبع شرائي
                score = 0.2
            elif position < 0.2:
                # قرب النطاق السفلي - تشبع بيعي
                score = 0.8
            else:
                # في منتصف النطاق
                score = 0.5
            
            # تعديل بناء على مستوى التقلب
            if volatility_ratio > 0.03:
                # تقلب عالي - فرص وتحديات
                score = score * 0.9 + 0.05
            elif volatility_ratio < 0.01:
                # تقلب منخفض - استقرار
                score = score * 0.9 + 0.05
            
            return max(0, min(1, score))
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return 0.5
    
    @staticmethod
    def calculate_market_sentiment():
        """حساب معنويات السوق (الخوف والجشع)"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fgi_value = int(data['data'][0]['value'])
                
                # تحويل مباشر (0-100 إلى 0-1)
                # 0 = جشع شديد (إشارة بيع) = 0.0
                # 100 = خوف شديد (إشارة شراء) = 1.0
                sentiment_score = 1.0 - (fgi_value / 100)
                
                return sentiment_score, fgi_value
            else:
                return 0.5, 50
        except Exception as e:
            print(f"Error fetching fear/greed index: {e}")
            return 0.5, 50
    
    @staticmethod
    def calculate_price_structure(df):
        """تحليل هيكل السعر"""
        try:
            if len(df) < 10:
                return 0.5
            
            # تحليل الشموع الأخيرة
            last_5_candles = df.tail(5)
            
            # حساب عدد الشموع الصاعدة مقابل الهابطة
            bullish_count = sum(1 for _, row in last_5_candles.iterrows() if row['close'] > row['open'])
            bearish_count = 5 - bullish_count
            
            # قوة الشموع
            candle_strengths = []
            for _, row in last_5_candles.iterrows():
                body_size = abs(row['close'] - row['open'])
                total_range = row['high'] - row['low']
                
                if total_range > 0:
                    strength = body_size / total_range
                    # شمعة صاعدة أقوى من هابطة
                    if row['close'] > row['open']:
                        candle_strengths.append(strength)
                    else:
                        candle_strengths.append(-strength)
            
            avg_candle_strength = sum(candle_strengths) / len(candle_strengths) if candle_strengths else 0
            
            # تحليل القمم والقيعان
            recent_high = last_5_candles['high'].max()
            recent_low = last_5_candles['low'].min()
            current_price = df['close'].iloc[-1]
            
            # موقع السعر في النطاق الأخير
            price_position = (current_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            
            # حساب النتيجة
            structure_score = 0.5
            
            # تأثير عدد الشموع
            if bullish_count >= 4:
                structure_score += 0.2
            elif bullish_count >= 3:
                structure_score += 0.1
            elif bearish_count >= 4:
                structure_score -= 0.2
            elif bearish_count >= 3:
                structure_score -= 0.1
            
            # تأثير قوة الشموع
            structure_score += avg_candle_strength * 0.2
            
            # تأثير موقع السعر
            if price_position > 0.7:
                structure_score -= 0.1  # قرب المقاومة
            elif price_position < 0.3:
                structure_score += 0.1  # قرب الدعم
            
            return max(0, min(1, structure_score))
            
        except Exception as e:
            print(f"Error calculating price structure: {e}")
            return 0.5

class SignalProcessor:
    """معالجة الإشارات"""
    
    @staticmethod
    def calculate_weighted_signal(indicator_scores):
        """حساب الإشارة المرجحة"""
        total_score = 0
        weighted_scores = {}
        
        for indicator, score in indicator_scores.items():
            if indicator in INDICATOR_WEIGHTS:
                weighted = score * INDICATOR_WEIGHTS[indicator]
                weighted_scores[indicator] = {
                    'raw_score': score,
                    'weighted_score': weighted,
                    'percentage': weighted * 100
                }
                total_score += weighted
        
        total_percentage = total_score * 100
        
        return {
            'total_score': total_score,
            'total_percentage': total_percentage,
            'weighted_scores': weighted_scores,
            'signal_strength': SignalProcessor.get_signal_strength(total_percentage),
            'signal_type': SignalProcessor.get_signal_type(total_percentage),
            'signal_color': SignalProcessor.get_signal_color(total_percentage)
        }
    
    @staticmethod
    def get_signal_strength(percentage):
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
    def get_signal_type(percentage):
        if percentage >= NOTIFICATION_THRESHOLDS['strong_buy']:
            return "شراء قوي"
        elif percentage >= NOTIFICATION_THRESHOLDS['buy']:
            return "شراء"
        elif percentage >= NOTIFICATION_THRESHOLDS['neutral_high']:
            return "محايد موجب"
        elif percentage >= NOTIFICATION_THRESHOLDS['neutral_low']:
            return "محايد سالب"
        elif percentage >= NOTIFICATION_THRESHOLDS['sell']:
            return "بيع"
        else:
            return "بيع قوي"
    
    @staticmethod
    def get_signal_color(percentage):
        if percentage >= NOTIFICATION_THRESHOLDS['strong_buy']:
            return "success"
        elif percentage >= NOTIFICATION_THRESHOLDS['buy']:
            return "info"
        elif percentage >= NOTIFICATION_THRESHOLDS['neutral_high']:
            return "secondary"
        elif percentage >= NOTIFICATION_THRESHOLDS['neutral_low']:
            return "warning"
        elif percentage >= NOTIFICATION_THRESHOLDS['sell']:
            return "warning"
        else:
            return "danger"

class NotificationManager:
    """مدير الإشعارات"""
    
    @staticmethod
    def check_and_send_notification(coin_data, previous_data):
        """التحقق وإرسال الإشعارات"""
        try:
            current_signal = coin_data['total_percentage']
            coin_symbol = coin_data['symbol']
            coin_name = coin_data['name']
            
            prev_signal = None
            if previous_data:
                prev_signal = previous_data.get('total_percentage', None)
            
            message = None
            notification_type = None
            
            # إشعارات بناء على مستوى الإشارة
            if current_signal >= NOTIFICATION_THRESHOLDS['strong_buy']:
                message = f"🚀 إشارة شراء قوية: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n📈 التغير 24h: {coin_data.get('24h_change', 0):+.2f}%"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "strong_buy"
            
            elif current_signal <= NOTIFICATION_THRESHOLDS['strong_sell']:
                message = f"⚠️ إشارة بيع قوية: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n📈 التغير 24h: {coin_data.get('24h_change', 0):+.2f}%"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "strong_sell"
            
            elif current_signal >= NOTIFICATION_THRESHOLDS['buy'] and (not prev_signal or prev_signal < NOTIFICATION_THRESHOLDS['buy']):
                message = f"📈 إشارة شراء: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "buy"
            
            elif current_signal <= NOTIFICATION_THRESHOLDS['sell'] and (not prev_signal or prev_signal > NOTIFICATION_THRESHOLDS['sell']):
                message = f"📉 إشارة بيع: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "sell"
            
            # إشعارات تغير كبير
            elif (prev_signal and 
                  abs(current_signal - prev_signal) >= NOTIFICATION_THRESHOLDS['change_threshold']):
                change = current_signal - prev_signal
                direction = "صاعد 📈" if change > 0 else "هابط 📉"
                signal_type = SignalProcessor.get_signal_type(current_signal)
                
                message = f"🔄 تغير كبير في {coin_name}"
                message += f"\n{prev_signal:.1f}% → {current_signal:.1f}% ({direction})"
                message += f"\n📶 الإشارة الحالية: {signal_type}"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "significant_change"
            
            if message:
                success = NotificationManager.send_ntfy_notification(message, notification_type)
                
                if success:
                    notification = {
                        'timestamp': datetime.now(),
                        'coin': coin_name,
                        'symbol': coin_symbol,
                        'message': message,
                        'type': notification_type,
                        'signal_strength': current_signal,
                        'price': coin_data.get('current_price', 0)
                    }
                    
                    signals_data['notifications'].append(notification)
                    
                    if len(signals_data['notifications']) > 50:
                        signals_data['notifications'] = signals_data['notifications'][-50:]
                    
                    return True
            
            return False
        except Exception as e:
            print(f"Error in notification: {e}")
            return False
    
    @staticmethod
    def send_ntfy_notification(message, notification_type):
        """إرسال إشعار عبر NTFY"""
        try:
            # تحديد الألوان والأيقونات بناء على نوع الإشعار
            tags = {
                'strong_buy': 'heavy_plus_sign,green_circle',
                'buy': 'chart_increasing,blue_circle',
                'strong_sell': 'heavy_minus_sign,red_circle',
                'sell': 'chart_decreasing,orange_circle',
                'significant_change': 'arrows_counterclockwise,yellow_circle'
            }
            
            priority = {
                'strong_buy': 'high',
                'strong_sell': 'high',
                'buy': 'default',
                'sell': 'default',
                'significant_change': 'default'
            }
            
            headers = {
                "Title": "📊 إشعار إشارة التشفير",
                "Priority": priority.get(notification_type, 'default'),
                "Tags": tags.get(notification_type, 'loudspeaker')
            }
            
            response = requests.post(
                NTFY_URL,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending NTFY notification: {e}")
            return False

# ======================
# الوظائف المساعدة
# ======================

def get_indicator_display_name(indicator_key):
    """تحويل اسم المؤشر للعرض"""
    names = {
        'trend_strength': 'قوة الاتجاه',
        'momentum': 'الزخم',
        'volume_analysis': 'تحليل الحجم',
        'volatility': 'التقلب',
        'market_sentiment': 'معنويات السوق',
        'price_structure': 'هيكل السعر'
    }
    return names.get(indicator_key, indicator_key)

def get_indicator_color(indicator_key):
    """الحصول على لون المؤشر"""
    colors = {
        'trend_strength': '#2E86AB',     # أزرق
        'momentum': '#A23B72',           # بنفسجي
        'volume_analysis': '#3BB273',    # أخضر
        'volatility': '#F18F01',         # برتقالي
        'market_sentiment': '#6C757D',   # رمادي
        'price_structure': '#8F2D56'     # أحمر غامق
    }
    return colors.get(indicator_key, '#2E86AB')

def get_indicator_description(indicator_key):
    """الحصول على وصف المؤشر"""
    descriptions = {
        'trend_strength': 'يقيس قوة واتجاه الاتجاه العام بناءً على المتوسطات المتحركة',
        'momentum': 'يقيس سرعة وقوة حركة السعر باستخدام RSI ومعدل التغير',
        'volume_analysis': 'يحلل نشاط التداول وعلاقة الحجم بحركة السعر',
        'volatility': 'يقيس مستوى التقلب باستخدام نطاقات بولينجر',
        'market_sentiment': 'يعكس المشاعر العامة للسوق باستخدام مؤشر الخوف والجشع',
        'price_structure': 'يحلل هيكل السعر وأنماط الشموع الحديثة'
    }
    return descriptions.get(indicator_key, '')

def format_number(value):
    """تنسيق الأرقام للعرض"""
    try:
        if value is None:
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

def format_percentage(value):
    """تنسيق النسب المئوية"""
    try:
        if value is None:
            return "0.00%"
        
        value = float(value)
        prefix = "+" if value > 0 else ""
        return f"{prefix}{value:.2f}%"
    except:
        return "0.00%"

def format_time_delta(dt):
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

# ======================
# الوظائف الرئيسية
# ======================

def update_signals():
    """تحديث جميع الإشارات"""
    global signals_data
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] تحديث الإشارات...")
    
    fetcher = BinanceDataFetcher()
    calculator = IndicatorsCalculator()
    
    # جلب مؤشر الخوف والجشع (مرة واحدة لجميع العملات)
    sentiment_score, fgi_value = calculator.calculate_market_sentiment()
    
    for coin in COINS:
        try:
            symbol = coin['symbol']
            name = coin['name']
            
            print(f"  جلب بيانات {name}...")
            
            # جلب البيانات
            df = fetcher.get_ohlcv(symbol, timeframe='1h', limit=200)
            if df is None or df.empty:
                print(f"  فشل جلب بيانات {name}")
                continue
            
            # جلب البيانات الحالية والإحصائيات
            current_price = fetcher.get_current_price(symbol)
            stats_24h = fetcher.get_24h_stats(symbol)
            
            # حساب المؤشرات الجديدة
            trend_score = calculator.calculate_trend_strength(df)
            momentum_score = calculator.calculate_momentum(df)
            volume_score = calculator.calculate_volume_analysis(df)
            volatility_score = calculator.calculate_volatility(df)
            price_structure_score = calculator.calculate_price_structure(df)
            
            # جمع درجات المؤشرات
            indicator_scores = {
                'trend_strength': trend_score,
                'momentum': momentum_score,
                'volume_analysis': volume_score,
                'volatility': volatility_score,
                'market_sentiment': sentiment_score,
                'price_structure': price_structure_score
            }
            
            # حساب الإشارة المرجحة
            previous_data = signals_data['coins'].get(symbol, {})
            
            signal_result = SignalProcessor.calculate_weighted_signal(indicator_scores)
            
            # إعداد بيانات العملة
            coin_data = {
                'symbol': symbol,
                'name': name,
                'current_price': current_price,
                '24h_change': stats_24h.get('change', 0) if stats_24h else 0,
                '24h_high': stats_24h.get('high', 0) if stats_24h else 0,
                '24h_low': stats_24h.get('low', 0) if stats_24h else 0,
                '24h_volume': stats_24h.get('volume', 0) if stats_24h else 0,
                'indicator_scores': indicator_scores,
                'total_percentage': signal_result['total_percentage'],
                'signal_strength': signal_result['signal_strength'],
                'signal_type': signal_result['signal_type'],
                'signal_color': signal_result['signal_color'],
                'weighted_scores': signal_result['weighted_scores'],
                'last_updated': datetime.now(),
                'fear_greed_value': fgi_value,
                'price_change': None
            }
            
            # حساب التغير إذا كانت هناك بيانات سابقة
            if previous_data and 'current_price' in previous_data:
                prev_price = previous_data['current_price']
                if prev_price > 0 and current_price > 0:
                    price_change = ((current_price - prev_price) / prev_price) * 100
                    coin_data['price_change'] = price_change
            
            # التحقق من الإشعارات
            NotificationManager.check_and_send_notification(coin_data, previous_data)
            
            # حفظ البيانات
            signals_data['coins'][symbol] = coin_data
            
            print(f"  {name}: {signal_result['total_percentage']:.1f}% ({signal_result['signal_type']})")
            
        except Exception as e:
            print(f"Error processing {coin['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # تحديث وقت التحديث الأخير
    signals_data['last_update'] = datetime.now()
    
    # حفظ في السجل
    history_entry = {
        'timestamp': datetime.now(),
        'signals': {symbol: data['total_percentage'] for symbol, data in signals_data['coins'].items()}
    }
    signals_data['history'].append(history_entry)
    
    # الحفاظ على آخر 100 سجل
    if len(signals_data['history']) > 100:
        signals_data['history'] = signals_data['history'][-100:]
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] تم تحديث الإشارات بنجاح")
    
    return True

def background_updater():
    """تحديث البيانات في الخلفية"""
    while True:
        try:
            update_signals()
            time.sleep(300)  # 5 دقائق
        except Exception as e:
            print(f"Error in background updater: {e}")
            time.sleep(60)  # انتظار دقيقة ثم إعادة المحاولة

# ======================
# Routes Flask
# ======================

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    coins_data = []
    
    for coin in COINS:
        symbol = coin['symbol']
        if symbol in signals_data['coins']:
            coin_info = signals_data['coins'][symbol].copy()
            
            # تحضير بيانات المؤشرات للعرض
            indicators = []
            weighted_scores = coin_info.get('weighted_scores', {})
            
            for ind_name, ind_data in weighted_scores.items():
                indicators.append({
                    'name': ind_name,
                    'display_name': get_indicator_display_name(ind_name),
                    'description': get_indicator_description(ind_name),
                    'raw_score': ind_data['raw_score'] * 100,
                    'weighted_score': ind_data['weighted_score'] * 100,
                    'percentage': ind_data['percentage']
                })
            
            coin_info['indicators'] = indicators
            coin_info['formatted_price'] = format_number(coin_info['current_price'])
            coin_info['formatted_24h_change'] = format_percentage(coin_info.get('24h_change', 0))
            coin_info['formatted_24h_volume'] = format_number(coin_info.get('24h_volume', 0))
            coin_info['formatted_price_change'] = format_percentage(coin_info.get('price_change', 0))
            coin_info['last_updated_str'] = format_time_delta(coin_info.get('last_updated'))
            
            coins_data.append(coin_info)
        else:
            # بيانات افتراضية
            coins_data.append({
                'symbol': coin['symbol'],
                'name': coin['name'],
                'current_price': 0,
                'formatted_price': '0',
                '24h_change': 0,
                'formatted_24h_change': '0.00%',
                'total_percentage': 50,
                'signal_strength': 'غير متوفر',
                'signal_type': 'محايد',
                'signal_color': 'secondary',
                'indicators': [],
                'last_updated': None,
                'last_updated_str': 'غير معروف',
                'fear_greed_value': 50,
                'price_change': 0,
                'formatted_price_change': '0.00%'
            })
    
    # ترتيب العملات حسب قوة الإشارة
    coins_data.sort(key=lambda x: x['total_percentage'], reverse=True)
    
    # بيانات الإشعارات الأخيرة
    recent_notifications = signals_data['notifications'][-10:] if signals_data['notifications'] else []
    
    # إحصائيات
    total_signals = [c['total_percentage'] for c in coins_data if c['total_percentage'] > 0]
    signal_types = {
        'strong_buy': sum(1 for c in coins_data if c.get('total_percentage', 0) >= NOTIFICATION_THRESHOLDS['strong_buy']),
        'buy': sum(1 for c in coins_data if NOTIFICATION_THRESHOLDS['buy'] <= c.get('total_percentage', 0) < NOTIFICATION_THRESHOLDS['strong_buy']),
        'neutral': sum(1 for c in coins_data if NOTIFICATION_THRESHOLDS['neutral_low'] <= c.get('total_percentage', 0) < NOTIFICATION_THRESHOLDS['neutral_high']),
        'sell': sum(1 for c in coins_data if NOTIFICATION_THRESHOLDS['sell'] <= c.get('total_percentage', 0) < NOTIFICATION_THRESHOLDS['neutral_low']),
        'strong_sell': sum(1 for c in coins_data if c.get('total_percentage', 0) < NOTIFICATION_THRESHOLDS['sell'])
    }
    
    stats = {
        'total_coins': len(COINS),
        'updated_coins': len(signals_data['coins']),
        'avg_signal': np.mean(total_signals) if total_signals else 50,
        'strong_buy_signals': signal_types['strong_buy'],
        'buy_signals': signal_types['buy'],
        'neutral_signals': signal_types['neutral'],
        'sell_signals': signal_types['sell'],
        'strong_sell_signals': signal_types['strong_sell'],
        'last_update': signals_data['last_update'],
        'last_update_str': format_time_delta(signals_data['last_update']),
        'total_notifications': len(signals_data['notifications'])
    }
    
    # حساب وقت التحديث التالي
    next_update_time = None
    if signals_data['last_update']:
        next_update_time = signals_data['last_update'] + timedelta(seconds=300)
    
    return render_template('index.html',
                         coins=coins_data,
                         stats=stats,
                         next_update_time=next_update_time,
                         notifications=recent_notifications,
                         get_indicator_color=get_indicator_color,
                         format_number=format_number,
                         format_percentage=format_percentage,
                         indicator_weights=INDICATOR_WEIGHTS)

@app.route('/api/signals')
def api_signals():
    """API لإرجاع الإشارات"""
    return jsonify(signals_data['coins'])

@app.route('/api/update', methods=['POST'])
def manual_update():
    """تحديث يدوي للإشارات"""
    try:
        success = update_signals()
        if success:
            return jsonify({
                'status': 'success', 
                'message': 'تم التحديث بنجاح',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'فشل التحديث'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """فحص صحة التطبيق"""
    now = datetime.now()
    last_update = signals_data['last_update']
    
    status = 'healthy'
    if last_update:
        time_since_update = (now - last_update).total_seconds()
        if time_since_update > 600:  # أكثر من 10 دقائق
            status = 'warning'
        elif time_since_update > 1800:  # أكثر من 30 دقيقة
            status = 'unhealthy'
    
    return jsonify({
        'status': status,
        'last_update': last_update.isoformat() if last_update else None,
        'time_since_update': (now - last_update).total_seconds() if last_update else None,
        'coins_available': len(signals_data['coins']),
        'uptime': time.time() - start_time if 'start_time' in globals() else 0,
        'version': '2.0.0'
    })

@app.route('/api/notifications')
def get_notifications():
    """الحصول على الإشعارات"""
    limit = request.args.get('limit', 10, type=int)
    notifications = signals_data['notifications'][-limit:] if signals_data['notifications'] else []
    return jsonify({'notifications': notifications, 'total': len(signals_data['notifications'])})

@app.route('/api/coins')
def get_coins():
    """الحصول على قائمة العملات"""
    return jsonify({'coins': COINS})

@app.route('/api/indicators')
def get_indicators():
    """الحصول على معلومات المؤشرات"""
    indicators_info = {}
    for key in INDICATOR_WEIGHTS.keys():
        indicators_info[key] = {
            'display_name': get_indicator_display_name(key),
            'description': get_indicator_description(key),
            'color': get_indicator_color(key),
            'weight': INDICATOR_WEIGHTS[key]
        }
    return jsonify({'indicators': indicators_info})

@app.route('/api/history')
def get_history():
    """الحصول على السجل التاريخي"""
    limit = request.args.get('limit', 50, type=int)
    history = signals_data['history'][-limit:] if signals_data['history'] else []
    return jsonify({'history': history, 'total': len(signals_data['history'])})

# ======================
# تشغيل التطبيق
# ======================

if __name__ == '__main__':
    # حفظ وقت البدء
    global start_time
    start_time = time.time()
    
    # بدء التحديث في الخلفية
    print("=" * 60)
    print("🚀 بدء تشغيل Crypto Signal Analyzer - الإصدار 2.0")
    print("=" * 60)
    print(f"📊 مراقبة العملات: {[coin['name'] for coin in COINS]}")
    print(f"📈 نظام المؤشرات المتقدم مع 6 مؤشرات رئيسية")
    print(f"⚡ التحديث التلقائي كل 5 دقائق")
    print(f"🔔 نظام إشعارات متقدم مع NTFY")
    print("=" * 60)
    
    # تحديث أولي
    try:
        update_signals()
        print("✅ التحديث الأولي تم بنجاح")
    except Exception as e:
        print(f"❌ خطأ في التحديث الأولي: {e}")
    
    # بدء خيط التحديث التلقائي
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # تشغيل Flask
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 تشغيل الخادم على المنفذ {port}")
    print(f"🔧 وضع التصحيح: {'مفعل' if debug_mode else 'معطل'}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
