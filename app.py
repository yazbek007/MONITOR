"""
Crypto Signal Analyzer Bot
Author: Crypto Analyst
Description: نظام تحليل مؤشرات الكريبتو مع واجهة Flask
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
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice
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
    {"symbol": "BNB/USDT", "name": "Binance Coin"},
    {"symbol": "SOL/USDT", "name": "Solana"}
]

# أوزان المؤشرات (20% لكل)
INDICATOR_WEIGHTS = {
    'fear_greed': 0.20,
    'rsi': 0.20,
    'volume': 0.20,
    'moving_averages': 0.20,
    'nvt': 0.20
}

# عتبات الإشعارات
NOTIFICATION_THRESHOLDS = {
    'strong_buy': 70,    # > 70% إشارة شراء قوية
    'strong_sell': 30,   # < 30% إشارة بيع قوية
    'change_threshold': 15  # تغير 15% لإرسال إشعار
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

class IndicatorsCalculator:
    """فئة لحساب المؤشرات"""
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """حساب مؤشر RSI"""
        rsi_indicator = RSIIndicator(close=df['close'], window=period)
        rsi = rsi_indicator.rsi().iloc[-1]
        
        # تحويل RSI إلى نسبة مئوية للإشارة
        if rsi <= 30:
            return 100  # تشبع بيعي قوي
        elif rsi >= 70:
            return 0    # تشبع شرائي قوي
        else:
            # تحويل خطي بين 30 و 70
            if rsi > 50:
                return max(0, 100 - ((rsi - 50) / 20 * 100))
            else:
                return min(100, ((50 - rsi) / 20 * 100))
    
    @staticmethod
    def calculate_volume_signal(df):
        """حساب إشارة الحجم"""
        current_volume = df['volume'].iloc[-1]
        avg_volume_20 = df['volume'].tail(20).mean()
        
        if current_volume > avg_volume_20 * 1.5:
            return 100  # حجم قوي جداً
        elif current_volume > avg_volume_20:
            return 75   # حجم أعلى من المتوسط
        elif current_volume < avg_volume_20 * 0.5:
            return 0    # حجم ضعيف جداً
        else:
            return 50   # حجم متوسط
    
    @staticmethod
    def calculate_moving_averages_signal(df):
        """حساب إشارة المتوسطات المتحركة"""
        # حساب المتوسطات
        ema_20 = EMAIndicator(close=df['close'], window=20).ema_indicator().iloc[-1]
        ema_50 = EMAIndicator(close=df['close'], window=50).ema_indicator().iloc[-1]
        ema_200 = EMAIndicator(close=df['close'], window=200).ema_indicator().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # تقييم الترتيب
        score = 0
        
        # سعر فوق EMA20
        if current_price > ema_20:
            score += 25
        
        # سعر فوق EMA50
        if current_price > ema_50:
            score += 25
        
        # سعر فوق EMA200
        if current_price > ema_200:
            score += 25
        
        # EMA20 فوق EMA50
        if ema_20 > ema_50:
            score += 15
        
        # EMA50 فوق EMA200
        if ema_50 > ema_200:
            score += 10
        
        return min(100, score)
    
    @staticmethod
    def calculate_fear_greed_index():
        """حساب مؤشر الخوف والجشع"""
        try:
            # جلب مؤشر الخوف والجشع من API
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fgi_value = int(data['data'][0]['value'])
                
                # تحويل إلى إشارة (0-100%)
                # 0-25: خوف شديد (شراء) -> 100%
                # 26-45: خوف -> 75%
                # 46-55: محايد -> 50%
                # 56-75: جشع -> 25%
                # 76-100: جشع شديد (بيع) -> 0%
                
                if fgi_value <= 25:
                    return 100, fgi_value
                elif fgi_value <= 45:
                    return 75, fgi_value
                elif fgi_value <= 55:
                    return 50, fgi_value
                elif fgi_value <= 75:
                    return 25, fgi_value
                else:
                    return 0, fgi_value
            else:
                return 50, 50  # قيمة افتراضية
        except:
            return 50, 50  # قيمة افتراضية في حالة الخطأ
    
    @staticmethod
    def calculate_nvt_signal(df, network_value):
        """حساب إشارة NVT (مبسطة)"""
        try:
            # متوسط الحجم اليومي (بالدولار)
            avg_daily_volume = df['volume'].tail(24).mean() * df['close'].iloc[-1]
            
            if avg_daily_volume == 0:
                return 50
            
            # نسبة NVT مبسطة
            nvt_ratio = network_value / avg_daily_volume
            
            # تحويل النسبة إلى إشارة
            # NVT منخفض = إيجابي، NVT مرتفع = سلبي
            if nvt_ratio < 20:
                return 100  # NVT منخفض جداً (إيجابي)
            elif nvt_ratio < 40:
                return 75
            elif nvt_ratio < 60:
                return 50
            elif nvt_ratio < 80:
                return 25
            else:
                return 0    # NVT مرتفع جداً (سلبي)
        except:
            return 50

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
            'signal_type': SignalProcessor.get_signal_type(total_percentage)
        }
    
    @staticmethod
    def get_signal_strength(percentage):
        """تحديد قوة الإشارة"""
        if percentage >= 80:
            return "قوية جداً"
        elif percentage >= 60:
            return "قوية"
        elif percentage >= 40:
            return "متوسطة"
        elif percentage >= 20:
            return "ضعيفة"
        else:
            return "ضعيفة جداً"
    
    @staticmethod
    def get_signal_type(percentage):
        """تحديد نوع الإشارة"""
        if percentage > 60:
            return "شراء"
        elif percentage < 40:
            return "بيع"
        else:
            return "محايد"

class NotificationManager:
    """مدير الإشعارات"""
    
    @staticmethod
    def check_and_send_notification(coin_data, previous_data):
        """التحقق وإرسال الإشعارات"""
        try:
            current_signal = coin_data['total_percentage']
            coin_symbol = coin_data['symbol']
            coin_name = coin_data['name']
            
            # البحث عن الإشارة السابقة
            prev_signal = None
            if previous_data:
                prev_signal = previous_data.get('total_percentage', None)
            
            # إشعارات قوة الإشارة
            message = None
            notification_type = None
            
            if current_signal >= NOTIFICATION_THRESHOLDS['strong_buy']:
                message = f"🟢 إشارة شراء قوية: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "strong_buy"
            
            elif current_signal <= NOTIFICATION_THRESHOLDS['strong_sell']:
                message = f"🔴 إشارة بيع قوية: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "strong_sell"
            
            # إشعار تغير كبير في الإشارة
            elif (prev_signal and 
                  abs(current_signal - prev_signal) >= NOTIFICATION_THRESHOLDS['change_threshold']):
                change = current_signal - prev_signal
                direction = "ارتفاع" if change > 0 else "انخفاض"
                message = f"📈 تغير كبير في إشارة {coin_name}"
                message += f"\n{current_signal:.1f}% ← {prev_signal:.1f}% ({direction})"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "significant_change"
            
            # إرسال الإشعار إذا كان هناك رسالة
            if message:
                success = NotificationManager.send_ntfy_notification(message)
                
                if success:
                    # حفظ الإشعار محلياً
                    notification = {
                        'timestamp': datetime.now(),
                        'coin': coin_name,
                        'message': message,
                        'type': notification_type,
                        'signal_strength': current_signal
                    }
                    
                    signals_data['notifications'].append(notification)
                    
                    # الحفاظ على آخر 20 إشعار فقط
                    if len(signals_data['notifications']) > 20:
                        signals_data['notifications'] = signals_data['notifications'][-20:]
                    
                    return True
            
            return False
        except Exception as e:
            print(f"Error in notification: {e}")
            return False
    
    @staticmethod
    def send_ntfy_notification(message):
        """إرسال إشعار عبر NTFY"""
        try:
            headers = {
                "Title": "🚀 Crypto Signal Alert",
                "Priority": "high",
                "Tags": "warning"
            }
            
            response = requests.post(
                NTFY_URL,
                data=message.encode('utf-8'),
                headers=headers
            )
            
            return response.status_code == 200
        except:
            return False

# ======================
# الوظائف الرئيسية
# ======================

def update_signals():
    """تحديث جميع الإشارات"""
    global signals_data
    
    print(f"[{datetime.now()}] تحديث الإشارات...")
    
    fetcher = BinanceDataFetcher()
    calculator = IndicatorsCalculator()
    
    # جلب مؤشر الخوف والجشع (مرة واحدة لجميع العملات)
    fear_greed_score, fgi_value = calculator.calculate_fear_greed_index()
    
    for coin in COINS:
        try:
            symbol = coin['symbol']
            name = coin['name']
            
            print(f"  جلب بيانات {name}...")
            
            # جلب البيانات
            df = fetcher.get_ohlcv(symbol, timeframe='1h', limit=500)
            if df is None or df.empty:
                continue
            
            current_price = fetcher.get_current_price(symbol)
            
            # حساب المؤشرات الفردية
            rsi_score = calculator.calculate_rsi(df)
            volume_score = calculator.calculate_volume_signal(df)
            ma_score = calculator.calculate_moving_averages_signal(df)
            
            # تقدير قيمة الشبكة (مبسط)
            network_value = current_price * 1_000_000  # تقدير مبسط
            
            nvt_score = calculator.calculate_nvt_signal(df, network_value)
            
            # جمع درجات المؤشرات
            indicator_scores = {
                'fear_greed': fear_greed_score / 100,
                'rsi': rsi_score / 100,
                'volume': volume_score / 100,
                'moving_averages': ma_score / 100,
                'nvt': nvt_score / 100
            }
            
            # حساب الإشارة المرجحة
            previous_data = signals_data['coins'].get(symbol, {})
            
            signal_result = SignalProcessor.calculate_weighted_signal(indicator_scores)
            
            # إعداد بيانات العملة
            coin_data = {
                'symbol': symbol,
                'name': name,
                'current_price': current_price,
                'price_change': 0,  # يمكن إضافة حساب التغير
                'indicator_scores': indicator_scores,
                'total_percentage': signal_result['total_percentage'],
                'signal_strength': signal_result['signal_strength'],
                'signal_type': signal_result['signal_type'],
                'weighted_scores': signal_result['weighted_scores'],
                'last_updated': datetime.now(),
                'fear_greed_value': fgi_value
            }
            
            # التحقق من الإشعارات
            NotificationManager.check_and_send_notification(coin_data, previous_data)
            
            # حفظ البيانات
            signals_data['coins'][symbol] = coin_data
            
        except Exception as e:
            print(f"Error processing {coin['name']}: {e}")
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
    
    print(f"[{datetime.now()}] تم تحديث الإشارات بنجاح")

def background_updater():
    """تحديث البيانات في الخلفية"""
    while True:
        update_signals()
        time.sleep(300)  # 5 دقائق

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
                    'raw_score': ind_data['raw_score'] * 100,
                    'weighted_score': ind_data['weighted_score'] * 100,
                    'percentage': ind_data['percentage']
                })
            
            coin_info['indicators'] = indicators
            coins_data.append(coin_info)
        else:
            # بيانات افتراضية
            coins_data.append({
                'symbol': coin['symbol'],
                'name': coin['name'],
                'current_price': 0,
                'total_percentage': 50,
                'signal_strength': 'غير متوفر',
                'signal_type': 'محايد',
                'indicators': [],
                'last_updated': None
            })
    
    # ترتيب العملات حسب قوة الإشارة
    coins_data.sort(key=lambda x: x['total_percentage'], reverse=True)
    
    # بيانات الإشعارات الأخيرة
    recent_notifications = signals_data['notifications'][-5:] if signals_data['notifications'] else []
    
    return render_template('index.html',
                         coins=coins_data,
                         last_update=signals_data['last_update'],
                         notifications=recent_notifications,
                         notification_count=len(signals_data['notifications']))

@app.route('/api/signals')
def api_signals():
    """API لإرجاع الإشارات"""
    return jsonify(signals_data['coins'])

@app.route('/api/update', methods=['POST'])
def manual_update():
    """تحديث يدوي للإشارات"""
    update_signals()
    return jsonify({'status': 'success', 'message': 'تم التحديث بنجاح'})

@app.route('/api/history/<symbol>')
def get_history(symbol):
    """الحصول على السجل التاريخي"""
    history_data = []
    
    for entry in signals_data['history']:
        if symbol in entry['signals']:
            history_data.append({
                'timestamp': entry['timestamp'].isoformat(),
                'signal': entry['signals'][symbol]
            })
    
    return jsonify(history_data)

def get_indicator_display_name(indicator_key):
    """تحويل اسم المؤشر للعرض"""
    names = {
        'fear_greed': 'مؤشر الخوف والجشع',
        'rsi': 'مؤشر RSI',
        'volume': 'الحجم التداولي',
        'moving_averages': 'المتوسطات المتحركة',
        'nvt': 'مؤشر NVT'
    }
    return names.get(indicator_key, indicator_key)

# ======================
# تشغيل التطبيق
# ======================

if __name__ == '__main__':
    # بدء التحديث في الخلفية
    print("🚀 بدء تشغيل Crypto Signal Analyzer...")
    
    # تحديث أولي
    update_signals()
    
    # بدء خيط التحديث التلقائي
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # تشغيل Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
