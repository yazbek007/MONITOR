"""
Crypto Signal Analyzer Bot - نسخة محدثة
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
import talib  # استخدام talib بدلاً من ta
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
    'strong_buy': 70,
    'strong_sell': 30,
    'change_threshold': 15
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
    """فئة لحساب المؤشرات باستخدام TA-Lib"""
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """حساب مؤشر RSI باستخدام TA-Lib"""
        try:
            rsi = talib.RSI(df['close'].values, timeperiod=period)[-1]
            
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
        except:
            return 50  # قيمة افتراضية في حالة الخطأ
    
    @staticmethod
    def calculate_volume_signal(df):
        """حساب إشارة الحجم"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume_20 = df['volume'].tail(20).mean()
            
            if avg_volume_20 == 0:
                return 50
            
            volume_ratio = current_volume / avg_volume_20
            
            if volume_ratio > 1.5:
                return 100
            elif volume_ratio > 1.0:
                return 75
            elif volume_ratio < 0.5:
                return 0
            else:
                return 50
        except:
            return 50
    
    @staticmethod
    def calculate_moving_averages_signal(df):
        """حساب إشارة المتوسطات المتحركة"""
        try:
            # حساب المتوسطات باستخدام TA-Lib
            ema_20 = talib.EMA(df['close'].values, timeperiod=20)[-1]
            ema_50 = talib.EMA(df['close'].values, timeperiod=50)[-1]
            ema_200 = talib.EMA(df['close'].values, timeperiod=200)[-1]
            current_price = df['close'].iloc[-1]
            
            # تقييم الترتيب
            score = 0
            
            if pd.notna(ema_20) and current_price > ema_20:
                score += 25
            
            if pd.notna(ema_50) and current_price > ema_50:
                score += 25
            
            if pd.notna(ema_200) and current_price > ema_200:
                score += 25
            
            if pd.notna(ema_20) and pd.notna(ema_50) and ema_20 > ema_50:
                score += 15
            
            if pd.notna(ema_50) and pd.notna(ema_200) and ema_50 > ema_200:
                score += 10
            
            return min(100, score)
        except Exception as e:
            print(f"Error in MA calculation: {e}")
            return 50
    
    @staticmethod
    def calculate_fear_greed_index():
        """حساب مؤشر الخوف والجشع"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fgi_value = int(data['data'][0]['value'])
                
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
                return 50, 50
        except:
            return 50, 50
    
    @staticmethod
    def calculate_nvt_signal(df, current_price):
        """حساب إشارة NVT (مبسطة)"""
        try:
            # متوسط الحجم اليومي (بالدولار)
            avg_daily_volume = df['volume'].tail(24).mean() * current_price
            
            if avg_daily_volume == 0:
                return 50
            
            # استخدام القيمة السوقية التقريبية (مبسطة)
            # يمكن تعديل هذه القيمة حسب العملة
            if "BTC" in df.index.name or "BTC" in str(df.columns):
                market_cap = current_price * 19_000_000  # تقدير تقريبي
            elif "ETH" in df.index.name or "ETH" in str(df.columns):
                market_cap = current_price * 120_000_000
            elif "BNB" in df.index.name or "BNB" in str(df.columns):
                market_cap = current_price * 150_000_000
            elif "SOL" in df.index.name or "SOL" in str(df.columns):
                market_cap = current_price * 400_000_000
            else:
                market_cap = current_price * 1_000_000
            
            nvt_ratio = market_cap / avg_daily_volume
            
            if nvt_ratio < 20:
                return 100
            elif nvt_ratio < 40:
                return 75
            elif nvt_ratio < 60:
                return 50
            elif nvt_ratio < 80:
                return 25
            else:
                return 0
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
            
            prev_signal = None
            if previous_data:
                prev_signal = previous_data.get('total_percentage', None)
            
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
            
            elif (prev_signal and 
                  abs(current_signal - prev_signal) >= NOTIFICATION_THRESHOLDS['change_threshold']):
                change = current_signal - prev_signal
                direction = "ارتفاع" if change > 0 else "انخفاض"
                message = f"📈 تغير كبير في إشارة {coin_name}"
                message += f"\n{current_signal:.1f}% ← {prev_signal:.1f}% ({direction})"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "significant_change"
            
            if message:
                success = NotificationManager.send_ntfy_notification(message)
                
                if success:
                    notification = {
                        'timestamp': datetime.now(),
                        'coin': coin_name,
                        'message': message,
                        'type': notification_type,
                        'signal_strength': current_signal
                    }
                    
                    signals_data['notifications'].append(notification)
                    
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
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
        except:
            return False

# ======================
# الوظائف المساعدة
# ======================

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

def get_indicator_color(indicator_key):
    """الحصول على لون المؤشر"""
    colors = {
        'fear_greed': '#2E86AB',
        'rsi': '#A23B72',
        'volume': '#3BB273',
        'moving_averages': '#F18F01',
        'nvt': '#6C757D'
    }
    return colors.get(indicator_key, '#2E86AB')

# ======================
# الوظائف الرئيسية
# ======================

def update_signals():
    """تحديث جميع الإشارات"""
    global signals_data
    
    print(f"[{datetime.now()}] تحديث الإشارات...")
    
    fetcher = BinanceDataFetcher()
    calculator = IndicatorsCalculator()
    
    # جلب مؤشر الخوف والجشع
    fear_greed_score, fgi_value = calculator.calculate_fear_greed_index()
    
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
            
            current_price = fetcher.get_current_price(symbol)
            
            # حساب المؤشرات
            rsi_score = calculator.calculate_rsi(df)
            volume_score = calculator.calculate_volume_signal(df)
            ma_score = calculator.calculate_moving_averages_signal(df)
            nvt_score = calculator.calculate_nvt_signal(df, current_price)
            
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
            
            print(f"  {name}: {signal_result['total_percentage']:.1f}% ({signal_result['signal_type']})")
            
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
    
    # الحفاظ على آخر 50 سجل
    if len(signals_data['history']) > 50:
        signals_data['history'] = signals_data['history'][-50:]
    
    print(f"[{datetime.now()}] تم تحديث الإشارات بنجاح")

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
                'last_updated': None,
                'fear_greed_value': 50
            })
    
    # ترتيب العملات حسب قوة الإشارة
    coins_data.sort(key=lambda x: x['total_percentage'], reverse=True)
    
    # بيانات الإشعارات الأخيرة
    recent_notifications = signals_data['notifications'][-5:] if signals_data['notifications'] else []
    
    # تمرير دالتي الألوان والأسماء للقالب
    return render_template('index.html',
                         coins=coins_data,
                         last_update=signals_data['last_update'],
                         notifications=recent_notifications,
                         notification_count=len(signals_data['notifications']),
                         get_indicator_color=get_indicator_color)

@app.route('/api/signals')
def api_signals():
    """API لإرجاع الإشارات"""
    return jsonify(signals_data['coins'])

@app.route('/api/update', methods=['POST'])
def manual_update():
    """تحديث يدوي للإشارات"""
    update_signals()
    return jsonify({'status': 'success', 'message': 'تم التحديث بنجاح'})

@app.route('/api/health')
def health_check():
    """فحص صحة التطبيق"""
    return jsonify({
        'status': 'healthy',
        'last_update': signals_data['last_update'].isoformat() if signals_data['last_update'] else None,
        'coins_available': len(signals_data['coins'])
    })

# ======================
# تشغيل التطبيق
# ======================

if __name__ == '__main__':
    # بدء التحديث في الخلفية
    print("🚀 بدء تشغيل Crypto Signal Analyzer...")
    print(f"📊 مراقبة العملات: {[coin['name'] for coin in COINS]}")
    
    # تحديث أولي
    try:
        update_signals()
    except Exception as e:
        print(f"Error in initial update: {e}")
    
    # بدء خيط التحديث التلقائي
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # تشغيل Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
