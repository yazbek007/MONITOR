"""
Crypto Signal Analyzer Bot - نسخة خالية من التبعيات المعقدة
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

# أوزان المؤشرات
INDICATOR_WEIGHTS = {
    'fear_greed': 0.20,
    'rsi': 0.20,
    'volume': 0.20,
    'moving_averages': 0.20,
    'price_action': 0.20
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
    """فئة لحساب المؤشرات باستخدام pandas/numpy فقط"""
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """حساب مؤشر RSI يدوياً"""
        try:
            # حساب تغيرات السعر
            delta = df['close'].diff()
            
            # فصل المكاسب والخسائر
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # حساب متوسط المكاسب والخسائر
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # حساب RS وRSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
            
            if pd.isna(rsi_value):
                return 50
            
            # تحويل RSI إلى درجة من 0-100
            if rsi_value <= 30:
                return 100  # تشبع بيعي قوي
            elif rsi_value >= 70:
                return 0    # تشبع شرائي قوي
            else:
                # تحويل خطي بين 30 و 70
                if rsi_value > 50:
                    return max(0, 100 - ((rsi_value - 50) / 20 * 100))
                else:
                    return min(100, ((50 - rsi_value) / 20 * 100))
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return 50
    
    @staticmethod
    def calculate_volume_signal(df):
        """حساب إشارة الحجم المعدلة"""
        try:
            if len(df) < 20:
                return 50
        
            current_volume = df['volume'].iloc[-1]
            avg_volume_20 = df['volume'].tail(20).mean()
        
            if avg_volume_20 == 0 or current_volume == 0:
                return 50
        
            volume_ratio = current_volume / avg_volume_20
        
            # توزيع أكثر منطقية
            if volume_ratio > 2.0:
                return 100  # حجم عالي جداً
            elif volume_ratio > 1.5:
                return 85   # حجم عالي
            elif volume_ratio > 1.2:
                return 70   # حجم أعلى من المتوسط
            elif volume_ratio > 0.8:
                return 55   # حجم طبيعي
            elif volume_ratio > 0.5:
                return 40   # حجم أقل من المتوسط
            elif volume_ratio > 0.3:
                return 25   # حجم منخفض
            else:
                return 10   # حجم منخفض جداً
        except:
            return 50
    
    @staticmethod
    def calculate_moving_averages_signal(df):
        """حساب إشارة المتوسطات المتحركة المعدلة"""
        try:
            if len(df) < 200:
                return 50
        
            # حساب المتوسطات
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()
            sma_200 = df['close'].rolling(window=200).mean()
        
            sma_20_value = sma_20.iloc[-1]
            sma_50_value = sma_50.iloc[-1]
            sma_200_value = sma_200.iloc[-1]
        
            current_price = df['close'].iloc[-1]
        
            # حساب المسافات النسبية
            score = 50  # نقطة بداية متوسطة
        
            # السعر مقابل المتوسطات (40 نقطة)
            if pd.notna(sma_20_value):
                distance_20 = ((current_price - sma_20_value) / sma_20_value) * 100
                if distance_20 > 5:
                    score += 15
                elif distance_20 > 2:
                    score += 10
                elif distance_20 > 0:
                    score += 5
                elif distance_20 > -2:
                    score -= 5
                elif distance_20 > -5:
                    score -= 10
                else:
                    score -= 15
        
            if pd.notna(sma_50_value):
                distance_50 = ((current_price - sma_50_value) / sma_50_value) * 100
                if distance_50 > 5:
                    score += 10
                elif distance_50 > 2:
                    score += 7
                elif distance_50 > 0:
                    score += 3
                elif distance_50 > -2:
                    score -= 3
                elif distance_50 > -5:
                    score -= 7
                else:
                    score -= 10
        
            if pd.notna(sma_200_value):
                distance_200 = ((current_price - sma_200_value) / sma_200_value) * 100
                if distance_200 > 5:
                    score += 15
                elif distance_200 > 2:
                    score += 10
                elif distance_200 > 0:
                    score += 5
                elif distance_200 > -2:
                    score -= 5
                elif distance_200 > -5:
                    score -= 10
                else:
                    score -= 15
        
            # الترتيب (20 نقطة)
            if pd.notna(sma_20_value) and pd.notna(sma_50_value):
                if sma_20_value > sma_50_value:
                    score += 10
                else:
                    score -= 5
        
            if pd.notna(sma_50_value) and pd.notna(sma_200_value):
                if sma_50_value > sma_200_value:
                    score += 10
                else:
                    score -= 5
        
            # التأكد من النتيجة بين 0-100
            return max(0, min(100, score))
        
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
                
                # تحويل إلى درجة 0-100
                if fgi_value <= 25:
                    return 100, fgi_value  # خوف شديد = إشارة شراء قوية
                elif fgi_value <= 45:
                    return 75, fgi_value   # خوف
                elif fgi_value <= 55:
                    return 50, fgi_value   # محايد
                elif fgi_value <= 75:
                    return 25, fgi_value   # جشع
                else:
                    return 0, fgi_value    # جشع شديد = إشارة بيع
            else:
                return 50, 50
        except:
            return 50, 50
    
    @staticmethod
    def calculate_price_action_signal(df):
        """حساب إشارة حركة السعر"""
        try:
            current_price = df['close'].iloc[-1]
            low_20 = df['low'].tail(20).min()
            high_20 = df['high'].tail(20).max()
            
            # حساب موضع السعر في النطاق
            if high_20 != low_20:
                position = (current_price - low_20) / (high_20 - low_20) * 100
            else:
                position = 50
            
            # تحليل الشموع
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            score = position  # البدء من موضع السعر
            
            # شمعة صاعدة قوية
            if last_candle['close'] > last_candle['open']:
                score += 10
            
            # شمعة أكبر من سابقتها
            body_size = abs(last_candle['close'] - last_candle['open'])
            prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
            
            if body_size > prev_body_size:
                score += 5
            
            # إغلاق أعلى من فتح
            if last_candle['close'] > last_candle['open'] and last_candle['close'] > prev_candle['close']:
                score += 15
            
            return min(100, max(0, score))
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
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "strong_buy"
            
            elif current_signal <= NOTIFICATION_THRESHOLDS['strong_sell']:
                message = f"🔴 إشارة بيع قوية: {coin_name} ({coin_symbol})"
                message += f"\n📊 القوة: {current_signal:.1f}%"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
                message += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                notification_type = "strong_sell"
            
            elif (prev_signal and 
                  abs(current_signal - prev_signal) >= NOTIFICATION_THRESHOLDS['change_threshold']):
                change = current_signal - prev_signal
                direction = "ارتفاع" if change > 0 else "انخفاض"
                message = f"📈 تغير كبير في إشارة {coin_name}"
                message += f"\n{current_signal:.1f}% ← {prev_signal:.1f}% ({direction})"
                message += f"\n💰 السعر: ${coin_data.get('current_price', 0):,.2f}"
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
        'price_action': 'حركة السعر'
    }
    return names.get(indicator_key, indicator_key)

def get_indicator_color(indicator_key):
    """الحصول على لون المؤشر"""
    colors = {
        'fear_greed': '#2E86AB',
        'rsi': '#A23B72',
        'volume': '#3BB273',
        'moving_averages': '#F18F01',
        'price_action': '#6C757D'
    }
    return colors.get(indicator_key, '#2E86AB')

def format_number(value):
    """تنسيق الأرقام للعرض"""
    try:
        if value is None:
            return "0"
        if value >= 1000000:
            return f"{value/1000000:.2f}M"
        elif value >= 1000:
            return f"{value/1000:.2f}K"
        else:
            return f"{value:.2f}"
    except:
        return "0"

def format_percentage(value):
    """تنسيق النسب المئوية"""
    try:
        if value is None:
            return "0%"
        return f"{value:.1f}%"
    except:
        return "0%"

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
            price_action_score = calculator.calculate_price_action_signal(df)
            
            # جمع درجات المؤشرات
            indicator_scores = {
                'fear_greed': fear_greed_score / 100,
                'rsi': rsi_score / 100,
                'volume': volume_score / 100,
                'moving_averages': ma_score / 100,
                'price_action': price_action_score / 100
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
                'fear_greed_value': fgi_value,
                'price_change': None
            }
            
            # حساب التغير إذا كانت هناك بيانات سابقة
            if previous_data and 'current_price' in previous_data:
                prev_price = previous_data['current_price']
                if prev_price > 0:
                    price_change = ((current_price - prev_price) / prev_price) * 100
                    coin_data['price_change'] = price_change
            
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
            coin_info['formatted_price'] = format_number(coin_info['current_price'])
            coin_info['formatted_change'] = format_percentage(coin_info.get('price_change', 0))
            
            coins_data.append(coin_info)
        else:
            # بيانات افتراضية
            coins_data.append({
                'symbol': coin['symbol'],
                'name': coin['name'],
                'current_price': 0,
                'formatted_price': '0',
                'total_percentage': 50,
                'signal_strength': 'غير متوفر',
                'signal_type': 'محايد',
                'indicators': [],
                'last_updated': None,
                'fear_greed_value': 50,
                'price_change': 0,
                'formatted_change': '0%'
            })
    
    # ترتيب العملات حسب قوة الإشارة
    coins_data.sort(key=lambda x: x['total_percentage'], reverse=True)
    
    # بيانات الإشعارات الأخيرة
    recent_notifications = signals_data['notifications'][-5:] if signals_data['notifications'] else []
    
    # إحصائيات
    total_signals = [c['total_percentage'] for c in coins_data if c['total_percentage'] > 0]
    stats = {
        'total_coins': len(COINS),
        'updated_coins': len(signals_data['coins']),
        'avg_signal': np.mean(total_signals) if total_signals else 50,
        'buy_signals': sum(1 for c in coins_data if c.get('signal_type') == 'شراء'),
        'sell_signals': sum(1 for c in coins_data if c.get('signal_type') == 'بيع'),
        'neutral_signals': sum(1 for c in coins_data if c.get('signal_type') == 'محايد')
    }
    
    return render_template('index.html',
                         coins=coins_data,
                         last_update=signals_data['last_update'],
                         notifications=recent_notifications,
                         notification_count=len(signals_data['notifications']),
                         stats=stats,
                         get_indicator_color=get_indicator_color,
                         format_number=format_number,
                         format_percentage=format_percentage)

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
        'coins_available': len(signals_data['coins']),
        'uptime': time.time() - start_time if 'start_time' in globals() else 0
    })

@app.route('/api/notifications')
def get_notifications():
    """الحصول على الإشعارات"""
    return jsonify({'notifications': signals_data['notifications'][-10:]})

@app.route('/api/coins')
def get_coins():
    """الحصول على قائمة العملات"""
    return jsonify({'coins': COINS})

# ======================
# تشغيل التطبيق
# ======================

if __name__ == '__main__':
    # حفظ وقت البدء
    global start_time
    start_time = time.time()
    
    # بدء التحديث في الخلفية
    print("🚀 بدء تشغيل Crypto Signal Analyzer...")
    print(f"📊 مراقبة العملات: {[coin['name'] for coin in COINS]}")
    print(f"⚡ نظام المؤشرات المدمج - بدون مكتبات خارجية")
    print(f"📈 المؤشرات المستخدمة: RSI، الحجم، المتوسطات المتحركة، حركة السعر، مؤشر الخوف والجشع")
    
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
    print(f"🌐 تشغيل الخادم على المنفذ {port}")
    print(f"⏰ التحديث التلقائي كل 5 دقائق")
    app.run(host='0.0.0.0', port=port, debug=False)
