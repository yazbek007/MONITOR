import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Binance API
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # NTFY
    NTFY_TOPIC = os.getenv('NTFY_TOPIC', 'crypto_signals_alerts')
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # App Settings
    UPDATE_INTERVAL = 300  # 5 دقائق بالثواني
    COINS_TO_MONITOR = [
        'BTC/USDT',
        'ETH/USDT', 
        'BNB/USDT',
        'SOL/USDT'
    ]
