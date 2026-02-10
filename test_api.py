"""
نقاط نهاية الاختبار لبوت التنفيذ
"""

import os
import json
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request

test_bp = Blueprint('test', __name__)

@test_bp.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """نقطة نهاية اختبار بسيطة"""
    try:
        logger = logging.getLogger(__name__)
        
        if request.method == 'POST':
            data = request.get_json() or {}
            logger.info(f"📨 اختبار POST من: {data.get('source', 'unknown')}")
        else:
            logger.info("📨 اختبار GET")
        
        response = {
            'success': True,
            'message': '✅ نقطة الاختبار تعمل بنجاح',
            'status': 'active',
            'timestamp': datetime.now().isoformat(),
            'method': request.method,
            'bot': 'executor_bot',
            'version': 'v1.0'
        }
        
        if request.method == 'POST':
            response['received_data'] = data
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@test_bp.route('/api/ping', methods=['GET'])
def ping():
    """فحص بسيط للاتصال"""
    return jsonify({
        'success': True,
        'message': 'pong',
        'timestamp': datetime.now().isoformat()
    })

@test_bp.route('/api/echo', methods=['POST'])
def echo():
    """إعادة البيانات المرسلة"""
    try:
        data = request.get_json() or {}
        return jsonify({
            'success': True,
            'echo': data,
            'timestamp': datetime.now().isoformat()
        })
    except:
        return jsonify({
            'success': False,
            'message': 'Invalid JSON'
        }), 400
