from flask import jsonify, request
from app import app, socketio
from main import ForexTradingSystem
import json
from datetime import datetime
import threading
import time
from queue import Queue
import asyncio

# Initialize trading system
trading_system = ForexTradingSystem()

# Data streaming queue
data_queue = Queue()

def background_stream_data():
    """Background task to stream market data"""
    while True:
        try:
            for pair in trading_system.active_pairs:
                analysis = trading_system.signal_generator.analyze_market(pair)
                signal = trading_system.signal_generator.generate_signal(analysis, pair)
                
                data = {
                    'pair': pair,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis,
                    'signal': signal,
                    'positions': trading_system.get_open_positions(),
                    'account': trading_system.get_account_info()
                }
                
                socketio.emit('market_update', data)
            time.sleep(1)  # Update every second
        except Exception as e:
            print(f"Error in data streaming: {str(e)}")
            time.sleep(5)  # Wait before retrying

# Start background task
stream_thread = threading.Thread(target=background_stream_data, daemon=True)
stream_thread.start()

@app.route('/api/config', methods=['GET'])
def get_config():
    with open('config.json', 'r') as f:
        return jsonify(json.load(f))

@app.route('/api/pairs', methods=['GET'])
def get_pairs():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return jsonify(config['symbols'])

@app.route('/api/analysis/<pair>', methods=['GET'])
def get_analysis(pair):
    try:
        analysis = trading_system.signal_generator.analyze_market(pair)
        signal = trading_system.signal_generator.generate_signal(analysis, pair)
        
        return jsonify({
            'indicators': {
                'rsi': analysis['rsi'],
                'macd': analysis['macd'],
                'sma_short': analysis['sma_short'],
                'sma_medium': analysis['sma_medium'],
                'sma_long': analysis['sma_long'],
                'bb_upper': analysis.get('bb_upper'),
                'bb_lower': analysis.get('bb_lower'),
                'atr': analysis.get('atr'),
                'adx': analysis.get('adx')
            },
            'signal': signal,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    try:
        positions = trading_system.get_open_positions()
        return jsonify({
            'positions': positions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account', methods=['GET'])
def get_account():
    try:
        account_info = trading_system.get_account_info()
        return jsonify({
            'account': account_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    try:
        data = request.get_json()
        pairs = data.get('pairs', [])
        risk_per_trade = data.get('risk_per_trade', 0.02)
        max_daily_risk = data.get('max_daily_risk', 0.05)
        
        # Update risk parameters
        trading_system.risk_manager.risk_per_trade = risk_per_trade
        trading_system.risk_manager.max_daily_risk = max_daily_risk
        
        trading_system.start_trading(pairs)
        return jsonify({
            'status': 'success',
            'message': f'Trading started for pairs: {", ".join(pairs)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    try:
        trading_system.stop_trading()
        return jsonify({
            'status': 'success',
            'message': 'Trading stopped successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/status', methods=['GET'])
def get_trading_status():
    try:
        return jsonify({
            'is_trading': trading_system.is_trading,
            'active_pairs': trading_system.active_pairs,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial data
    socketio.emit('connection_established', {
        'status': 'connected',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('subscribe')
def handle_subscribe(data):
    pair = data.get('pair')
    if pair:
        if pair not in trading_system.active_pairs:
            trading_system.active_pairs.append(pair)
        socketio.emit('subscription_confirmed', {
            'pair': pair,
            'status': 'subscribed',
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    pair = data.get('pair')
    if pair and pair in trading_system.active_pairs:
        trading_system.active_pairs.remove(pair)
        socketio.emit('unsubscription_confirmed', {
            'pair': pair,
            'status': 'unsubscribed',
            'timestamp': datetime.now().isoformat()
        }) 