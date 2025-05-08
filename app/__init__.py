"""
Quantum Forex Trading System - App Package
"""

__version__ = "1.0.0"
__author__ = "Akashh-andro"

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

from app import routes 