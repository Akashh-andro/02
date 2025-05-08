import os
import json
from typing import Dict, Any, Optional
import logging

class Config:
    def __init__(self, config_file: str = 'config.json'):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.config: Dict[str, Any] = {
            'trading': {
                'initial_capital': 10000,
                'max_positions': 5,
                'risk_per_trade': 0.02,
                'max_daily_risk': 0.05,
                'max_drawdown': 0.15,
                'max_correlation': 0.7
            },
            'exchange': {
                'name': None,
                'api_key': None,
                'api_secret': None,
                'testnet': True
            },
            'strategies': {
                'default': {
                    'type': 'MACD',
                    'parameters': {
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9
                    }
                }
            },
            'symbols': [],
            'timeframes': ['1h', '4h', '1d'],
            'logging': {
                'level': 'INFO',
                'file': 'trading.log'
            },
            'backtesting': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 10000
            }
        }
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.save_config()
                self.logger.info(f"Created new configuration file: {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.config['trading']
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange configuration"""
        return self.config['exchange']
    
    def get_strategy_config(self, strategy_name: str = 'default') -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.config['strategies'].get(strategy_name, self.config['strategies']['default'])
    
    def get_symbols(self) -> list:
        """Get list of trading symbols"""
        return self.config['symbols']
    
    def get_timeframes(self) -> list:
        """Get list of timeframes"""
        return self.config['timeframes']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config['logging']
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        return self.config['backtesting']
    
    def update_trading_config(self, config: Dict[str, Any]):
        """Update trading configuration"""
        self.config['trading'].update(config)
        self.save_config()
    
    def update_exchange_config(self, config: Dict[str, Any]):
        """Update exchange configuration"""
        self.config['exchange'].update(config)
        self.save_config()
    
    def update_strategy_config(self, strategy_name: str, config: Dict[str, Any]):
        """Update strategy configuration"""
        if strategy_name not in self.config['strategies']:
            self.config['strategies'][strategy_name] = {}
        self.config['strategies'][strategy_name].update(config)
        self.save_config()
    
    def add_symbol(self, symbol: str):
        """Add trading symbol"""
        if symbol not in self.config['symbols']:
            self.config['symbols'].append(symbol)
            self.save_config()
    
    def remove_symbol(self, symbol: str):
        """Remove trading symbol"""
        if symbol in self.config['symbols']:
            self.config['symbols'].remove(symbol)
            self.save_config()
    
    def add_timeframe(self, timeframe: str):
        """Add timeframe"""
        if timeframe not in self.config['timeframes']:
            self.config['timeframes'].append(timeframe)
            self.save_config()
    
    def remove_timeframe(self, timeframe: str):
        """Remove timeframe"""
        if timeframe in self.config['timeframes']:
            self.config['timeframes'].remove(timeframe)
            self.save_config()
    
    def set_logging_level(self, level: str):
        """Set logging level"""
        self.config['logging']['level'] = level
        self.save_config()
    
    def set_logging_file(self, file: str):
        """Set logging file"""
        self.config['logging']['file'] = file
        self.save_config()
    
    def set_backtesting_dates(self, start_date: str, end_date: str):
        """Set backtesting date range"""
        self.config['backtesting']['start_date'] = start_date
        self.config['backtesting']['end_date'] = end_date
        self.save_config()
    
    def set_backtesting_capital(self, capital: float):
        """Set backtesting initial capital"""
        self.config['backtesting']['initial_capital'] = capital
        self.save_config()
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Check required fields
            required_fields = {
                'trading': ['initial_capital', 'max_positions', 'risk_per_trade'],
                'exchange': ['name'],
                'strategies': ['default'],
                'logging': ['level', 'file'],
                'backtesting': ['start_date', 'end_date', 'initial_capital']
            }
            
            for section, fields in required_fields.items():
                if section not in self.config:
                    self.logger.error(f"Missing section: {section}")
                    return False
                
                for field in fields:
                    if field not in self.config[section]:
                        self.logger.error(f"Missing field: {section}.{field}")
                        return False
            
            # Validate values
            if self.config['trading']['risk_per_trade'] <= 0 or self.config['trading']['risk_per_trade'] > 1:
                self.logger.error("Invalid risk_per_trade value")
                return False
            
            if self.config['trading']['max_daily_risk'] <= 0 or self.config['trading']['max_daily_risk'] > 1:
                self.logger.error("Invalid max_daily_risk value")
                return False
            
            if self.config['trading']['max_drawdown'] <= 0 or self.config['trading']['max_drawdown'] > 1:
                self.logger.error("Invalid max_drawdown value")
                return False
            
            if self.config['trading']['max_correlation'] <= 0 or self.config['trading']['max_correlation'] > 1:
                self.logger.error("Invalid max_correlation value")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return False 