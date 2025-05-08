import React, { useState, useEffect } from 'react';
import MarketAnalysis from './MarketAnalysis';
import PositionManager from './PositionManager';
import AccountSummary from './AccountSummary';
import TradingControls from './TradingControls';
import Chart from './Chart';

const TradingDashboard = ({ marketData, socket }) => {
    const [selectedPair, setSelectedPair] = useState(null);
    const [tradingStatus, setTradingStatus] = useState({
        isTrading: false,
        activePairs: []
    });

    useEffect(() => {
        // Fetch initial trading status
        fetch('/api/trading/status')
            .then(res => res.json())
            .then(data => setTradingStatus(data))
            .catch(console.error);

        // Listen for trading status updates
        socket.on('trading_status_update', (data) => {
            setTradingStatus(data);
        });
    }, [socket]);

    const handlePairSelect = (pair) => {
        setSelectedPair(pair);
        socket.emit('subscribe', { pair });
    };

    return (
        <div className="dashboard">
            <div className="row">
                <div className="col-md-8">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Market Analysis</h5>
                        </div>
                        <div className="card-body">
                            {selectedPair && (
                                <>
                                    <Chart 
                                        pair={selectedPair}
                                        data={marketData[selectedPair]}
                                    />
                                    <MarketAnalysis 
                                        data={marketData[selectedPair]}
                                    />
                                </>
                            )}
                        </div>
                    </div>
                </div>
                <div className="col-md-4">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Trading Controls</h5>
                        </div>
                        <div className="card-body">
                            <TradingControls 
                                socket={socket}
                                tradingStatus={tradingStatus}
                            />
                        </div>
                    </div>
                    <div className="card mt-3">
                        <div className="card-header">
                            <h5 className="mb-0">Account Summary</h5>
                        </div>
                        <div className="card-body">
                            <AccountSummary 
                                data={marketData[selectedPair]?.account}
                            />
                        </div>
                    </div>
                </div>
            </div>
            <div className="row mt-3">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Open Positions</h5>
                        </div>
                        <div className="card-body">
                            <PositionManager 
                                positions={marketData[selectedPair]?.positions}
                                socket={socket}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TradingDashboard; 