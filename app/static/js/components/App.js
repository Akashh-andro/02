import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import TradingDashboard from './TradingDashboard';
import Sidebar from './Sidebar';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';

const App = () => {
    const [socket, setSocket] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [marketData, setMarketData] = useState({});
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Initialize socket connection
        const newSocket = io('http://localhost:5000');
        setSocket(newSocket);

        // Socket event handlers
        newSocket.on('connect', () => {
            setIsConnected(true);
            setLoading(false);
        });

        newSocket.on('disconnect', () => {
            setIsConnected(false);
        });

        newSocket.on('market_update', (data) => {
            setMarketData(prevData => ({
                ...prevData,
                [data.pair]: {
                    ...data,
                    timestamp: new Date(data.timestamp)
                }
            }));
        });

        newSocket.on('error', (error) => {
            setError(error.message);
        });

        // Cleanup on unmount
        return () => {
            newSocket.disconnect();
        };
    }, []);

    if (loading) {
        return <LoadingSpinner />;
    }

    if (error) {
        return (
            <div className="alert alert-danger" role="alert">
                Error: {error}
            </div>
        );
    }

    return (
        <ErrorBoundary>
            <div className="app-container">
                <Sidebar 
                    socket={socket}
                    isConnected={isConnected}
                />
                <main className="main-content">
                    <TradingDashboard 
                        marketData={marketData}
                        socket={socket}
                    />
                </main>
            </div>
        </ErrorBoundary>
    );
};

export default App; 