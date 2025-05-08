import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

const Chart = ({ pair, data }) => {
    const chartContainerRef = useRef();
    const chartRef = useRef();
    const candlestickSeriesRef = useRef();
    const volumeSeriesRef = useRef();

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Initialize chart
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 400,
            layout: {
                background: { color: '#ffffff' },
                textColor: '#333',
            },
            grid: {
                vertLines: { color: '#f0f0f0' },
                horzLines: { color: '#f0f0f0' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Create candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        // Create volume series
        const volumeSeries = chart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
            scaleMargins: {
                top: 0.8,
                bottom: 0,
            },
        });

        // Store references
        chartRef.current = chart;
        candlestickSeriesRef.current = candlestickSeries;
        volumeSeriesRef.current = volumeSeries;

        // Handle resize
        const handleResize = () => {
            chart.applyOptions({
                width: chartContainerRef.current.clientWidth,
            });
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    useEffect(() => {
        if (!data || !candlestickSeriesRef.current || !volumeSeriesRef.current) return;

        // Update chart data
        const candleData = {
            time: new Date(data.timestamp).getTime() / 1000,
            open: data.analysis.open,
            high: data.analysis.high,
            low: data.analysis.low,
            close: data.analysis.close,
        };

        const volumeData = {
            time: new Date(data.timestamp).getTime() / 1000,
            value: data.analysis.volume,
            color: data.analysis.close >= data.analysis.open ? '#26a69a' : '#ef5350',
        };

        candlestickSeriesRef.current.update(candleData);
        volumeSeriesRef.current.update(volumeData);
    }, [data]);

    return (
        <div className="chart-container">
            <div ref={chartContainerRef} />
        </div>
    );
};

export default Chart; 