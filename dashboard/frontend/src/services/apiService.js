/**
 * API Service for communicating with the Flask backend
 * Provides methods to fetch oil price data, events, and analysis results
 */

import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  timeout: 30000, // 30 seconds timeout for analysis operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.error || error.response.statusText;
      throw new Error(`API Error (${error.response.status}): ${message}`);
    } else if (error.request) {
      // Network error
      throw new Error('Network Error: Unable to connect to the server');
    } else {
      // Other error
      throw new Error(`Request Error: ${error.message}`);
    }
  }
);

export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Data endpoints
  getOilPrices: (params = {}) => {
    const queryParams = new URLSearchParams();
    if (params.start_date) queryParams.append('start_date', params.start_date);
    if (params.end_date) queryParams.append('end_date', params.end_date);
    if (params.limit) queryParams.append('limit', params.limit.toString());
    
    const url = `/data/oil-prices${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return api.get(url);
  },

  getEvents: (params = {}) => {
    const queryParams = new URLSearchParams();
    if (params.category) queryParams.append('category', params.category);
    if (params.start_date) queryParams.append('start_date', params.start_date);
    if (params.end_date) queryParams.append('end_date', params.end_date);
    
    const url = `/data/events${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return api.get(url);
  },

  getDataSummary: () => api.get('/data/summary'),

  // Analysis endpoints
  getChangePoints: (params = {}) => {
    const queryParams = new URLSearchParams();
    if (params.model_type) queryParams.append('model_type', params.model_type);
    if (params.confidence) queryParams.append('confidence', params.confidence.toString());
    
    const url = `/analysis/change-points${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return api.get(url);
  },

  getEventCorrelations: () => api.get('/analysis/correlations'),

  runAnalysis: (params) => {
    return api.post('/analysis/run', params);
  },

  // Utility methods
  formatDate: (date) => {
    if (date instanceof Date) {
      return date.toISOString().split('T')[0];
    }
    return date;
  },

  // Data processing utilities
  processOilPriceData: (data) => {
    return data.map(item => ({
      ...item,
      Date: new Date(item.Date),
      Price: parseFloat(item.Price),
      Log_Returns: parseFloat(item.Log_Returns) || 0
    }));
  },

  processEventsData: (events) => {
    return events.map(event => ({
      ...event,
      Date: new Date(event.Date)
    }));
  },

  // Chart data preparation
  prepareChartData: (oilPrices, events = [], changePoints = []) => {
    const chartData = oilPrices.map(item => ({
      date: item.Date,
      price: item.Price,
      returns: item.Log_Returns
    }));

    // Add events as annotations
    const eventAnnotations = events.map(event => ({
      x: event.Date,
      text: event.Event,
      category: event.Category,
      impact: event.Expected_Impact
    }));

    // Add change points as vertical lines
    const changePointLines = changePoints.map(cp => ({
      x: new Date(cp.date),
      probability: cp.probability,
      type: cp.type
    }));

    return {
      data: chartData,
      events: eventAnnotations,
      changePoints: changePointLines
    };
  },

  // Analysis utilities
  calculateVolatility: (returns, window = 30) => {
    const result = [];
    for (let i = window - 1; i < returns.length; i++) {
      const slice = returns.slice(i - window + 1, i + 1);
      const mean = slice.reduce((sum, val) => sum + val, 0) / slice.length;
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / slice.length;
      const volatility = Math.sqrt(variance);
      result.push(volatility);
    }
    return result;
  },

  calculateMovingAverage: (data, window = 30) => {
    const result = [];
    for (let i = window - 1; i < data.length; i++) {
      const slice = data.slice(i - window + 1, i + 1);
      const average = slice.reduce((sum, val) => sum + val, 0) / slice.length;
      result.push(average);
    }
    return result;
  }
};

export default apiService;