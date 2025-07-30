import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';

// Import components
import Navigation from './components/Navigation';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Events from './pages/Events';
import About from './pages/About';

// Import services
import { apiService } from './services/apiService';

import './App.css';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

function App() {
  const [appData, setAppData] = useState({
    oilPrices: [],
    events: [],
    changePoints: [],
    loading: true,
    error: null
  });

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        setAppData(prev => ({ ...prev, loading: true, error: null }));
        
        // Load oil prices and events in parallel
        const [oilPricesResponse, eventsResponse, summaryResponse] = await Promise.all([
          apiService.getOilPrices({ limit: 1000 }),
          apiService.getEvents(),
          apiService.getDataSummary()
        ]);

        setAppData(prev => ({
          ...prev,
          oilPrices: oilPricesResponse.data || [],
          events: eventsResponse.events || [],
          summary: summaryResponse,
          loading: false
        }));

        // Load change points analysis
        try {
          const changePointsResponse = await apiService.getChangePoints();
          setAppData(prev => ({
            ...prev,
            changePoints: changePointsResponse.change_points || []
          }));
        } catch (error) {
          console.warn('Change points not available yet:', error.message);
        }

      } catch (error) {
        console.error('Error loading data:', error);
        setAppData(prev => ({
          ...prev,
          loading: false,
          error: error.message
        }));
      }
    };

    loadData();
  }, []);

  const updateAnalysisResults = async () => {
    try {
      const changePointsResponse = await apiService.getChangePoints();
      const correlationsResponse = await apiService.getEventCorrelations();
      
      setAppData(prev => ({
        ...prev,
        changePoints: changePointsResponse.change_points || [],
        correlations: correlationsResponse.correlations || []
      }));
    } catch (error) {
      console.error('Error updating analysis results:', error);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Navigation />
          
          <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
            <Routes>
              <Route 
                path="/" 
                element={
                  <Dashboard 
                    data={appData} 
                    onUpdateAnalysis={updateAnalysisResults}
                  />
                } 
              />
              <Route 
                path="/analysis" 
                element={
                  <Analysis 
                    data={appData} 
                    onUpdateAnalysis={updateAnalysisResults}
                  />
                } 
              />
              <Route 
                path="/events" 
                element={<Events data={appData} />} 
              />
              <Route 
                path="/about" 
                element={<About />} 
              />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;