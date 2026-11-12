import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuroraProvider } from './context/AuroraContext';
import { Layout } from './components/Layout';
import { LandingPage } from './pages/LandingPage';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetUploadPage } from './pages/DatasetUploadPage';
import { DatasetOverviewPage } from './pages/DatasetOverviewPage';
import { ModelSelectionPage } from './pages/ModelSelectionPage';
import { ProcessingPage } from './pages/ProcessingPage';
import { ResultsDashboardPage } from './pages/ResultsDashboardPage';
import { ModelComparisonPage } from './pages/ModelComparisonPage';
import { HyperparameterTuningPage } from './pages/HyperparameterTuningPage';

export const App: React.FC = () => {
  return (
    <AuroraProvider>
      <BrowserRouter>
        <Routes>
          {/* Dedicated Landing Page */}
          <Route path="/" element={<LandingPage />} />

          {/* Main Application Screens with Unified Layout */}
          <Route
            path="/dashboard"
            element={
              <Layout>
                <DashboardPage />
              </Layout>
            }
          />
          <Route
            path="/upload"
            element={
              <Layout>
                <DatasetUploadPage />
              </Layout>
            }
          />
          <Route
            path="/overview"
            element={
              <Layout>
                <DatasetOverviewPage />
              </Layout>
            }
          />
          <Route
            path="/select-approach"
            element={
              <Layout>
                <ModelSelectionPage />
              </Layout>
            }
          />
          <Route
            path="/processing"
            element={
              <Layout>
                <ProcessingPage />
              </Layout>
            }
          />
          <Route
            path="/results"
            element={
              <Layout>
                <ResultsDashboardPage />
              </Layout>
            }
          />
          <Route
            path="/comparison"
            element={
              <Layout>
                <ModelComparisonPage />
              </Layout>
            }
          />
          <Route
            path="/tuning"
            element={
              <Layout>
                <HyperparameterTuningPage />
              </Layout>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuroraProvider>
  );
};

export default App;

