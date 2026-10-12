import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import VideoDetailPage from './pages/VideoDetailPage';
import PersonalizedPage from './pages/PersonalizedPage';
import VisualDemoPage from './pages/VisualDemoPage';
import DiagnosticsPage from './pages/DiagnosticsPage';
import TryVideoPage from './pages/TryVideoPage';
import api from './services/api';

export default function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    // Initial health check
    api.getHealth()
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: 'offline', index_size: 2983 }));
  }, []);

  return (
    <Router>
      <div className="d-flex flex-column min-vh-100" style={{ backgroundColor: 'var(--yt-bg)' }}>
        {/* Top Navbar */}
        <Navbar health={health} />

        {/* Content Layout: Sidebar + Main Stream */}
        <div className="d-flex flex-grow-1">
          <Sidebar health={health} />

          <main className="flex-grow-1 p-3 p-md-4 pb-5 pb-lg-4" style={{ minWidth: 0, overflowX: 'hidden' }}>
            <div className="container-fluid px-0 px-md-2" style={{ maxWidth: '1600px' }}>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/video/:videoId" element={<VideoDetailPage />} />
                <Route path="/personalized" element={<PersonalizedPage />} />
                <Route path="/visual-demo" element={<VisualDemoPage />} />
                <Route path="/diagnostics" element={<DiagnosticsPage />} />
                <Route path="/try-video" element={<TryVideoPage />} />
              </Routes>
            </div>
          </main>
        </div>

        {/* Mobile Bottom Navigation Bar */}
        <nav className="d-lg-none fixed-bottom bg-black border-top border-secondary border-opacity-25 py-2 px-3 d-flex justify-content-around z-3" style={{ backdropFilter: 'blur(10px)' }}>
          <NavLink to="/" className={({ isActive }) => `text-decoration-none text-center ${isActive ? 'text-danger' : 'text-secondary'}`}>
            <i className="bi bi-house-door-fill d-block fs-5"></i>
            <span style={{ fontSize: '0.68rem' }}>Home</span>
          </NavLink>
          <NavLink to="/try-video" className={({ isActive }) => `text-decoration-none text-center ${isActive ? 'text-danger' : 'text-secondary'}`}>
            <i className="bi bi-cloud-arrow-up-fill d-block fs-5"></i>
            <span style={{ fontSize: '0.68rem' }}>Upload</span>
          </NavLink>
          <NavLink to="/personalized" className={({ isActive }) => `text-decoration-none text-center ${isActive ? 'text-danger' : 'text-secondary'}`}>
            <i className="bi bi-person-fill d-block fs-5"></i>
            <span style={{ fontSize: '0.68rem' }}>For You</span>
          </NavLink>
          <NavLink to="/visual-demo" className={({ isActive }) => `text-decoration-none text-center ${isActive ? 'text-danger' : 'text-secondary'}`}>
            <i className="bi bi-image-fill d-block fs-5"></i>
            <span style={{ fontSize: '0.68rem' }}>Visual</span>
          </NavLink>
          <NavLink to="/diagnostics" className={({ isActive }) => `text-decoration-none text-center ${isActive ? 'text-danger' : 'text-secondary'}`}>
            <i className="bi bi-cpu-fill d-block fs-5"></i>
            <span style={{ fontSize: '0.68rem' }}>System</span>
          </NavLink>
        </nav>
      </div>
    </Router>
  );
}
