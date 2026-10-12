import React from 'react';
import { NavLink } from 'react-router-dom';

export default function Sidebar({ health }) {
  return (
    <aside className="yt-sidebar d-none d-lg-block">
      <div className="mb-4">
        <NavLink
          to="/"
          className={({ isActive }) => `yt-nav-link ${isActive ? 'active' : ''}`}
        >
          <i className="bi bi-house-door-fill fs-5"></i>
          <span>Home Catalog</span>
        </NavLink>

        <NavLink
          to="/try-video"
          className={({ isActive }) => `yt-nav-link ${isActive ? 'active' : ''}`}
        >
          <i className="bi bi-cloud-arrow-up-fill fs-5"></i>
          <span>Try Your Video</span>
        </NavLink>

        <NavLink
          to="/personalized"
          className={({ isActive }) => `yt-nav-link ${isActive ? 'active' : ''}`}
        >
          <i className="bi bi-person-lines-fill fs-5"></i>
          <span>Personalized</span>
        </NavLink>

        <NavLink
          to="/visual-demo"
          className={({ isActive }) => `yt-nav-link ${isActive ? 'active' : ''}`}
        >
          <i className="bi bi-image-fill fs-5"></i>
          <span>Visual Similarity</span>
        </NavLink>

        <NavLink
          to="/diagnostics"
          className={({ isActive }) => `yt-nav-link ${isActive ? 'active' : ''}`}
        >
          <i className="bi bi-cpu-fill fs-5"></i>
          <span>System & Architecture</span>
        </NavLink>
      </div>

      <hr style={{ borderColor: 'var(--yt-border)' }} />

      {/* Production AI Specifications */}
      <div className="px-3 py-2 text-secondary small">
        <div className="fw-bold text-uppercase text-light mb-2" style={{ fontSize: '0.75rem', letterSpacing: '0.5px' }}>
          Production AI Weights
        </div>
        <div className="d-flex justify-content-between mb-1">
          <span>Text (Phase 8C)</span>
          <span className="text-light fw-bold">90%</span>
        </div>
        <div className="progress mb-2" style={{ height: '4px', backgroundColor: '#333' }}>
          <div className="progress-bar bg-primary" style={{ width: '90%' }}></div>
        </div>

        <div className="d-flex justify-content-between mb-1">
          <span>Metadata Filter</span>
          <span className="text-light fw-bold">10%</span>
        </div>
        <div className="progress mb-2" style={{ height: '4px', backgroundColor: '#333' }}>
          <div className="progress-bar bg-info" style={{ width: '10%' }}></div>
        </div>

        <div className="d-flex justify-content-between mb-1">
          <span>Visual (Phase 12C)</span>
          <span className="text-secondary fw-semibold">0% (Separate)</span>
        </div>
        <div className="progress mb-2" style={{ height: '4px', backgroundColor: '#333' }}>
          <div className="progress-bar bg-success" style={{ width: '0%' }}></div>
        </div>

        <div className="d-flex justify-content-between mb-1">
          <span>Audio (Phase 13)</span>
          <span className="text-secondary fw-semibold">0% (Extraction)</span>
        </div>
        <div className="progress mb-3" style={{ height: '4px', backgroundColor: '#333' }}>
          <div className="progress-bar bg-warning" style={{ width: '0%' }}></div>
        </div>

        <div className="p-2 rounded bg-dark border border-secondary border-opacity-25" style={{ fontSize: '0.72rem' }}>
          <div className="text-light fw-semibold mb-1">
            <i className="bi bi-database me-1 text-danger"></i> PostgreSQL + pgvector
          </div>
          <div>HNSW indexed: <span className="text-light fw-bold">{health?.index_size || 2983}</span> items</div>
          <div>Vector Dim: <span className="text-light fw-bold">512-D</span></div>
          <div>ANN Speedup: <span className="text-success fw-bold">10.3x vs Exact</span></div>
        </div>
      </div>
    </aside>
  );
}
