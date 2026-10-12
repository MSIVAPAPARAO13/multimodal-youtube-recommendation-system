import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';

export default function Navbar({ health }) {
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/?q=${encodeURIComponent(searchQuery.trim())}`);
    } else {
      navigate('/');
    }
  };

  const isBackendOk = health?.status === 'ok';

  return (
    <nav className="navbar yt-navbar navbar-expand-lg">
      <div className="container-fluid d-flex align-items-center justify-content-between">
        {/* Brand */}
        <div className="d-flex align-items-center gap-3">
          <Link to="/" className="yt-logo-text">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="32" height="32">
              <rect width="48" height="48" rx="10" fill="#FF0000" />
              <polygon points="18,14 36,24 18,34" fill="#FFFFFF" />
            </svg>
            <span>Recom<span className="text-danger">Tube</span></span>
            <span className="yt-logo-badge">Multimodal AI</span>
          </Link>
        </div>

        {/* Center Search */}
        <div className="d-none d-md-flex justify-content-center flex-grow-1 mx-4">
          <form className="d-flex yt-search-container" onSubmit={handleSearch}>
            <input
              type="text"
              className="form-control yt-search-input"
              placeholder="Search 2,983 video catalog by title, topic..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button className="btn yt-search-btn" type="submit" title="Search">
              <i className="bi bi-search"></i>
            </button>
            {searchQuery && (
              <button
                type="button"
                className="btn btn-sm btn-link text-secondary position-absolute"
                style={{ right: '55px', top: '50%', transform: 'translateY(-50%)' }}
                onClick={() => {
                  setSearchQuery('');
                  navigate('/');
                }}
              >
                <i className="bi bi-x-lg"></i>
              </button>
            )}
          </form>
        </div>

        {/* Right Status & Nav */}
        <div className="d-flex align-items-center gap-2">
          {/* Health indicator */}
          <Link
            to="/diagnostics"
            className="btn btn-sm d-flex align-items-center gap-2 text-decoration-none border-secondary-subtle"
            style={{
              backgroundColor: isBackendOk ? 'rgba(25, 135, 84, 0.15)' : 'rgba(220, 53, 69, 0.15)',
              border: '1px solid',
              borderColor: isBackendOk ? '#198754' : '#dc3545',
              borderRadius: '20px',
              padding: '0.35rem 0.8rem',
            }}
          >
            <span
              className="rounded-circle"
              style={{
                width: '8px',
                height: '8px',
                backgroundColor: isBackendOk ? '#20c997' : '#dc3545',
              }}
            ></span>
            <span className="small text-light">
              {isBackendOk ? `HNSW Index: ${health?.index_size || 2983}` : 'Connecting...'}
            </span>
          </Link>

          <Link
            to="/try-video"
            className={`btn btn-sm ${location.pathname === '/try-video' ? 'btn-danger' : 'btn-outline-danger'} rounded-pill px-3`}
          >
            <i className="bi bi-cloud-arrow-up-fill me-1"></i>
            <span className="d-none d-sm-inline">Try Video</span>
          </Link>

          <Link
            to="/personalized"
            className={`btn btn-sm ${location.pathname === '/personalized' ? 'btn-danger' : 'btn-outline-light'} rounded-pill px-3`}
          >
            <i className="bi bi-person-fill me-1"></i>
            <span className="d-none d-sm-inline">Personalized</span>
          </Link>
        </div>
      </div>
    </nav>
  );
}
