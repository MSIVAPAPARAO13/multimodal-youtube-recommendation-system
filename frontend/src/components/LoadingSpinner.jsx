import React from 'react';

export default function LoadingSpinner({ text = 'Retrieving nearest candidates from pgvector index...' }) {
  return (
    <div className="d-flex flex-column align-items-center justify-content-center py-5">
      <div className="spinner-border text-danger mb-3" style={{ width: '3rem', height: '3rem' }} role="status">
        <span className="visually-hidden">Loading...</span>
      </div>
      <div className="text-secondary small fw-medium">{text}</div>
    </div>
  );
}
