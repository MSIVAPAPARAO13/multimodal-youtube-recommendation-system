import React from 'react';

export default function EmptyState({ title = 'No Videos Found', message = 'No results match your criteria.', onAction, actionText }) {
  return (
    <div className="text-center py-5 px-3">
      <div className="mb-3 text-secondary">
        <i className="bi bi-search display-3 opacity-50"></i>
      </div>
      <h5 className="text-light fw-bold mb-2">{title}</h5>
      <p className="text-secondary small mb-4" style={{ maxWidth: '400px', margin: '0 auto' }}>
        {message}
      </p>
      {onAction && actionText && (
        <button className="btn btn-outline-danger btn-sm rounded-pill px-4" onClick={onAction}>
          {actionText}
        </button>
      )}
    </div>
  );
}
