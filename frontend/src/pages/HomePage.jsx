import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import api from '../services/api';
import VideoCard from '../components/VideoCard';
import LoadingSpinner from '../components/LoadingSpinner';
import EmptyState from '../components/EmptyState';

const FILTER_CHIPS = [
  { id: 'all', label: 'All Videos' },
  { id: 'multimodal', label: 'Multimodal Ingested' },
  { id: 'machine_learning', label: 'Machine Learning' },
  { id: 'programming', label: 'Programming' },
  { id: 'cloud', label: 'Cloud Computing' },
  { id: 'web', label: 'Web Development' },
  { id: 'cybersecurity', label: 'Cybersecurity' },
  { id: 'popular', label: 'Popular Catalog' },
];

export default function HomePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryParam = searchParams.get('q') || '';

  const [videos, setVideos] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeChip, setActiveChip] = useState('all');
  const [offset, setOffset] = useState(0);
  const limit = 36;

  useEffect(() => {
    loadVideos(true);
  }, [queryParam, activeChip]);

  const loadVideos = async (reset = false) => {
    try {
      setLoading(true);
      setError(null);

      const currentOffset = reset ? 0 : offset;
      let effectiveQuery = queryParam;

      if (activeChip !== 'all' && activeChip !== 'multimodal' && activeChip !== 'popular' && !queryParam) {
        effectiveQuery = activeChip.replace('_', ' ');
      }

      const data = await api.getVideos({
        limit,
        offset: currentOffset,
        query: effectiveQuery,
        multimodalOnly: activeChip === 'multimodal',
      });

      let fetched = data.videos || [];

      if (reset) {
        setVideos(fetched);
        setOffset(limit);
      } else {
        setVideos((prev) => [...prev, ...fetched]);
        setOffset((prev) => prev + limit);
      }
      setTotalCount(data.total || fetched.length);
    } catch (err) {
      console.error('Failed to load videos:', err);
      setError('Unable to connect to recommendation backend. Please ensure the FastAPI server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleChipClick = (chipId) => {
    setActiveChip(chipId);
    if (queryParam) {
      setSearchParams({});
    }
  };

  return (
    <div className="py-2">
      {/* Category Chips Bar */}
      <div className="yt-chips-container mb-4">
        {FILTER_CHIPS.map((chip) => (
          <button
            key={chip.id}
            className={`yt-chip ${activeChip === chip.id && !queryParam ? 'active' : ''}`}
            onClick={() => handleChipClick(chip.id)}
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Header and Results Count */}
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          {queryParam ? (
            <h5 className="text-light fw-bold mb-1">
              Search results for "<span className="text-danger">{queryParam}</span>"
            </h5>
          ) : (
            <h5 className="text-light fw-bold mb-1">
              Recommended for You
            </h5>
          )}
          <span className="text-secondary small">
            Showing {videos.length} of {totalCount} indexed videos in pgvector HNSW database
          </span>
        </div>

        {queryParam && (
          <button
            className="btn btn-sm btn-outline-secondary rounded-pill px-3"
            onClick={() => setSearchParams({})}
          >
            Clear Search
          </button>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="alert alert-danger d-flex align-items-center justify-content-between" role="alert">
          <div>
            <i className="bi bi-exclamation-triangle-fill me-2"></i>
            {error}
          </div>
          <button className="btn btn-sm btn-outline-danger" onClick={() => loadVideos(true)}>
            Retry
          </button>
        </div>
      )}

      {/* Video Grid */}
      {videos.length > 0 ? (
        <>
          <div className="row row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-xl-4 g-3 mb-4">
            {videos.map((video) => (
              <VideoCard key={video.video_id} video={video} />
            ))}
          </div>

          {/* Load More Button */}
          {videos.length < totalCount && (
            <div className="text-center my-4">
              <button
                className="btn btn-outline-light rounded-pill px-4 py-2"
                onClick={() => loadVideos(false)}
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Load More Videos'}
              </button>
            </div>
          )}
        </>
      ) : !loading && (
        <EmptyState
          title="No Videos Found"
          message={queryParam ? `No videos matched "${queryParam}". Try searching for another topic or reset filters.` : 'No videos found in the index.'}
          actionText="View All Videos"
          onAction={() => {
            setActiveChip('all');
            setSearchParams({});
          }}
        />
      )}

      {/* Loading state */}
      {loading && videos.length === 0 && (
        <LoadingSpinner text="Retrieving video catalog from PostgreSQL pgvector index..." />
      )}
    </div>
  );
}
