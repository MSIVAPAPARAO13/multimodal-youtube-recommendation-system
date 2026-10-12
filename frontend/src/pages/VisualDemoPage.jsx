import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

export default function VisualDemoPage() {
  const [availableVideos, setAvailableVideos] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState('10040');
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [featureType, setFeatureType] = useState('resnet50_mean_2048');

  // Load actual videos with features dynamically from PostgreSQL
  useEffect(() => {
    loadCatalog();
  }, [featureType]);

  const loadCatalog = async () => {
    try {
      const res = await api.getMultimodalCatalog(featureType);
      if (res.videos && res.videos.length > 0) {
        setAvailableVideos(res.videos);
        // If current selection is not in list, pick the first available
        if (!res.videos.some((v) => v.video_id === selectedVideoId)) {
          setSelectedVideoId(res.videos[0].video_id);
        }
      }
    } catch (e) {
      console.warn('Could not load multimodal catalog dynamically:', e);
    }
  };

  useEffect(() => {
    if (selectedVideoId) {
      loadVisualCandidates();
    }
  }, [selectedVideoId, featureType]);

  const loadVisualCandidates = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getVisualRecommendations(selectedVideoId, {
        limit: 8,
        featureType,
      });
      setCandidates(res.candidates || []);
    } catch (err) {
      console.error('Visual retrieval error:', err);
      setError('Visual candidates could not be retrieved for this video.');
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  const currentVideo = availableVideos.find((v) => v.video_id === selectedVideoId) || {
    video_id: selectedVideoId,
    title: `Video ${selectedVideoId}`,
    category_id: 'Multimodal Asset',
  };

  return (
    <div className="py-2">
      {/* Header */}
      <div className="mb-4">
        <h4 className="text-light fw-bold mb-1">
          <i className="bi bi-camera-reels-fill text-success me-2"></i>
          Phase 12C: Direct Visual Video-to-Video Retrieval
        </h4>
        <p className="text-secondary small mb-0">
          Evaluates frame-level visual embeddings extracted via frozen ResNet-50 (2048-D) and cross-modal linear projection heads. Excluded self-query from candidate set.
        </p>
      </div>

      {/* Scientific Principle Alert */}
      <div className="alert alert-dark border-secondary small mb-4">
        <div className="d-flex align-items-center gap-2 mb-1">
          <span className="badge bg-success">Scientific Baseline</span>
          <strong className="text-light">Controlled Multimodal Evaluation:</strong>
        </div>
        <div className="text-secondary">
          In Phase 12C, visual cosine similarity retrieval was proven functional ($&gt;0.85$ intra-class similarity), but production ranking weights are deliberately configured to $w_{`{visual}`} = 0.0$ until supervised alignment demonstrates statistically significant NDCG@10 lift over pure text co-engagement ($0.4578$). This page demonstrates visual candidate retrieval as an independent capability.
        </div>
      </div>

      {/* Select Source Video Bar */}
      <div className="yt-detail-card mb-4">
        <div className="d-flex flex-wrap justify-content-between align-items-center mb-3 gap-2">
          <h6 className="text-light fw-bold mb-0">
            Select Source Query Video:
          </h6>
          <div className="d-flex align-items-center gap-2">
            <span className="text-secondary small">Feature Space:</span>
            <select
              className="form-select form-select-sm bg-dark text-light border-secondary"
              value={featureType}
              onChange={(e) => setFeatureType(e.target.value)}
              style={{ width: 'auto' }}
            >
              <option value="resnet50_mean_2048">ResNet-50 (2048-D Mean Pooled)</option>
              <option value="resnet50_projected_512">ResNet-50 (512-D Projected Head)</option>
              <option value="resnet50_thumbnail_2048">Thumbnail (2048-D)</option>
            </select>
          </div>
        </div>

        <div className="d-flex flex-wrap gap-2">
          {availableVideos.length > 0 ? (
            availableVideos.map((v) => (
              <button
                key={v.video_id}
                className={`btn btn-sm ${selectedVideoId === v.video_id ? 'btn-success' : 'btn-outline-secondary'}`}
                onClick={() => setSelectedVideoId(v.video_id)}
                title={v.title}
              >
                <i className="bi bi-play-circle me-1"></i>
                {v.title.length > 30 ? v.title.substring(0, 30) + '...' : v.title}
              </button>
            ))
          ) : (
            <span className="text-secondary small">Loading available videos from database...</span>
          )}
        </div>
      </div>

      {/* Query Video vs Visual Results Comparison */}
      <div className="row g-4">
        {/* Source Video Card */}
        <div className="col-lg-4">
          <div className="yt-detail-card h-100">
            <span className="badge bg-primary mb-2">Query Item</span>
            <h6 className="text-light fw-bold mb-2">{currentVideo.title}</h6>
            <div className="yt-thumb-wrapper mb-3" style={{ aspectRatio: '16/9' }}>
              <div className="d-flex flex-column align-items-center justify-content-center h-100 bg-black">
                <i className="bi bi-camera-video fs-1 text-success mb-2"></i>
                <span className="small text-secondary">ID: {currentVideo.video_id}</span>
              </div>
            </div>
            <div className="small text-secondary">
              <div>Category: <strong className="text-light">{currentVideo.category_id || 'Technology'}</strong></div>
              <div>Representation: <strong className="text-light">{featureType}</strong></div>
              <div>Source: <strong className="text-light">Aligned Real Media Asset</strong></div>
            </div>
            <div className="mt-3">
              <Link to={`/video/${encodeURIComponent(currentVideo.video_id)}`} className="btn btn-outline-light btn-sm w-100">
                View Full Video Details &rarr;
              </Link>
            </div>
          </div>
        </div>

        {/* Visual Matches Stream */}
        <div className="col-lg-8">
          <div className="yt-detail-card h-100">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h6 className="text-light fw-bold mb-0">
                <i className="bi bi-bounding-box text-success me-2"></i>
                Visual Nearest Neighbors ({candidates.length} Retrieved)
              </h6>
              <span className="small text-secondary">
                Cosine Similarity Space
              </span>
            </div>

            {loading ? (
              <LoadingSpinner text="Querying ResNet-50 visual vector space..." />
            ) : error ? (
              <div className="alert alert-danger small">{error}</div>
            ) : candidates.length > 0 ? (
              <div className="row row-cols-1 row-cols-sm-2 g-3">
                {candidates.map((cand) => (
                  <div key={cand.video_id} className="col">
                    <div className="p-3 rounded bg-dark border border-secondary border-opacity-25 h-100">
                      <div className="d-flex justify-content-between align-items-center mb-2">
                        <span className="badge bg-success bg-opacity-25 text-success border border-success border-opacity-50">
                          Visual Rank #{cand.rank}
                        </span>
                        <span className="badge bg-success">
                          Sim: {(cand.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="yt-thumb-wrapper mb-2" style={{ aspectRatio: '16/9' }}>
                        <div className="d-flex flex-column align-items-center justify-content-center h-100 bg-black">
                          <i className="bi bi-play-circle fs-2 text-light opacity-75"></i>
                        </div>
                      </div>
                      <h6 className="text-light small fw-bold text-truncate mb-1">
                        Video {cand.video_id}
                      </h6>
                      <div className="text-secondary small" style={{ fontSize: '0.72rem' }}>
                        <div>Dim: {cand.dimension}</div>
                        <div>Head: {cand.feature_type}</div>
                      </div>
                      <div className="mt-2">
                        <Link
                          to={`/video/${encodeURIComponent(cand.video_id)}`}
                          className="btn btn-outline-secondary btn-sm py-0 w-100"
                          style={{ fontSize: '0.75rem' }}
                        >
                          View Candidate
                        </Link>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-5 text-secondary">
                No visual candidates found for this query item.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
