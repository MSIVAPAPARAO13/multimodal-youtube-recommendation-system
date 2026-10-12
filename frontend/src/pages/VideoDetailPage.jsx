import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';
import VideoCard from '../components/VideoCard';
import LoadingSpinner from '../components/LoadingSpinner';

export default function VideoDetailPage() {
  const { videoId } = useParams();

  const [video, setVideo] = useState(null);
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Recommendations state
  const [activeTab, setActiveTab] = useState('text');
  const [rankingMode, setRankingMode] = useState('baseline');
  const [recommendations, setRecommendations] = useState([]);
  const [recLoading, setRecLoading] = useState(false);
  const [recMeta, setRecMeta] = useState(null);

  // Visual recommendations state
  const [visualCandidates, setVisualCandidates] = useState([]);
  const [visualLoading, setVisualLoading] = useState(false);
  const [visualError, setVisualError] = useState(null);

  // User Interactive Actions state
  const [isLiked, setIsLiked] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [actionFeedback, setActionFeedback] = useState(null);

  const showFeedback = (msg) => {
    setActionFeedback(msg);
    setTimeout(() => setActionFeedback(null), 3500);
  };

  const handleLike = async () => {
    const next = !isLiked;
    setIsLiked(next);
    if (next) {
      await api.recordInteraction('demo_user', videoId, 'like').catch(() => {});
      showFeedback('Added to your Liked Videos and updated user interest profile!');
    } else {
      showFeedback('Removed like.');
    }
  };

  const handleSave = async () => {
    const next = !isSaved;
    setIsSaved(next);
    if (next) {
      await api.recordInteraction('demo_user', videoId, 'save').catch(() => {});
      showFeedback('Saved to your Watch Later queue!');
    } else {
      showFeedback('Removed from Watch Later.');
    }
  };

  const handleShare = () => {
    try {
      navigator.clipboard.writeText(window.location.href);
      showFeedback('Link copied to clipboard!');
    } catch {
      showFeedback(`Share link: ${window.location.href}`);
    }
  };

  const handleLogWatch = async () => {
    await api.recordInteraction('demo_user', videoId, 'view').catch(() => {});
    showFeedback('Watch event recorded into PostgreSQL. Check the Personalized page to see updated profile!');
  };

  useEffect(() => {
    window.scrollTo(0, 0);
    setIsLiked(false);
    setIsSaved(false);
    loadVideoData();
  }, [videoId]);

  useEffect(() => {
    if (videoId) {
      loadRecommendations();
    }
  }, [videoId, rankingMode]);

  useEffect(() => {
    if (videoId && activeTab === 'visual') {
      loadVisualRecommendations();
    }
  }, [videoId, activeTab]);

  const loadVideoData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch video details
      const v = await api.getVideoDetails(videoId);
      setVideo(v);

      // Record interaction in background for personalization profile
      api.recordInteraction('demo_user', videoId, 'view').catch(() => {});

      // Fetch assets
      const assetData = await api.getVideoAssets(videoId);
      setAssets(assetData.assets || []);
    } catch (err) {
      console.error('Failed to load video:', err);
      setError('Video not found in catalog or database error.');
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendations = async () => {
    try {
      setRecLoading(true);
      const res = await api.getVideoRecommendations(videoId, {
        limit: 8,
        rankingMode,
        candidatePoolSize: 50,
      });
      setRecommendations(res.recommendations || []);
      setRecMeta(res);
    } catch (err) {
      console.error('Recommendation retrieval failed:', err);
      setRecommendations([]);
    } finally {
      setRecLoading(false);
    }
  };

  const loadVisualRecommendations = async () => {
    try {
      setVisualLoading(true);
      setVisualError(null);
      const res = await api.getVisualRecommendations(videoId, { limit: 6 });
      setVisualCandidates(res.candidates || []);
    } catch (err) {
      console.warn('Visual retrieval error:', err);
      setVisualError(
        'Direct visual features (ResNet-50 2048-D) are not registered for this video. Try videos like 10040, machine_learning_0, or programming_0.'
      );
      setVisualCandidates([]);
    } finally {
      setVisualLoading(false);
    }
  };

  if (loading) {
    return <LoadingSpinner text="Hydrating video metadata and assets from database..." />;
  }

  if (error || !video) {
    return (
      <div className="container py-5 text-center">
        <div className="alert alert-danger mb-4">
          <i className="bi bi-exclamation-octagon-fill me-2"></i>
          {error || 'Video not found.'}
        </div>
        <Link to="/" className="btn btn-outline-light rounded-pill px-4">
          <i className="bi bi-arrow-left me-2"></i>Return to Home
        </Link>
      </div>
    );
  }

  const channelTitle = video.channel_title || (video.youtube_video_id ? 'YouTube Partner' : 'MicroLens Catalog');

  return (
    <div className="py-2">
      <div className="row g-4">
        {/* Left / Main Column: Video Player Mock & Details */}
        <div className="col-lg-8">
          {/* Video Player Box */}
          <div className="yt-player-box mb-3 overflow-hidden rounded" style={{ backgroundColor: '#000', aspectRatio: '16/9' }}>
            {video.playable_url ? (
              <video
                controls
                className="w-100 h-100"
                style={{ objectFit: 'contain' }}
                poster={video.thumbnail_url || undefined}
                src={video.playable_url}
              >
                Your browser does not support HTML5 video playback.
              </video>
            ) : video.is_youtube_embed && video.youtube_embed_id ? (
              <iframe
                src={`https://www.youtube.com/embed/${video.youtube_embed_id}?autoplay=0`}
                title={video.title}
                className="w-100 h-100 border-0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            ) : (
              <div className="d-flex flex-column align-items-center justify-content-center h-100 p-4 text-center bg-dark text-secondary">
                <i className="bi bi-film text-secondary opacity-50 mb-2" style={{ fontSize: '3.5rem' }}></i>
                <h5 className="text-light mt-1 mb-1">{video.title}</h5>
                <span className="badge bg-secondary bg-opacity-25 text-secondary border border-secondary border-opacity-50 mt-1 mb-2">
                  Catalog Reference Item (No Media File Ingested)
                </span>
                <p className="small text-secondary mb-0" style={{ maxWidth: '440px' }}>
                  This video is indexed for semantic text and pgvector HNSW ANN retrieval. Local video stream or embed URL is not ingested for this record.
                </p>
              </div>
            )}
          </div>

          {/* Video Title and Metadata */}
          <h4 className="text-light fw-bold mb-2">{video.title}</h4>

          <div className="d-flex flex-wrap justify-content-between align-items-center py-2 mb-3 border-bottom border-secondary border-opacity-25 gap-3">
            <div className="d-flex align-items-center gap-3">
              <div className="yt-channel-avatar" style={{ width: '44px', height: '44px' }}>
                {channelTitle.charAt(0).toUpperCase()}
              </div>
              <div>
                <div className="fw-semibold text-light">{channelTitle}</div>
                <div className="text-secondary small">
                  Category: {video.category_id || 'Educational / Tech'}
                  {video.duration > 0 && (
                    <span className="ms-2">
                      &bull; Duration: {Math.floor(video.duration / 60)}:{String(video.duration % 60).padStart(2, '0')}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="d-flex align-items-center gap-2">
              <span className="badge bg-primary bg-opacity-25 text-primary border border-primary border-opacity-50 px-2 py-1">
                <i className="bi bi-file-text me-1"></i>Text 512-D
              </span>
              {video.feature_count > 0 && (
                <span className="badge bg-success bg-opacity-25 text-success border border-success border-opacity-50 px-2 py-1">
                  <i className="bi bi-camera-video me-1"></i>ResNet-50
                </span>
              )}
              {video.asset_count > 0 && (
                <span className="badge bg-warning bg-opacity-25 text-warning border border-warning border-opacity-50 px-2 py-1">
                  <i className="bi bi-music-note-beamed me-1"></i>{video.asset_count} Assets
                </span>
              )}
            </div>
          </div>

          {/* Interactive Action Bar */}
          <div className="d-flex flex-wrap align-items-center justify-content-between mb-3 gap-2">
            <div className="d-flex align-items-center gap-2">
              <button
                className={`btn btn-sm rounded-pill px-3 ${isLiked ? 'btn-danger' : 'btn-outline-light'}`}
                onClick={handleLike}
                title="Like video to update personalization profile"
              >
                <i className={`bi ${isLiked ? 'bi-hand-thumbs-up-fill' : 'bi-hand-thumbs-up'} me-1`}></i>
                {isLiked ? 'Liked' : 'Like'}
              </button>

              <button
                className={`btn btn-sm rounded-pill px-3 ${isSaved ? 'btn-secondary' : 'btn-outline-light'}`}
                onClick={handleSave}
                title="Save video to watch list"
              >
                <i className={`bi ${isSaved ? 'bi-bookmark-fill' : 'bi-bookmark'} me-1`}></i>
                {isSaved ? 'Saved' : 'Save'}
              </button>

              <button
                className="btn btn-sm btn-outline-light rounded-pill px-3"
                onClick={handleShare}
                title="Copy video link"
              >
                <i className="bi bi-share me-1"></i>Share
              </button>
            </div>

            <button
              className="btn btn-sm btn-outline-danger rounded-pill px-3"
              onClick={handleLogWatch}
              title="Record watch interaction for Phase 10 user personalization"
            >
              <i className="bi bi-play-circle me-1"></i>Record Watch Event
            </button>
          </div>

          {/* Action Feedback Toast Banner */}
          {actionFeedback && (
            <div className="alert alert-success alert-dismissible py-2 px-3 small mb-3 fade show d-flex align-items-center justify-content-between">
              <div>
                <i className="bi bi-check-circle-fill me-2 text-success"></i>
                {actionFeedback}
              </div>
              <button type="button" className="btn-close btn-close-white small p-2" onClick={() => setActionFeedback(null)}></button>
            </div>
          )}

          {/* Description Box */}
          <div className="yt-detail-card mb-4">
            <div className="d-flex justify-content-between text-secondary small fw-semibold mb-2">
              <span>{video.view_count ? `${Number(video.view_count).toLocaleString()} views` : 'Catalog Item'}</span>
              <span>External ID: {video.video_id}</span>
            </div>
            <p className="text-light opacity-90 small mb-0" style={{ whiteSpace: 'pre-line' }}>
              {video.description || 'No description provided in metadata manifest. Indexed in PostgreSQL pgvector via Phase 8C contrastive learning text projection.'}
            </p>
          </div>

          {/* Aligned Multimodal Assets Card */}
          {assets.length > 0 && (
            <div className="yt-detail-card mb-4">
              <h6 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
                <i className="bi bi-diagram-3-fill text-danger"></i>
                Aligned Multimodal Media Assets (Phase 12A/13)
              </h6>
              <div className="table-responsive">
                <table className="table table-dark table-sm table-hover mb-0" style={{ fontSize: '0.8rem' }}>
                  <thead>
                    <tr className="text-secondary">
                      <th>Type</th>
                      <th>Status</th>
                      <th>Storage Subpath</th>
                      <th>Checksum (SHA-256)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {assets.map((asset) => (
                      <tr key={asset.id}>
                        <td>
                          <span className="badge bg-secondary">{asset.asset_type}</span>
                        </td>
                        <td>
                          <span className={`badge ${asset.status === 'READY' ? 'bg-success' : 'bg-warning'}`}>
                            {asset.status}
                          </span>
                        </td>
                        <td className="text-truncate" style={{ maxWidth: '180px' }} title={asset.storage_path}>
                          {asset.storage_path}
                        </td>
                        <td className="font-monospace text-secondary text-truncate" style={{ maxWidth: '160px' }} title={asset.checksum_sha256}>
                          {asset.checksum_sha256 ? `${asset.checksum_sha256.substring(0, 12)}...` : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Recommendation Signal & Model Architecture Explanation */}
          <div className="yt-detail-card mb-4 border border-secondary border-opacity-25">
            <h6 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
              <i className="bi bi-cpu-fill text-primary"></i>
              Recommendation Signal & Model Architecture (Phase 8C / 11B / 14)
            </h6>
            <div className="row g-3 small">
              <div className="col-md-6">
                <div className="p-2 rounded bg-black border border-secondary border-opacity-25 h-100">
                  <div className="text-secondary fw-semibold mb-1">
                    <i className="bi bi-vector-pen text-info me-1"></i>Active Retrieval Embedding
                  </div>
                  <div className="text-light">
                    <strong>Phase 8C Contrastive BERT (512-D)</strong>
                  </div>
                  <div className="text-secondary mt-1">
                    Text-projected embedding indexed in PostgreSQL via <code>pgvector HNSW</code> (M=16, ef_search=40). 2,983 catalog vectors.
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-2 rounded bg-black border border-secondary border-opacity-25 h-100">
                  <div className="text-secondary fw-semibold mb-1">
                    <i className="bi bi-sliders text-warning me-1"></i>Reranking Weight Policy
                  </div>
                  <div className="text-light">
                    <code>w_text=0.90, w_metadata=0.10, w_visual=0.00</code>
                  </div>
                  <div className="text-secondary mt-1">
                    Visual & audio modalities isolated from production to avoid cold-start degradation. Nearest-neighbor scores represent cosine similarity.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Recommendations */}
        <div className="col-lg-4">
          <div className="yt-detail-card">
            {/* Tabs */}
            <ul className="nav nav-tabs mb-3">
              <li className="nav-item">
                <button
                  className={`nav-link ${activeTab === 'text' ? 'active' : ''}`}
                  onClick={() => setActiveTab('text')}
                >
                  <i className="bi bi-cpu me-1"></i>Related Videos
                </button>
              </li>
              <li className="nav-item">
                <button
                  className={`nav-link ${activeTab === 'visual' ? 'active' : ''}`}
                  onClick={() => setActiveTab('visual')}
                >
                  <i className="bi bi-image me-1"></i>Visual Similarity
                </button>
              </li>
            </ul>

            {/* TAB 1: TEXT CO-ENGAGEMENT RECOMMENDATIONS */}
            {activeTab === 'text' && (
              <div>
                {/* Ranking Mode Toggle */}
                <div className="d-flex align-items-center justify-content-between mb-3 p-2 rounded bg-black border border-secondary border-opacity-25">
                  <span className="small text-secondary fw-semibold">Ranking Mode:</span>
                  <div className="btn-group btn-group-sm">
                    <button
                      className={`btn btn-sm ${rankingMode === 'baseline' ? 'btn-danger' : 'btn-outline-secondary'}`}
                      onClick={() => setRankingMode('baseline')}
                      title="Phase 8C HNSW ANN Retrieval"
                    >
                      HNSW ANN
                    </button>
                    <button
                      className={`btn btn-sm ${rankingMode === 'rerank' ? 'btn-danger' : 'btn-outline-secondary'}`}
                      onClick={() => setRankingMode('rerank')}
                      title="Phase 11B Candidate Re-Ranking"
                    >
                      Phase 11B Rerank
                    </button>
                  </div>
                </div>

                {recMeta && (
                  <div className="small text-secondary mb-3 d-flex justify-content-between">
                    <span>Backend: <strong className="text-light">{recMeta.retrieval_backend}</strong></span>
                    <span>Pool: <strong className="text-light">{recMeta.count} candidates</strong></span>
                  </div>
                )}

                {recLoading ? (
                  <LoadingSpinner text="Retrieving nearest neighbors from pgvector..." />
                ) : recommendations.length > 0 ? (
                  <div className="d-flex flex-column gap-3">
                    {recommendations.map((item) => (
                      <div key={item.video_id} className="row g-2 align-items-center">
                        <div className="col-5">
                          <Link to={`/video/${encodeURIComponent(item.video_id)}`}>
                            <div className="yt-thumb-wrapper rounded overflow-hidden position-relative" style={{ aspectRatio: '16/9', backgroundColor: '#1c2024' }}>
                              {item.thumbnail_url ? (
                                <img
                                  src={item.thumbnail_url}
                                  alt={item.title}
                                  className="w-100 h-100"
                                  style={{ objectFit: 'cover' }}
                                  onError={(e) => {
                                    e.target.style.display = 'none';
                                    if (e.target.nextSibling) e.target.nextSibling.style.display = 'flex';
                                  }}
                                />
                              ) : null}
                              <div
                                className="align-items-center justify-content-center h-100 w-100"
                                style={{ display: item.thumbnail_url ? 'none' : 'flex' }}
                              >
                                <i className="bi bi-film fs-4 text-secondary opacity-50"></i>
                              </div>
                              {item.duration > 0 && (
                                <div className="yt-duration-badge">
                                  {Math.floor(item.duration / 60)}:{String(item.duration % 60).padStart(2, '0')}
                                </div>
                              )}
                              <div className="yt-score-badge">
                                Sim: {item.score.toFixed(3)}
                              </div>
                            </div>
                          </Link>
                        </div>
                        <div className="col-7">
                          <Link
                            to={`/video/${encodeURIComponent(item.video_id)}`}
                            className="text-decoration-none text-light"
                          >
                            <div
                              className="fw-semibold small text-truncate-2"
                              style={{
                                display: '-webkit-box',
                                WebkitLineClamp: 2,
                                WebkitBoxOrient: 'vertical',
                                overflow: 'hidden',
                                lineHeight: '1.25',
                              }}
                              title={item.title}
                            >
                              {item.title}
                            </div>
                          </Link>
                          <div className="text-secondary small mt-1">
                            {item.channel_title || 'MicroLens Item'}
                          </div>
                          <span className="badge bg-dark border border-secondary text-secondary small">
                            Rank #{item.rank}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-secondary small text-center py-4">
                    No recommendations available for this query item.
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: VISUAL VIDEO-TO-VIDEO SIMILARITY (PHASE 12C) */}
            {activeTab === 'visual' && (
              <div>
                <div className="alert alert-dark border-secondary small mb-3">
                  <i className="bi bi-info-circle-fill text-info me-2"></i>
                  <strong>Phase 12C Direct Visual Retrieval</strong> computes nearest neighbors using ResNet-50 visual feature vectors (2048-D). Visual ranking weights are kept strictly separate from the production text pipeline ($w_{`{visual}`} = 0.0$).
                </div>

                {visualLoading ? (
                  <LoadingSpinner text="Searching visual feature vector space..." />
                ) : visualError ? (
                  <div className="alert alert-warning small">
                    <i className="bi bi-exclamation-triangle-fill me-2"></i>
                    {visualError}
                    <div className="mt-2">
                      <Link to="/visual-demo" className="btn btn-sm btn-outline-dark text-light">
                        Explore Videos with Visual Features &rarr;
                      </Link>
                    </div>
                  </div>
                ) : visualCandidates.length > 0 ? (
                  <div className="d-flex flex-column gap-3">
                    {visualCandidates.map((cand) => (
                      <div key={cand.video_id} className="row g-2 align-items-center">
                        <div className="col-5">
                          <Link to={`/video/${encodeURIComponent(cand.video_id)}`}>
                            <div className="yt-thumb-wrapper" style={{ aspectRatio: '16/9' }}>
                              <div className="d-flex align-items-center justify-content-center h-100 bg-success bg-opacity-25">
                                <i className="bi bi-camera-video-fill fs-3 text-success"></i>
                              </div>
                              <div className="yt-score-badge bg-success">
                                Sim: {cand.score.toFixed(3)}
                              </div>
                            </div>
                          </Link>
                        </div>
                        <div className="col-7">
                          <Link
                            to={`/video/${encodeURIComponent(cand.video_id)}`}
                            className="text-decoration-none text-light"
                          >
                            <div className="fw-semibold small text-truncate">
                              Video {cand.video_id}
                            </div>
                          </Link>
                          <div className="text-secondary small mt-1">
                            Dim: {cand.dimension} &bull; {cand.feature_type}
                          </div>
                          <span className="badge bg-success bg-opacity-25 text-success small">
                            Visual Rank #{cand.rank}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-secondary small text-center py-4">
                    No visual candidates returned.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
