import React, { useState, useEffect } from 'react';
import api from '../services/api';
import VideoCard from '../components/VideoCard';
import LoadingSpinner from '../components/LoadingSpinner';

const PRESET_USERS = [
  { id: 'demo_user', label: 'Default Demo User' },
  { id: 'ml_enthusiast', label: 'ML & AI Explorer' },
  { id: 'cloud_architect', label: 'Cloud & DevOps Fan' },
  { id: 'new_guest_user', label: 'Cold-Start Guest (No History)' },
];

export default function PersonalizedPage() {
  const [userId, setUserId] = useState('demo_user');
  const [customUserId, setCustomUserId] = useState('');
  const [rankingMode, setRankingMode] = useState('baseline');
  const [excludeSeen, setExcludeSeen] = useState(true);

  const [interactions, setInteractions] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState(null);

  // Quick Add Video to History
  const [addVideoId, setAddVideoId] = useState('');
  const [addingInteraction, setAddingInteraction] = useState(false);
  const [quickSamples, setQuickSamples] = useState([]);

  useEffect(() => {
    api.getMultimodalCatalog().then((res) => {
      if (res.videos && res.videos.length > 0) {
        setQuickSamples(res.videos.slice(0, 4));
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    loadUserHistory();
    loadPersonalizedRecommendations();
  }, [userId, rankingMode, excludeSeen]);

  const loadUserHistory = async () => {
    try {
      setHistoryLoading(true);
      const res = await api.getUserInteractions(userId, 20);
      setInteractions(res.interactions || []);
    } catch {
      setInteractions([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleClearHistory = async () => {
    try {
      await api.clearUserInteractions(userId);
      await loadUserHistory();
      await loadPersonalizedRecommendations();
    } catch (err) {
      console.error('Failed to clear interactions:', err);
    }
  };

  const loadPersonalizedRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getUserRecommendations(userId, {
        limit: 12,
        rankingMode,
        excludeSeen,
      });
      setRecommendations(res.recommendations || []);
      setMeta(res);
    } catch (err) {
      console.error('Personalization failed:', err);
      setError('Failed to fetch personalized recommendations for user.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddInteraction = async (e) => {
    e.preventDefault();
    if (!addVideoId.trim()) return;
    try {
      setAddingInteraction(true);
      await api.recordInteraction(userId, addVideoId.trim(), 'view');
      setAddVideoId('');
      await loadUserHistory();
      await loadPersonalizedRecommendations();
    } catch (err) {
      alert('Could not record interaction. Please check video ID.');
    } finally {
      setAddingInteraction(false);
    }
  };

  const handleSimulatePresetVideo = async (sampleVidId) => {
    try {
      await api.recordInteraction(userId, sampleVidId, 'view');
      await loadUserHistory();
      await loadPersonalizedRecommendations();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="py-2">
      {/* Header */}
      <div className="mb-4">
        <h4 className="text-light fw-bold mb-1">
          <i className="bi bi-person-hearts text-danger me-2"></i>
          Recency-Weighted User Personalization
        </h4>
        <p className="text-secondary small mb-0">
          Phase 10: Dynamically constructs an L2-normalized 512-D user interest vector ($\lambda = 0.85$ position decay) from chronological watch history and queries the PostgreSQL HNSW index.
        </p>
      </div>

      <div className="row g-4 mb-4">
        {/* Profile Switcher Card */}
        <div className="col-lg-5">
          <div className="yt-detail-card h-100">
            <h6 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
              <i className="bi bi-person-badge-fill text-primary"></i>
              Simulate User Profile
            </h6>

            {/* Presets */}
            <div className="d-flex flex-wrap gap-2 mb-3">
              {PRESET_USERS.map((u) => (
                <button
                  key={u.id}
                  className={`btn btn-sm ${userId === u.id ? 'btn-danger' : 'btn-outline-secondary'}`}
                  onClick={() => setUserId(u.id)}
                >
                  {u.label}
                </button>
              ))}
            </div>

            {/* Custom User Input */}
            <div className="input-group input-group-sm mb-3">
              <input
                type="text"
                className="form-control bg-dark text-light border-secondary"
                placeholder="Or enter custom user ID..."
                value={customUserId}
                onChange={(e) => setCustomUserId(e.target.value)}
              />
              <button
                className="btn btn-outline-danger"
                onClick={() => {
                  if (customUserId.trim()) setUserId(customUserId.trim());
                }}
              >
                Set User
              </button>
            </div>

            <hr style={{ borderColor: 'var(--yt-border)' }} />

            {/* Quick Watch Event Logger */}
            <h6 className="text-light fw-semibold small mb-2">Simulate Video Watch Event:</h6>
            <form onSubmit={handleAddInteraction} className="input-group input-group-sm mb-2">
              <input
                type="text"
                className="form-control bg-dark text-light border-secondary"
                placeholder="Video ID (e.g. 10040, 2, machine_learning_0)"
                value={addVideoId}
                onChange={(e) => setAddVideoId(e.target.value)}
              />
              <button className="btn btn-danger" type="submit" disabled={addingInteraction}>
                {addingInteraction ? 'Saving...' : 'Add Watch Event'}
              </button>
            </form>

            <div className="d-flex flex-wrap gap-1 mb-3">
              <span className="small text-secondary me-1">Quick Add:</span>
              {quickSamples.length > 0 ? (
                quickSamples.map((s) => (
                  <button
                    key={s.video_id}
                    className="btn btn-xs btn-outline-secondary py-0 px-2 small text-truncate"
                    style={{ fontSize: '0.75rem', maxWidth: '120px' }}
                    onClick={() => handleSimulatePresetVideo(s.video_id)}
                    title={s.title}
                  >
                    + {s.title.substring(0, 12)}...
                  </button>
                ))
              ) : (
                <button
                  className="btn btn-xs btn-outline-secondary py-0 px-2 small"
                  style={{ fontSize: '0.75rem' }}
                  onClick={() => handleSimulatePresetVideo('10040')}
                >
                  + Sample 10040
                </button>
              )}
            </div>

            {/* Watch History List */}
            <div className="border border-secondary border-opacity-25 rounded p-2 bg-black bg-opacity-50">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="small text-secondary fw-semibold">
                  Recorded Watch History ({interactions.length}):
                </span>
                <div className="d-flex align-items-center gap-2">
                  {historyLoading && <span className="spinner-border spinner-border-sm text-secondary"></span>}
                  {interactions.length > 0 && (
                    <button
                      className="btn btn-xs btn-outline-danger py-0 px-2 small"
                      style={{ fontSize: '0.72rem' }}
                      onClick={handleClearHistory}
                      title="Clear watch history for this user"
                    >
                      Clear All
                    </button>
                  )}
                </div>
              </div>

              {interactions.length > 0 ? (
                <div style={{ maxHeight: '140px', overflowY: 'auto' }}>
                  {interactions.map((ev, i) => (
                    <div key={ev.id || i} className="d-flex justify-content-between small text-secondary py-1 border-bottom border-dark">
                      <span className="text-light text-truncate" style={{ maxWidth: '180px' }}>
                        <i className="bi bi-play-circle me-1 text-danger"></i>
                        Video: {ev.video_id}
                      </span>
                      <span style={{ fontSize: '0.7rem' }}>
                        {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : 'Just now'}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="small text-secondary fst-italic py-2 text-center">
                  No interactions recorded yet. Cold-start fallback will be triggered.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Model Telemetry & Controls Card */}
        <div className="col-lg-7">
          <div className="yt-detail-card h-100">
            <h6 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
              <i className="bi bi-sliders text-danger"></i>
              Personalization Pipeline Diagnostics
            </h6>

            <div className="row g-3 mb-3">
              <div className="col-sm-6">
                <div className="p-3 rounded bg-dark border border-secondary border-opacity-25">
                  <div className="text-secondary small">Personalization Status</div>
                  <div className="fs-5 fw-bold text-light mt-1">
                    {meta?.personalization_applied ? (
                      <span className="text-success">
                        <i className="bi bi-check-circle-fill me-1"></i>Active (Profile Vector)
                      </span>
                    ) : (
                      <span className="text-warning">
                        <i className="bi bi-arrow-repeat me-1"></i>Cold-Start Fallback
                      </span>
                    )}
                  </div>
                  <div className="text-secondary small mt-1">
                    {meta?.fallback_reason ? `Reason: ${meta.fallback_reason}` : `History length: ${meta?.history_length || 0}`}
                  </div>
                </div>
              </div>

              <div className="col-sm-6">
                <div className="p-3 rounded bg-dark border border-secondary border-opacity-25">
                  <div className="text-secondary small">Retrieval Architecture</div>
                  <div className="fs-5 fw-bold text-light mt-1">
                    {meta?.retrieval_backend === 'pgvector' ? 'pgvector HNSW' : 'Exact Search'}
                  </div>
                  <div className="text-secondary small mt-1">
                    Representation: 512-D Co-Engagement
                  </div>
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="d-flex flex-wrap align-items-center justify-content-between p-2 rounded bg-black border border-secondary border-opacity-25 gap-2">
              <div className="form-check form-switch mb-0">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="excludeSeenSwitch"
                  checked={excludeSeen}
                  onChange={(e) => setExcludeSeen(e.target.checked)}
                />
                <label className="form-check-label text-light small" htmlFor="excludeSeenSwitch">
                  Exclude Previously Seen Items
                </label>
              </div>

              <div className="d-flex align-items-center gap-2">
                <span className="small text-secondary">Stage 2 Ranking:</span>
                <div className="btn-group btn-group-sm">
                  <button
                    className={`btn btn-sm ${rankingMode === 'baseline' ? 'btn-danger' : 'btn-outline-secondary'}`}
                    onClick={() => setRankingMode('baseline')}
                  >
                    HNSW Baseline
                  </button>
                  <button
                    className={`btn btn-sm ${rankingMode === 'rerank' ? 'btn-danger' : 'btn-outline-secondary'}`}
                    onClick={() => setRankingMode('rerank')}
                  >
                    Phase 11B Rerank
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations Output Grid */}
      <h5 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
        <i className="bi bi-stars text-warning"></i>
        Personalized Candidate Stream ({recommendations.length} Items)
      </h5>

      {loading ? (
        <LoadingSpinner text="Computing recency-weighted profile and retrieving candidates..." />
      ) : error ? (
        <div className="alert alert-danger">{error}</div>
      ) : recommendations.length > 0 ? (
        <div className="row row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-xl-4 g-3">
          {recommendations.map((item) => (
            <VideoCard
              key={item.video_id}
              video={item}
              score={item.score}
              rank={item.rank}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-5 text-secondary">
          No recommendations available for this profile configuration.
        </div>
      )}
    </div>
  );
}
