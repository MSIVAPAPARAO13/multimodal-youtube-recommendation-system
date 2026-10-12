import React, { useState } from 'react';
import { Link } from 'react-router-dom';

function formatDuration(sec) {
  if (sec === undefined || sec === null || isNaN(sec) || sec <= 0) return null;
  const mins = Math.floor(sec / 60);
  const remaining = Math.floor(sec % 60);
  return `${mins}:${remaining < 10 ? '0' : ''}${remaining}`;
}

export default function VideoCard({ video, score, rank, showModality = true }) {
  const [imageError, setImageError] = useState(false);
  const videoId = video.video_id || video.id;
  const title = video.title || 'Untitled Video';
  const channel = video.channel_title || null;
  const category = video.category_id || 'Educational';

  // Determine thumbnail URL
  let thumbSrc = video.thumbnail_url || null;
  if (!thumbSrc && video.youtube_video_id) {
    const ytId = String(video.youtube_video_id).trim();
    if (!ytId.startsWith('local_') && /^[a-zA-Z0-9_-]{11}$/.test(ytId)) {
      thumbSrc = `https://img.youtube.com/vi/${ytId}/mqdefault.jpg`;
    }
  }

  // Multimodal indicators
  const isMultimodal = Boolean(video.has_visual || (video.feature_count && video.feature_count > 0));
  const durationStr = formatDuration(video.duration);

  return (
    <div className="col">
      <Link
        to={`/video/${encodeURIComponent(videoId)}`}
        className="yt-card text-decoration-none"
        aria-label={`View video: ${title}`}
      >
        <div className="yt-thumb-wrapper" style={{ background: '#1c2024' }}>
          {thumbSrc && !imageError ? (
            <img
              src={thumbSrc}
              alt={title}
              className="yt-thumb-img"
              loading="lazy"
              onError={() => setImageError(true)}
            />
          ) : (
            <div className="d-flex flex-column align-items-center justify-content-center h-100 p-3 text-center bg-dark bg-opacity-75">
              <i className="bi bi-file-earmark-text text-secondary fs-2 mb-1"></i>
              <span className="badge bg-secondary bg-opacity-25 text-secondary border border-secondary border-opacity-25 small px-2 py-1">
                Catalog Reference Item
              </span>
            </div>
          )}

          {/* Real verified duration only */}
          {durationStr && (
            <div className="yt-duration-badge" title="Verified Media Duration">
              {durationStr}
            </div>
          )}

          {/* Modality Badges */}
          {showModality && (
            <div className="yt-modality-badges">
              <span className="yt-badge-text" title="Phase 8C 512-D Co-Engagement Text Embedding">
                <i className="bi bi-file-text-fill me-1"></i>Text 512D
              </span>
              {isMultimodal && (
                <span className="yt-badge-visual" title="ResNet-50 Visual Feature Vector Available">
                  <i className="bi bi-camera-video-fill me-1"></i>ResNet-50
                </span>
              )}
            </div>
          )}

          {/* Cosine similarity score badge (not a probability percentage) */}
          {score !== undefined && (
            <div className="yt-score-badge" title="Cosine Similarity Score [-1.0, 1.0]">
              <i className="bi bi-compass me-1"></i>
              Sim: {typeof score === 'number' ? score.toFixed(3) : score}
            </div>
          )}

          {/* Rank Badge if present */}
          {rank !== undefined && (
            <div
              className="position-absolute"
              style={{
                top: '8px',
                right: score !== undefined ? '90px' : '8px',
                backgroundColor: 'rgba(0,0,0,0.85)',
                color: '#ffc107',
                fontSize: '0.72rem',
                fontWeight: '700',
                padding: '2px 7px',
                borderRadius: '8px',
              }}
            >
              #{rank}
            </div>
          )}
        </div>

        <div className="yt-card-content">
          <div className="yt-channel-avatar">
            {(channel || title).charAt(0).toUpperCase()}
          </div>
          <div className="yt-card-info">
            <h6 className="yt-card-title text-light" title={title}>
              {title}
            </h6>
            <div className="yt-card-meta">
              {channel && <div>{channel}</div>}
              <div className="text-secondary small d-flex gap-2">
                <span>{category}</span>
                {video.view_count && Number(video.view_count) > 0 ? (
                  <span>&bull; {Number(video.view_count).toLocaleString()} views</span>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </Link>
    </div>
  );
}
