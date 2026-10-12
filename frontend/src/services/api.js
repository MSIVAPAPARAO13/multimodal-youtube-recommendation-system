import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // System Health
  async getHealth() {
    const res = await client.get('/health');
    return res.data;
  },

  async getRecommendationHealth() {
    const res = await client.get('/api/recommendations/health');
    return res.data;
  },

  // Video Catalog & Search
  async getVideos({ limit = 48, offset = 0, query = '', multimodalOnly = false } = {}) {
    const params = { limit, offset };
    if (query && query.trim()) {
      params.query = query.trim();
    }
    if (multimodalOnly) {
      params.multimodal_only = true;
    }
    const res = await client.get('/index/list', { params });
    return res.data;
  },

  async getMultimodalCatalog(featureType = null) {
    const params = {};
    if (featureType) {
      params.feature_type = featureType;
    }
    const res = await client.get('/api/features/catalog', { params });
    return res.data;
  },

  // Video Detail & Assets
  async getVideoDetails(videoId) {
    const res = await client.get(`/api/videos/${encodeURIComponent(videoId)}`);
    return res.data;
  },

  async getVideoAssets(videoId) {
    try {
      const res = await client.get(`/api/youtube/assets/${encodeURIComponent(videoId)}`);
      return res.data;
    } catch {
      return { video_id: videoId, total_assets: 0, assets: [] };
    }
  },

  // Recommendation Endpoints
  async getVideoRecommendations(videoId, { limit = 10, rankingMode = 'baseline', candidatePoolSize = 50 } = {}) {
    const res = await client.get(`/api/recommendations/video/${encodeURIComponent(videoId)}`, {
      params: {
        limit,
        ranking_mode: rankingMode,
        candidate_pool_size: candidatePoolSize,
      },
    });
    return res.data;
  },

  async getVisualRecommendations(videoId, { limit = 8, featureType = 'resnet50_mean_2048' } = {}) {
    const res = await client.get(`/api/recommendations/visual/${encodeURIComponent(videoId)}`, {
      params: {
        limit,
        feature_type: featureType,
      },
    });
    return res.data;
  },

  // User Personalization
  async getUserRecommendations(userId, { limit = 12, rankingMode = 'baseline', seedVideoId = null, excludeSeen = true } = {}) {
    const params = {
      limit,
      ranking_mode: rankingMode,
      exclude_seen: excludeSeen,
    };
    if (seedVideoId) {
      params.seed_video_id = seedVideoId;
    }
    const res = await client.get(`/api/recommendations/user/${encodeURIComponent(userId)}`, { params });
    return res.data;
  },

  // User Interactions
  async recordInteraction(userId, videoId, interactionType = 'view') {
    const res = await client.post('/api/interactions', {
      user_id: userId,
      video_id: videoId,
      interaction_type: interactionType,
    });
    return res.data;
  },

  async getUserInteractions(userId, limit = 20) {
    const res = await client.get(`/api/interactions/user/${encodeURIComponent(userId)}`, {
      params: { limit },
    });
    return res.data;
  },

  async clearUserInteractions(userId) {
    const res = await client.delete(`/api/interactions/user/${encodeURIComponent(userId)}`);
    return res.data;
  },

  // Custom Video Upload & Analysis (Try Your Own Video)
  async analyzeCustomVideo(formData, onUploadProgress) {
    const res = await client.post('/api/custom-video/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
    return res.data;
  },
};

export default api;

