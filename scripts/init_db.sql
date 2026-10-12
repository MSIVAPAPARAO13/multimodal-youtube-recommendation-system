-- Database Initialization Script
-- This runs automatically when PostgreSQL container starts

-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create videos table
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255) UNIQUE NOT NULL,
    youtube_video_id VARCHAR(64),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    video_path VARCHAR(1000),
    channel_id VARCHAR(255),
    channel_title VARCHAR(255),
    published_at TIMESTAMP,
    category_id VARCHAR(64),
    thumbnail_url VARCHAR(1000),
    duration INTEGER,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    privacy_status VARCHAR(50),
    ingestion_status VARCHAR(50) DEFAULT 'PENDING',
    ingestion_error TEXT,
    last_ingestion_attempt TIMESTAMP,
    tags TEXT[],
    embedding vector(512),  -- 512-dimensional vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for fast similarity search using HNSW (efficient for dynamic data)
CREATE INDEX IF NOT EXISTS ix_video_embedding
ON videos
USING hnsw (embedding vector_cosine_ops);

-- Create index on video_id for fast lookups
CREATE INDEX IF NOT EXISTS ix_video_id ON videos(video_id);
CREATE INDEX IF NOT EXISTS ix_youtube_video_id ON videos(youtube_video_id);

-- Create video_ingestions table
CREATE TABLE IF NOT EXISTS video_ingestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255),
    youtube_video_id VARCHAR(64) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    video_path VARCHAR(1000),
    audio_path VARCHAR(1000),
    thumbnail_path VARCHAR(1000),
    transcript_text TEXT,
    transcript_language VARCHAR(32),
    transcript_available BOOLEAN DEFAULT FALSE,
    transcript_provider VARCHAR(64),
    frames_path VARCHAR(1000),
    frames_count INTEGER DEFAULT 0,
    pipeline_version VARCHAR(32) DEFAULT '12A',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_video_ingestions_youtube_id ON video_ingestions(youtube_video_id);
CREATE INDEX IF NOT EXISTS ix_video_ingestions_status ON video_ingestions(status);
CREATE INDEX IF NOT EXISTS ix_video_ingestions_video_id ON video_ingestions(video_id);

-- Create video assets table (Phase 12A)
CREATE TABLE IF NOT EXISTS video_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255) NOT NULL,
    youtube_video_id VARCHAR(64),
    asset_type VARCHAR(32) NOT NULL,
    source_url VARCHAR(1000),
    storage_path VARCHAR(1000) NOT NULL,
    mime_type VARCHAR(64),
    file_size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    status VARCHAR(32) NOT NULL DEFAULT 'READY',
    attempt_count INTEGER NOT NULL DEFAULT 1,
    error_category VARCHAR(64),
    error_message TEXT,
    pipeline_version VARCHAR(32) NOT NULL DEFAULT '12A',
    extra_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_video_asset_lookup ON video_assets(video_id, asset_type);
CREATE INDEX IF NOT EXISTS ix_video_assets_checksum ON video_assets(checksum_sha256);
CREATE INDEX IF NOT EXISTS ix_video_assets_status ON video_assets(status);

-- Create video features table (Phase 12B)
CREATE TABLE IF NOT EXISTS video_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255) NOT NULL,
    youtube_video_id VARCHAR(64),
    asset_id UUID,
    modality VARCHAR(32) NOT NULL,
    feature_type VARCHAR(64) NOT NULL,
    encoder_name VARCHAR(64) NOT NULL,
    encoder_version VARCHAR(64) NOT NULL,
    checkpoint_id VARCHAR(128),
    dimension INTEGER NOT NULL,
    preprocessing_config JSONB,
    source_checksum VARCHAR(64) NOT NULL,
    embedding JSONB NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    is_stale BOOLEAN NOT NULL DEFAULT FALSE,
    is_trained BOOLEAN NOT NULL DEFAULT FALSE,
    validation_status VARCHAR(32) NOT NULL DEFAULT 'VALID',
    pipeline_version VARCHAR(32) NOT NULL DEFAULT '12B',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_video_feature_lookup ON video_features(video_id, modality, feature_type, encoder_name);
CREATE INDEX IF NOT EXISTS ix_video_features_source_checksum ON video_features(source_checksum);
CREATE INDEX IF NOT EXISTS ix_video_features_modality ON video_features(modality);

-- Create user interactions table
CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    video_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user interactions
CREATE INDEX IF NOT EXISTS ix_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS ix_interaction_video_id ON user_interactions(video_id);
CREATE INDEX IF NOT EXISTS ix_timestamp ON user_interactions(timestamp);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_videos_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Log success
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully!';
END $$;
