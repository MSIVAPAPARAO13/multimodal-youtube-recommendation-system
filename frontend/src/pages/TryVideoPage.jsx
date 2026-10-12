import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import VideoCard from '../components/VideoCard';
import LoadingSpinner from '../components/LoadingSpinner';

export default function TryVideoPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  // Upload and Analysis state
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('visual');

  const fileInputRef = useRef(null);

  const handleLoadSample = async () => {
    try {
      setAnalyzing(true);
      setError(null);
      setUploadProgress(10);
      const sampleUrl = 'http://localhost:8001/media/local_10040/video.mp4';
      const response = await fetch(sampleUrl);
      if (!response.ok) throw new Error('Could not fetch catalog sample media');
      const blob = await response.blob();
      const sampleFile = new File([blob], 'sample_microservices_docker.mp4', { type: 'video/mp4' });
      setFile(sampleFile);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(sampleFile));
      setTitle('Building Microservices with Docker and Go');
      setDescription('Hands-on engineering walkthrough on microservice architecture, containerization with Docker, and Go backend services.');

      const formData = new FormData();
      formData.append('file', sampleFile);
      formData.append('title', 'Building Microservices with Docker and Go');
      formData.append('description', 'Hands-on engineering walkthrough on microservice architecture, containerization with Docker, and Go backend services.');
      formData.append('max_recommendations', '8');

      const data = await api.analyzeCustomVideo(formData, (progressEvent) => {
        if (progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        }
      });

      setAnalysisResult(data);
      if (data.visual_recommendations && data.visual_recommendations.length > 0) {
        setActiveTab('visual');
      } else if (data.text_recommendations && data.text_recommendations.length > 0) {
        setActiveTab('text');
      }
    } catch (err) {
      console.error('Failed to load sample video:', err);
      setError(err.message || 'Failed to load and analyze sample video.');
    } finally {
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('sample') === 'true') {
      handleLoadSample();
    }
  }, []);

  const handleFileSelect = (selectedFile) => {
    setError(null);
    setAnalysisResult(null);

    if (!selectedFile) return;

    // Client-side extension check
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const ext = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    if (!validExtensions.includes(ext)) {
      setError(`Unsupported file format '${ext}'. Please choose an MP4, AVI, MOV, MKV, or WEBM video.`);
      return;
    }

    // Client-side size check (100 MB)
    const maxBytes = 100 * 1024 * 1024;
    if (selectedFile.size > maxBytes) {
      setError(`File size (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 100 MB limit.`);
      return;
    }

    setFile(selectedFile);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select or drop a video file to analyze.');
      return;
    }

    try {
      setAnalyzing(true);
      setError(null);
      setUploadProgress(0);

      const formData = new FormData();
      formData.append('file', file);
      if (title.trim()) formData.append('title', title.trim());
      if (description.trim()) formData.append('description', description.trim());
      formData.append('max_recommendations', '8');

      const data = await api.analyzeCustomVideo(formData, (progressEvent) => {
        if (progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        }
      });

      setAnalysisResult(data);
      // If visual recommendations are available, select visual tab; else text
      if (data.visual_recommendations && data.visual_recommendations.length > 0) {
        setActiveTab('visual');
      } else if (data.text_recommendations && data.text_recommendations.length > 0) {
        setActiveTab('text');
      }
    } catch (err) {
      console.error('Custom video analysis failed:', err);
      const detail = err.response?.data?.detail || err.message || 'Analysis failed. Please ensure the video format is valid.';
      setError(detail);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setTitle('');
    setDescription('');
    setAnalysisResult(null);
    setError(null);
    setUploadProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="py-2">
      {/* Page Header */}
      <div className="d-flex flex-wrap align-items-center justify-content-between mb-4 gap-3">
        <div>
          <h3 className="fw-bold text-light mb-1 d-flex align-items-center gap-2">
            <i className="bi bi-cloud-arrow-up-fill text-danger"></i>
            Try Your Own Video
          </h3>
          <p className="text-secondary small mb-0">
            Upload a local video to extract ResNet-50 visual features, validate audio availability, encode text with Phase 8C BERT, and discover matching catalog videos.
          </p>
        </div>

        <div className="d-flex align-items-center gap-2">
          <button
            type="button"
            className="btn btn-sm btn-outline-danger d-flex align-items-center gap-2 shadow-sm"
            onClick={handleLoadSample}
            disabled={analyzing}
            title="Load sample video from local catalog for immediate verification"
          >
            {analyzing ? (
              <>
                <span className="spinner-border spinner-border-sm" role="status"></span>
                <span>Analyzing Demo...</span>
              </>
            ) : (
              <>
                <i className="bi bi-play-circle-fill"></i>
                <span>Try with Sample Video</span>
              </>
            )}
          </button>
          <span className="badge bg-dark border border-secondary text-secondary">
            <i className="bi bi-shield-check me-1 text-success"></i> Ephemeral &bull; Never Persisted
          </span>
          <span className="badge bg-dark border border-secondary text-secondary">
            Max 100 MB &bull; Up to 10 min
          </span>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="alert alert-danger alert-dismissible fade show d-flex align-items-center justify-content-between mb-4">
          <div>
            <i className="bi bi-exclamation-triangle-fill me-2"></i>
            <strong>Analysis Error:</strong> {error}
          </div>
          <button type="button" className="btn-close btn-close-white" onClick={() => setError(null)}></button>
        </div>
      )}

      {/* Upload and Form Section */}
      <div className="row g-4 mb-4">
        {/* Left: Drag & Drop Zone and Preview */}
        <div className="col-lg-7">
          <div className="yt-detail-card h-100">
            <h5 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
              <i className="bi bi-file-earmark-play-fill text-danger"></i>
              1. Video Media Upload
            </h5>

            {!previewUrl ? (
              <div
                className={`p-4 p-md-5 rounded border-2 border-dashed text-center d-flex flex-column align-items-center justify-content-center transition-all ${
                  isDragging ? 'border-danger bg-danger bg-opacity-10' : 'border-secondary border-opacity-50 bg-black'
                }`}
                style={{
                  minHeight: '260px',
                  cursor: 'pointer',
                  borderStyle: 'dashed',
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <i className="bi bi-cloud-arrow-up text-secondary opacity-75 mb-3" style={{ fontSize: '3.5rem' }}></i>
                <h6 className="text-light fw-semibold mb-1">Drag and drop your video file here</h6>
                <p className="text-secondary small mb-3">or click to browse from your device</p>
                <div className="d-flex flex-wrap gap-2 justify-content-center">
                  <span className="badge bg-secondary bg-opacity-25 text-secondary">.MP4</span>
                  <span className="badge bg-secondary bg-opacity-25 text-secondary">.WEBM</span>
                  <span className="badge bg-secondary bg-opacity-25 text-secondary">.MOV</span>
                  <span className="badge bg-secondary bg-opacity-25 text-secondary">.MKV</span>
                  <span className="badge bg-secondary bg-opacity-25 text-secondary">.AVI</span>
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  className="d-none"
                  accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
                  onChange={(e) => handleFileSelect(e.target.files?.[0])}
                />
              </div>
            ) : (
              <div>
                <div className="position-relative rounded overflow-hidden mb-3 bg-black" style={{ aspectRatio: '16/9' }}>
                  <video
                    src={previewUrl}
                    controls
                    className="w-100 h-100"
                    style={{ objectFit: 'contain' }}
                  >
                    Your browser does not support HTML5 video preview.
                  </video>
                </div>
                <div className="d-flex justify-content-between align-items-center p-2 rounded bg-black border border-secondary border-opacity-25">
                  <div className="text-truncate me-2 small">
                    <span className="text-light fw-semibold">{file.name}</span>{' '}
                    <span className="text-secondary">({(file.size / (1024 * 1024)).toFixed(2)} MB)</span>
                  </div>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-danger rounded-pill px-3"
                    onClick={handleReset}
                    disabled={analyzing}
                  >
                    <i className="bi bi-arrow-repeat me-1"></i>Change Video
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: Metadata Inputs & Execution */}
        <div className="col-lg-5">
          <div className="yt-detail-card h-100 d-flex flex-column justify-content-between">
            <div>
              <h5 className="text-light fw-bold mb-3 d-flex align-items-center gap-2">
                <i className="bi bi-card-text text-primary"></i>
                2. Optional Context Metadata
              </h5>
              <p className="text-secondary small mb-3">
                If provided, your title and description will be encoded using the active <strong>Phase 8C Contrastive BERT (512-D)</strong> model to retrieve matching text-semantic catalog items.
              </p>

              <div className="mb-3">
                <label className="form-label text-secondary small fw-semibold">Video Title (Optional)</label>
                <input
                  type="text"
                  className="form-control bg-black text-light border-secondary border-opacity-50"
                  placeholder="e.g. Introduction to Neural Networks"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  disabled={analyzing}
                />
              </div>

              <div className="mb-3">
                <label className="form-label text-secondary small fw-semibold">Description / Topics (Optional)</label>
                <textarea
                  className="form-control bg-black text-light border-secondary border-opacity-50"
                  rows={3}
                  placeholder="e.g. Deep learning basics, backpropagation algorithm, PyTorch tutorial"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  disabled={analyzing}
                ></textarea>
              </div>
            </div>

            <div className="mt-4 pt-3 border-top border-secondary border-opacity-25">
              {analyzing ? (
                <div>
                  <div className="d-flex justify-content-between text-secondary small mb-1">
                    <span>
                      {uploadProgress < 100
                        ? `Uploading video (${uploadProgress}%)...`
                        : 'Analyzing frames, extracting ResNet-50 features & audio...'}
                    </span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="progress mb-3" style={{ height: '6px', backgroundColor: '#333' }}>
                    <div
                      className="progress-bar progress-bar-striped progress-bar-animated bg-danger"
                      style={{ width: `${Math.max(uploadProgress, 25)}%` }}
                    ></div>
                  </div>
                  <LoadingSpinner text="Processing video via OpenCV & PyTorch encoders..." />
                </div>
              ) : (
                <div className="d-flex gap-2">
                  <button
                    type="button"
                    className="btn btn-danger rounded-pill px-4 flex-grow-1 fw-semibold"
                    onClick={handleSubmit}
                    disabled={!file}
                  >
                    <i className="bi bi-play-circle-fill me-2"></i>
                    Analyze Video & Discover Similar
                  </button>
                  {file && (
                    <button
                      type="button"
                      className="btn btn-outline-secondary rounded-pill px-3"
                      onClick={handleReset}
                    >
                      Reset
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Results Display */}
      {analysisResult && (
        <div className="mt-4">
          {/* Analysis Summary Card */}
          <div className="yt-detail-card mb-4 border border-secondary border-opacity-25">
            <div className="d-flex flex-wrap justify-content-between align-items-center mb-3 pb-2 border-bottom border-secondary border-opacity-25 gap-3">
              <div>
                <h5 className="text-light fw-bold mb-1 d-flex align-items-center gap-2">
                  <i className="bi bi-check-circle-fill text-success"></i>
                  Video Analysis & Modality Report
                </h5>
                <span className="text-secondary small">
                  File: <strong className="text-light">{analysisResult.video_metadata.filename}</strong> &bull; Size: {analysisResult.video_metadata.size_mb} MB &bull; Duration: {analysisResult.video_metadata.duration_seconds}s &bull; Resolution: {analysisResult.video_metadata.width}x{analysisResult.video_metadata.height}
                </span>
              </div>
              <span className="badge bg-success bg-opacity-25 text-success border border-success border-opacity-50 px-3 py-2">
                Ephemeral Analysis Verified
              </span>
            </div>

            {/* Extracted Frames Strip */}
            {analysisResult.extracted_frames && analysisResult.extracted_frames.length > 0 && (
              <div className="mb-4">
                <div className="text-secondary small fw-semibold mb-2">
                  <i className="bi bi-camera-fill me-1 text-light"></i> Extracted Deterministic Frames (Sampled for ResNet-50):
                </div>
                <div className="row g-2">
                  {analysisResult.extracted_frames.map((frameB64, idx) => (
                    <div key={idx} className="col-6 col-sm-4 col-md-3">
                      <div className="rounded overflow-hidden border border-secondary border-opacity-50" style={{ aspectRatio: '16/9' }}>
                        <img
                          src={frameB64}
                          alt={`Extracted Frame ${idx + 1}`}
                          className="w-100 h-100"
                          style={{ objectFit: 'cover' }}
                        />
                      </div>
                      <div className="text-secondary small text-center mt-1" style={{ fontSize: '0.72rem' }}>
                        Sample Frame #{idx + 1}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Modalities Intelligence Cards */}
            <div className="row g-3">
              {/* Visual Modality */}
              <div className="col-md-4">
                <div className="p-3 rounded bg-black border border-secondary border-opacity-25 h-100">
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-bold text-light">
                      <i className="bi bi-image-fill text-success me-2"></i>Visual Modality
                    </span>
                    <span className={`badge ${analysisResult.modalities.visual.status === 'EXTRACTED' ? 'bg-success' : 'bg-warning'}`}>
                      {analysisResult.modalities.visual.status}
                    </span>
                  </div>
                  <div className="text-secondary small mb-1">
                    Model: <strong className="text-light">{analysisResult.modalities.visual.model}</strong>
                  </div>
                  <div className="text-secondary small mb-1">
                    Dimension: <span className="badge bg-dark border border-secondary text-secondary">2048-D</span>
                  </div>
                  <p className="text-secondary small mb-0 mt-2" style={{ fontSize: '0.78rem' }}>
                    {analysisResult.modalities.visual.details}
                  </p>
                </div>
              </div>

              {/* Audio Modality */}
              <div className="col-md-4">
                <div className="p-3 rounded bg-black border border-secondary border-opacity-25 h-100">
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-bold text-light">
                      <i className="bi bi-volume-up-fill text-warning me-2"></i>Audio Modality
                    </span>
                    <span className={`badge ${analysisResult.modalities.audio.status === 'EXTRACTED_NON_SILENT' ? 'bg-success' : 'bg-secondary'}`}>
                      {analysisResult.modalities.audio.status}
                    </span>
                  </div>
                  <div className="text-secondary small mb-1">
                    Acoustic Check: <strong className="text-light">{analysisResult.modalities.audio.details}</strong>
                  </div>
                  <div className="text-secondary small mb-1">
                    Production Weight: <span className="badge bg-dark border border-secondary text-secondary">0.00 (Isolated)</span>
                  </div>
                  <p className="text-secondary small mb-0 mt-2" style={{ fontSize: '0.78rem' }}>
                    {analysisResult.audio_status_note}
                  </p>
                </div>
              </div>

              {/* Text Modality */}
              <div className="col-md-4">
                <div className="p-3 rounded bg-black border border-secondary border-opacity-25 h-100">
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-bold text-light">
                      <i className="bi bi-file-text-fill text-primary me-2"></i>Text Modality
                    </span>
                    <span className={`badge ${analysisResult.modalities.text.status === 'EMBEDDING_GENERATED' ? 'bg-primary' : 'bg-secondary'}`}>
                      {analysisResult.modalities.text.status}
                    </span>
                  </div>
                  <div className="text-secondary small mb-1">
                    Model: <strong className="text-light">{analysisResult.modalities.text.model}</strong>
                  </div>
                  <div className="text-secondary small mb-1">
                    Vector Dim: <span className="badge bg-dark border border-secondary text-secondary">512-D</span>
                  </div>
                  <p className="text-secondary small mb-0 mt-2" style={{ fontSize: '0.78rem' }}>
                    {analysisResult.modalities.text.details}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations Navigation Tabs */}
          <div className="yt-detail-card mb-4">
            <div className="d-flex flex-wrap justify-content-between align-items-center mb-3">
              <h5 className="text-light fw-bold mb-0">Matching Catalog Videos</h5>
              <div className="btn-group btn-group-sm">
                <button
                  type="button"
                  className={`btn ${activeTab === 'visual' ? 'btn-danger' : 'btn-outline-secondary'}`}
                  onClick={() => setActiveTab('visual')}
                >
                  <i className="bi bi-image me-1"></i>
                  Visual Similarity ({analysisResult.visual_recommendations?.length || 0})
                </button>
                <button
                  type="button"
                  className={`btn ${activeTab === 'text' ? 'btn-danger' : 'btn-outline-secondary'}`}
                  onClick={() => setActiveTab('text')}
                >
                  <i className="bi bi-cpu me-1"></i>
                  Text Metadata Query ({analysisResult.text_recommendations?.length || 0})
                </button>
              </div>
            </div>

            {/* TAB A: Visual Recommendations */}
            {activeTab === 'visual' && (
              <div>
                <div className="alert alert-dark border-secondary small mb-3">
                  <i className="bi bi-info-circle-fill text-info me-2"></i>
                  <strong>Visual Nearest Neighbors</strong> computed via cosine similarity between your upload's 8-frame ResNet-50 2048-D embedding and the catalog's verified visual feature vectors. Scores represent genuine cosine similarity.
                </div>

                {analysisResult.visual_recommendations && analysisResult.visual_recommendations.length > 0 ? (
                  <div className="row g-3">
                    {analysisResult.visual_recommendations.map((item) => (
                      <div key={item.video_id} className="col-12 col-sm-6 col-lg-4 col-xl-3">
                        <VideoCard video={item} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-5 text-secondary">
                    <i className="bi bi-image text-secondary opacity-50 mb-2" style={{ fontSize: '2.5rem' }}></i>
                    <h6>No visual candidates available</h6>
                    <p className="small mb-0">
                      Catalog visual features were not found for comparison, or no compatible embeddings were produced.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* TAB B: Text Recommendations */}
            {activeTab === 'text' && (
              <div>
                <div className="alert alert-dark border-secondary small mb-3">
                  <i className="bi bi-info-circle-fill text-primary me-2"></i>
                  <strong>Text Semantic Recommendations</strong> retrieved from PostgreSQL <code>pgvector HNSW</code> index (2,983 catalog vectors) using the Phase 8C Contrastive BERT (512-D) projection of your supplied title/description.
                </div>

                {analysisResult.text_recommendations && analysisResult.text_recommendations.length > 0 ? (
                  <div className="row g-3">
                    {analysisResult.text_recommendations.map((item) => (
                      <div key={item.video_id} className="col-12 col-sm-6 col-lg-4 col-xl-3">
                        <VideoCard video={item} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-5 text-secondary">
                    <i className="bi bi-file-text text-secondary opacity-50 mb-2" style={{ fontSize: '2.5rem' }}></i>
                    <h6>No text recommendations generated</h6>
                    <p className="small mb-0">
                      {analysisResult.modalities.text.status === 'NO_TEXT_PROVIDED'
                        ? 'No title or description was provided for this upload. Enter a title above and re-analyze to search the text index.'
                        : 'No matching catalog items were retrieved for this text query.'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
