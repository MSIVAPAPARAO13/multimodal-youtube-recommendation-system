# Multimodal Video Recommender

A content-based video recommendation system that combines visual, audio, and text features using a two-tower neural network architecture. Trained with contrastive learning on the MicroLens-50K dataset.

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [API](#api)
- [Database](#database)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Architecture

```
                            Two-Tower Architecture
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Video Frames ──► ResNet-50 ──────┐                                    │
│                                    │                                    │
│   Audio Track ───► Wav2Vec2 ───────┼──► Gated Fusion ──► 512-d Embed    │
│                                    │                                    │
│   Title Text ────► BERT ───────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Query Video                              Candidate Videos
     │                                         │
     ▼                                         ▼
┌─────────┐                              ┌─────────┐
│  Tower  │                              │  Tower  │
│(shared) │                              │(shared) │
└────┬────┘                              └────┬────┘
     │                                         │
     ▼                                         ▼
  512-d ──────────► Cosine Similarity ◄────── 512-d
                           │
                           ▼
                   Recommendation Score
```

The two-tower design encodes videos independently, allowing precomputation of all candidate embeddings. At query time, we only need to encode the query video and compute dot products against the index.

## How It Works

### Feature Extraction

Each video is processed through three specialized encoders:

**Video Encoder (ResNet-50)**
- Samples 8 frames uniformly from the video
- Each frame passes through ResNet-50 pretrained on ImageNet
- Frame features are mean-pooled temporally
- Output: 512-dimensional visual embedding

**Audio Encoder (Wav2Vec2)**
- Extracts first 5 seconds of audio at 16kHz
- Processes through Wav2Vec2 pretrained on LibriSpeech
- Hidden states are mean-pooled across time
- Output: 512-dimensional audio embedding

**Text Encoder (BERT)**
- Tokenizes video title using BERT tokenizer
- Passes through BERT-base pretrained on Wikipedia/BookCorpus
- Uses [CLS] token representation
- Output: 512-dimensional text embedding

### Multimodal Fusion

The three modality embeddings are combined using Gated Fusion:

```python
# Learned gate weights determine modality importance
gate_weights = softmax(W @ concat(video, audio, text))  # [3,]

# Weighted combination
fused = gate_weights[0] * video + gate_weights[1] * audio + gate_weights[2] * text
```

This allows the model to learn that music videos should weight audio heavily, while tutorials might weight text more.

### Training

The model is trained with InfoNCE contrastive loss:

- Each batch contains N videos
- Each video is paired with itself (positive) and N-1 others (negatives)
- Loss pushes positive pairs together, negative pairs apart
- Temperature parameter (0.07) controls distribution sharpness

Only the fusion and projection layers are trained. The pretrained encoders are frozen to leverage transfer learning and prevent overfitting.

### Inference

1. Precompute embeddings for all candidate videos
2. Store embeddings in index (in-memory or PostgreSQL with pgvector)
3. For a query video, compute its embedding
4. Find top-K candidates by cosine similarity

## Models

| Component | Architecture | Pretrained On | Parameters | Trainable |
|-----------|--------------|---------------|------------|-----------|
| Video Encoder | ResNet-50 | ImageNet | 23M | No |
| Audio Encoder | Wav2Vec2-base | LibriSpeech (960h) | 95M | No |
| Text Encoder | BERT-base | Wikipedia + BookCorpus | 110M | No |
| Fusion + Projection | Gated Fusion + Linear | - | ~1M | Yes |

## Project Structure

```
multimodal-video-recommender/
├── src/
│   ├── features/                    # Feature extractors
│   │   ├── video_encoder.py         # ResNet-50 wrapper
│   │   ├── audio_encoder.py         # Wav2Vec2 wrapper
│   │   └── text_encoder.py          # BERT wrapper
│   │
│   ├── models/                      # Neural network architectures
│   │   ├── fusion.py                # ConcatFusion, GatedFusion, AttentionFusion
│   │   └── two_tower.py             # VideoTower, TwoTowerModel
│   │
│   ├── data/                        # Data loading and processing
│   │   ├── processors.py            # VideoProcessor, AudioProcessor
│   │   ├── dataset.py               # Generic PyTorch Dataset
│   │   └── microlens_dataset.py     # MicroLens-specific loader
│   │
│   ├── training/                    # Training infrastructure
│   │   ├── losses.py                # InfoNCE, TripletLoss, ContrastiveLoss
│   │   └── trainer.py               # Training loop with checkpointing
│   │
│   ├── api/                         # REST API
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── app.py                   # FastAPI server with PostgreSQL index
│   │
│   ├── db/                          # Database layer
│   │   ├── config.py                # Connection configuration
│   │   ├── models.py                # SQLAlchemy ORM models
│   │   ├── database.py              # Session management
│   │   └── repository.py            # CRUD operations, similarity search
│   │
│   └── utils/
│       └── device.py                # CUDA/MPS/CPU device selection
│
├── scripts/
│   ├── train_cloud.py               # Training script with dataset download
│   ├── gcp_setup.sh                 # GCP VM creation
│   ├── vm_setup.sh                  # VM environment setup
│   └── init_db.sql                  # PostgreSQL schema initialization
│
├── data/
│   └── raw/microlens/               # Dataset directory
│       ├── MicroLens-50k_titles.csv
│       ├── MicroLens-50k_pairs.csv
│       └── videos/
│
├── checkpoints/                     # Saved model weights
├── logs/                            # TensorBoard logs
├── docker-compose.yml               # PostgreSQL + API services
├── Dockerfile                       # Container build
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.10+
- ffmpeg (video/audio processing)
- Docker (optional, for PostgreSQL)

### Setup

```bash
git clone https://github.com/sananm/multimodal-video-recommender.git
cd multimodal-video-recommender

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Download Pretrained Model

Download the trained model from the GitHub release:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/best.pt \
    https://github.com/sananm/multimodal-video-recommender/releases/download/v1.0.0/best.pt
```

Or download manually from the [Releases page](https://github.com/sananm/multimodal-video-recommender/releases).

### System Dependencies

macOS:
```bash
brew install ffmpeg
```

Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0
```

### Database Dependencies (optional)

```bash
pip install sqlalchemy psycopg2-binary pgvector python-dotenv
```

## Training

### Local Training

For development and testing on CPU or Apple Silicon:

```bash
python scripts/train_cloud.py \
    --epochs 5 \
    --max_videos 100 \
    --batch_size 4 \
    --lr 0.0001
```

### Cloud Training (GCP)

For production training with GPU acceleration:

**1. Create VM**

- Go to Google Cloud Console > Compute Engine
- Create instance:
  - Machine type: n1-standard-4 (4 vCPU, 15GB RAM)
  - GPU: NVIDIA T4 x 1
  - Boot disk: Deep Learning VM with PyTorch (Ubuntu), 100GB SSD
- Request GPU quota if needed (IAM & Admin > Quotas)

**2. Setup Environment**

```bash
gcloud compute ssh YOUR_VM_NAME --zone=YOUR_ZONE

sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0
pip3 install transformers librosa opencv-python pandas tqdm tensorboard

git clone https://github.com/sananm/multimodal-video-recommender.git
cd multimodal-video-recommender
```

**3. Run Training**

```bash
# Use tmux to persist session across disconnections
tmux

python3 scripts/train_cloud.py \
    --download \
    --epochs 10 \
    --max_videos 500 \
    --batch_size 8 \
    --lr 0.00001

# Detach: Ctrl+B, then D
# Reattach: tmux attach
```

**4. Download Model**

In GCP SSH browser: Download file
```
~/multimodal-video-recommender/checkpoints/best.pt
```

**5. Stop VM**

```bash
gcloud compute instances stop YOUR_VM_NAME --zone=YOUR_ZONE
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 8 | Samples per batch |
| `--lr` | 0.0001 | Learning rate |
| `--max_videos` | 500 | Maximum videos to use |
| `--download` | False | Download dataset first |
| `--data_dir` | data/raw/microlens | Dataset location |

## API

### Running the Server

```bash
# Start database
docker-compose up -d db

# Configure connection
cp .env.example .env

# Start API
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
    "status": "ok",
    "model_loaded": true,
    "index_size": 10,
    "device": "mps",
    "storage": "postgresql"
}
```

#### Index Videos

```bash
curl -X POST http://localhost:8000/index/microlens \
    -H "Content-Type: application/json" \
    -d '{"max_videos": 10}'
```

```json
{
    "indexed": 10,
    "skipped": 0,
    "total_in_index": 10,
    "message": "Indexed 10 videos, skipped 0 (already in DB)"
}
```

#### Get Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"video_id": "1002", "top_k": 5}'
```

```json
{
    "query_video_id": "1002",
    "query_title": "appropriate play meme brothers...",
    "recommendations": [
        {"video_id": "10037", "score": 0.9386, "title": "Originally I love you..."},
        {"video_id": "10039", "score": 0.9321, "title": "this bizarre manipulation..."},
        {"video_id": "10034", "score": 0.9290, "title": "demon water water..."},
        {"video_id": "10036", "score": 0.9196, "title": "24 Hui Hui actually..."},
        {"video_id": "10005", "score": 0.9185, "title": "you build my guess..."}
    ]
}
```

#### List Indexed Videos

```bash
curl http://localhost:8000/index/list
```

#### Index Single Video

```bash
curl -X POST http://localhost:8000/index \
    -H "Content-Type: application/json" \
    -d '{
        "video_id": "custom_1",
        "video_path": "/path/to/video.mp4",
        "title": "My Video Title"
    }'
```

#### Delete Video

```bash
curl -X DELETE http://localhost:8000/index/1002
```

## Database

### Configuration

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env`:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=video_recommender
```

### Docker Commands

```bash
# Start PostgreSQL
docker-compose up -d db

# View logs
docker-compose logs db

# Stop (preserves data)
docker-compose stop db

# Stop and remove (preserves data volume)
docker-compose down

# Stop and delete all data
docker-compose down -v
```

### Schema

```sql
CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT,
    description TEXT,
    video_path TEXT,
    tags TEXT[],
    embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON videos USING ivfflat (embedding vector_cosine_ops);
```

### Similarity Search

pgvector enables efficient approximate nearest neighbor search:

```sql
SELECT video_id, title, 1 - (embedding <=> query_embedding) AS similarity
FROM videos
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

The `<=>` operator computes cosine distance. Subtracting from 1 converts to similarity.

## Device Support

The system automatically selects the best available compute device:

```python
from src.utils.device import get_device

device = get_device()  # Returns: cuda > mps > cpu
```

| Device | Hardware | Notes |
|--------|----------|-------|
| cuda | NVIDIA GPU | Fastest training |
| mps | Apple Silicon | M1/M2/M3 Macs |
| cpu | Any | Slowest, but always available |

## Dataset

This project uses the [MicroLens-50K](https://recsys.westlake.edu.cn/MicroLens-Dataset/) dataset from Westlake University:

- 50,000 short videos with titles
- User interaction pairs (user_id, video_id, timestamp)
- Videos are from a TikTok-style short video platform

The training script downloads videos automatically with the `--download` flag.

## Troubleshooting

### Port Already in Use

```bash
lsof -ti:8000 | xargs kill -9
```

### PostgreSQL Port Conflict

If port 5432 is already in use (local PostgreSQL installation), the docker-compose.yml maps to port 5433. Update `.env` accordingly:

```
POSTGRES_PORT=5433
```

### Training Loss is NaN

Lower the learning rate:

```bash
python3 scripts/train_cloud.py --lr 0.00001
```

### Partial File / Corrupted Video Warnings

Some videos may be corrupted from incomplete downloads. Remove small files:

```bash
find data/raw/microlens/videos/ -name "*.mp4" -size -50k -delete
```

### SSL Certificate Errors (macOS)

If pretrained model downloads fail, the API falls back to random initialization. For proper weights, install certificates:

```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

## References

- Yi, X., et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." RecSys 2019. [Paper](https://dl.acm.org/doi/10.1145/3298689.3346996)
- Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. [Paper](https://arxiv.org/abs/2002.05709)
- MicroLens Dataset. Westlake University. [Website](https://recsys.westlake.edu.cn/MicroLens-Dataset/)
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." NAACL 2019. [Paper](https://arxiv.org/abs/1810.04805)
- Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS 2020. [Paper](https://arxiv.org/abs/2006.11477)
"# multimodal-youtube-recommendation-system" 
