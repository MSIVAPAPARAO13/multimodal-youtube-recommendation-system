# YouTube Data API Integration Architecture & Specification

**Phase 3 — YouTube Data API Integration**  
**Date:** October 8, 2026  
**Status:** **PHASE 3 STATUS: PASS**

---

## 1. System Architecture

The YouTube Data API integration serves as an isolated external metadata ingestion module for the multimodal recommendation system. It translates incoming user-supplied video URLs or video IDs into normalized internal metadata schemas without directly exposing raw Google API structures or credentials.

```
┌────────────────────────────────────────────────────────┐
│  User Request / Ingestion Input                        │
│  (YouTube URL or 11-char Video ID)                     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/integrations/youtube/parser.py                    │
│  - extract_video_id()                                  │
│  - Validates 11-char format (regex: [a-zA-Z0-9_-]{11}) │
│  - Rejects non-YouTube domains / malformed strings     │
│  - Zero API calls performed during parsing             │
└──────────────────────────┬─────────────────────────────┘
                           │ video_id
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/integrations/youtube/service.py                   │
│  - InMemoryMetadataCache (1-hour TTL, max 1000 items)  │
│  - Cache Hit -> Return cached YouTubeVideoMetadata     │
└──────────────────────────┬─────────────────────────────┘
                           │ (Cache Miss)
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/integrations/youtube/client.py                    │
│  - YouTubeAPIClient (HTTP GET /v3/videos)              │
│  - Minimal parts requested:                            │
│    part=snippet,contentDetails,statistics,status       │
│  - Timeout (10s), Retries with exponential backoff     │
│  - Quota, Auth, and Privacy status handling            │
└──────────────────────────┬─────────────────────────────┘
                           │ Raw JSON item
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/integrations/youtube/schemas.py                   │
│  - Normalized YouTubeVideoMetadata                     │
│  - ISO 8601 duration parser (PT#H#M#S -> seconds)      │
│  - Nested API response mapping                         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI: GET /api/youtube/video/{video_id}            │
│  - Offloaded to worker threadpool                      │
│  - API key strictly redacted from output and headers   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Configuration & Environment Variables

Settings are centralized in `src/db/config.py` using typed `pydantic-settings`:

| Setting Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `YOUTUBE_API_KEY` | `Optional[str]` | `None` | Google Cloud API key with YouTube Data API v3 enabled |
| `YOUTUBE_API_BASE_URL` | `str` | `https://www.googleapis.com/youtube/v3` | Canonical Google API endpoint |
| `YOUTUBE_API_TIMEOUT` | `float` | `10.0` | HTTP request timeout in seconds |
| `YOUTUBE_API_ENABLED` | `bool` | `true` | Feature flag to enable or disable external calls |

### Environment Setup (`.env.example` / `.env.template`):
```bash
# YouTube Data API v3 Configuration
YOUTUBE_API_KEY=your_youtube_api_key_here
YOUTUBE_API_BASE_URL=https://www.googleapis.com/youtube/v3
YOUTUBE_API_TIMEOUT=10.0
YOUTUBE_API_ENABLED=true
```

> [!CAUTION]
> Never commit `.env` or hardcode real API keys into repository source code.

---

## 3. URL Parsing & Validation

The parser module (`src/integrations/youtube/parser.py`) validates and normalizes all recognized YouTube URL patterns into a canonical 11-character video ID:

| Supported Input Format | Example Input | Extracted Video ID |
| :--- | :--- | :--- |
| **Standard Watch** | `https://www.youtube.com/watch?v=dQw4w9WgXcQ` | `dQw4w9WgXcQ` |
| **Domain without www** | `https://youtube.com/watch?v=dQw4w9WgXcQ` | `dQw4w9WgXcQ` |
| **With Query Params** | `https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=shared` | `dQw4w9WgXcQ` |
| **Short Domain** | `https://youtu.be/dQw4w9WgXcQ` | `dQw4w9WgXcQ` |
| **Shorts Format** | `https://www.youtube.com/shorts/dQw4w9WgXcQ` | `dQw4w9WgXcQ` |
| **Embed Format** | `https://www.youtube.com/embed/dQw4w9WgXcQ` | `dQw4w9WgXcQ` |
| **Raw Video ID** | `dQw4w9WgXcQ` | `dQw4w9WgXcQ` |

**Rejection Policy:**
- Malformed URLs, non-YouTube domains (e.g., `vimeo.com`), empty strings, or IDs not matching `^[a-zA-Z0-9_-]{11}$` raise `YouTubeInvalidVideoIdError` with HTTP 400.
- No network requests are made during URL parsing.

---

## 4. Metadata Schema & Field Normalization

Raw responses from Google's `videos.list` endpoint are normalized into `YouTubeVideoMetadata` (`src/integrations/youtube/schemas.py`):

```python
class YouTubeVideoMetadata(BaseModel):
    video_id: str                      # 11-char canonical ID
    title: str                         # Video title
    description: str                   # Clean description string
    channel_id: Optional[str]          # Channel ID
    channel_title: Optional[str]       # Channel display name
    published_at: Optional[str]        # ISO 8601 timestamp
    category_id: Optional[str]         # YouTube category ID
    tags: list[str]                    # Video tags
    thumbnail_default: Optional[str]   # 120x90 URL
    thumbnail_medium: Optional[str]    # 320x180 URL
    thumbnail_high: Optional[str]      # 480x360 URL
    duration: Optional[str]            # ISO 8601 string (e.g. PT3M33S)
    duration_seconds: Optional[int]    # Total seconds (e.g. 213)
    view_count: int                    # Engagement view count
    like_count: int                    # Engagement like count
    comment_count: int                 # Engagement comment count
    privacy_status: Optional[str]      # public, unlisted, private
```

---

## 5. API Endpoint Specification

### `GET /api/youtube/video/{video_id:path}`

Accepts either an 11-character video ID or a URL-encoded YouTube URL.

#### Successful Response (HTTP 200 OK):
```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
  "description": "The official video for Never Gonna Give You Up by Rick Astley",
  "channel_title": "Rick Astley",
  "published_at": "2009-10-25T06:57:33Z",
  "category_id": "10",
  "tags": ["rick astley", "never gonna give you up"],
  "thumbnail": {
    "default": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg",
    "medium": "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg",
    "high": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
  },
  "duration": "PT3M33S",
  "statistics": {
    "views": 1500000000,
    "likes": 17000000,
    "comments": 2500000
  }
}
```

---

## 6. Error Handling & Status Code Mapping

Domain exceptions in `src/integrations/youtube/exceptions.py` are mapped to HTTP status codes:

| Domain Exception | HTTP Status Code | Cause / Trigger |
| :--- | :---: | :--- |
| `YouTubeInvalidVideoIdError` | **400 Bad Request** | Malformed URL or non-11-character video ID |
| `YouTubeAuthenticationError` | **401 Unauthorized** | Invalid API key (`keyInvalid`, `apiNotActivated`) |
| `YouTubePrivateVideoError` | **403 Forbidden** | Private, restricted, or region-blocked video |
| `YouTubeNotFoundError` | **404 Not Found** | Video not found in YouTube Data API (`items: []`) |
| `YouTubeQuotaExceededError` | **429 Too Many Requests** | Google API daily quota reached (`quotaExceeded`) |
| `YouTubeConfigurationError` | **500 Server Error** | `YOUTUBE_API_KEY` missing from environment |
| `YouTubeAPIError` | **502 / 504 Gateway Error** | Upstream Google service error or timeout |
| *(Feature Disabled)* | **503 Service Unavailable** | `YOUTUBE_API_ENABLED=false` in configuration |

---

## 7. Quota Considerations & Caching

- **Quota Consumption:** The `videos.list` endpoint costs **1 quota unit** per request (with a standard Google default project quota of 10,000 units/day).
- **Application-Level In-Memory Cache:** `InMemoryMetadataCache` caches normalized metadata with a **1-hour TTL** (configurable) and an LRU-style maximum capacity of 1,000 items. Repeated recommendations or queries for the same video cost 0 quota units.
- **Phase 4 Preparation:** Persistent storage in PostgreSQL (`videos.youtube_video_id` column added in Phase 3) will be leveraged in Phase 4 for persistent caching.

---

## 8. Security Auditing

1. **Credentials Isolation:** `YOUTUBE_API_KEY` is loaded exclusively via environment variables / `.env`.
2. **Git Safety:** `.env` is listed in `.gitignore`. Only `.env.example` and `.env.template` are tracked.
3. **Redaction:** The API key parameter is stripped from error logs, domain exception messages, and endpoint response bodies and headers.
4. **URL Injection Protection:** The parser strictly whitelist-validates YouTube hostnames and extracts only regex-verified 11-character IDs, completely preventing server-side request forgery (SSRF).

---

## 9. Automated Testing

Comprehensive test suites added in `tests/`:
- `tests/test_youtube_parser.py`: 8 test cases verifying all valid URL variants and rejection rules.
- `tests/test_youtube_client.py`: 8 test cases covering 200 OK, empty items (404), private (403), quota (429), auth error (401), timeouts (504), and malformed JSON.
- `tests/test_youtube_service.py`: 5 test cases testing duration parsing, normalization, caching, and eviction.
- `tests/test_youtube_api.py`: 9 test cases testing endpoint routing, full URLs, error code mappings, and API key leak protection.

**Total Tests:** **63 PASSED, 0 FAILED** (33 Phase 1+2 tests + 30 Phase 3 YouTube tests).
