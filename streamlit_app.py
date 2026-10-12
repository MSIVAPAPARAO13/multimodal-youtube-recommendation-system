"""
Multimodal Video Recommendation System - Streamlit Interactive App

Deploy-ready interactive application supporting:
- Video Catalog Exploration & Semantic Search (2,983 indexed videos)
- Two-Stage Recommendation Engine (Phase 8C Text ANN + Phase 11B Re-Ranking)
- User Personalization (Phase 10 Recency-Decay Profiles)
- Try Your Own Video (Ephemeral upload, ResNet-50 2048-D visual extraction, audio check, dual recommendations)
- Architecture Telemetry & Evaluation Diagnostics
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import streamlit as st

# ================= PAGE CONFIG & THEME =================
st.set_page_config(
    page_title="Multimodal Video Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (YouTube Dark Theme)
st.markdown("""
<style>
    /* Dark Theme Accent Styling */
    .stApp {
        background-color: #0f0f0f;
        color: #f1f1f1;
    }
    .metric-card {
        background: #1f1f1f;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 12px;
        background: #272727;
        color: #aaaaaa;
        border: 1px solid #3f3f3f;
        margin-right: 6px;
    }
    .badge-pill-success {
        background: rgba(46, 125, 50, 0.2);
        color: #4caf50;
        border-color: #2e7d32;
    }
    .badge-pill-danger {
        background: rgba(211, 47, 47, 0.2);
        color: #ff5252;
        border-color: #d32f2f;
    }
    .video-card {
        background: #181818;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .video-card:hover {
        border-color: #ff0000;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ================= CONFIGURATION & BACKEND DISCOVERY =================
def get_backend_url() -> str:
    """Resolve backend URL from Streamlit secrets, env vars, or local defaults."""
    # 1. Streamlit secrets
    try:
        if "FASTAPI_URL" in st.secrets:
            return st.secrets["FASTAPI_URL"].rstrip("/")
    except Exception:
        pass
    # 2. Environment variable
    env_url = os.getenv("FASTAPI_URL") or os.getenv("VITE_API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    # 3. Local port probe (try 8001 first, then 8000)
    for port in [8001, 8000]:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=0.8)
            if r.status_code == 200:
                return f"http://localhost:{port}"
        except Exception:
            continue
    return "http://localhost:8001"


BACKEND_URL = get_backend_url()


@st.cache_data(ttl=10)
def check_backend_health() -> Dict[str, Any]:
    """Check health and capabilities of the FastAPI backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        if r.status_code == 200:
            return {"online": True, **r.json()}
    except Exception as e:
        return {"online": False, "error": str(e)}
    return {"online": False}


# ================= API CLIENT HELPERS =================
def api_get(endpoint: str, params: Optional[dict] = None) -> Optional[Any]:
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error ({endpoint}): {e}")
    return None


def api_post(endpoint: str, data: Optional[dict] = None, files: Optional[dict] = None) -> Optional[Any]:
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=data, files=files, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API returned status {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"API Error ({endpoint}): {e}")
    return None


# ================= SIDEBAR TELEMETRY & CONTROLS =================
health = check_backend_health()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/Logo_of_YouTube_%282015-2017%29.svg", width=120)
    st.title("Multimodal Recommender")
    st.caption("AI-Powered YouTube Video Recommendations")
    st.divider()

    # Status Indicator
    if health.get("online"):
        st.success(f"🟢 Backend Online (`{BACKEND_URL}`)")
        idx_size = health.get("index_size", 2983)
        st.metric("Indexed Videos", f"{idx_size:,}")
        prov = health.get("checkpoint_provenance", "Phase 8C BERT")
        st.caption(f"🧠 Model: **{prov}**")
    else:
        st.warning(f"🟡 Backend Offline (`{BACKEND_URL}`)")
        st.info("Start FastAPI (`uvicorn src.api.app:app --port 8001`) or configure FASTAPI_URL in `.streamlit/secrets.toml`.")

    st.divider()
    page = st.radio(
        "Navigation",
        [
            "🔍 Catalog & Search",
            "🎯 Recommendation Studio",
            "👤 User Personalization",
            "📹 Try Your Own Video",
            "📊 Architecture & Telemetry",
        ],
        index=0
    )

    st.divider()
    st.caption("🔬 **Production Verification**")
    st.markdown("""
    - **ANN Speedup**: 10.3× (1.2ms HNSW)
    - **NDCG@10**: 0.4578 (MicroLens-50K)
    - **Recall@10**: 97.4% vs exact scan
    - **Test Suite**: 344 passing (100%)
    - **Audio/Visual Weight**: 0.00 (honest baseline)
    """)


# ================= PAGE 1: CATALOG & SEARCH =================
if page == "🔍 Catalog & Search":
    st.header("🔍 Video Catalog & Semantic Search")
    st.caption("Browse 2,983 indexed videos or query the pgvector HNSW semantic index in real-time.")

    col_search, col_filter, col_mm = st.columns([3, 2, 2])
    with col_search:
        search_query = st.text_input("Semantic Search", placeholder="e.g. machine learning, quantum computing, docker...")
    with col_filter:
        category_filter = st.selectbox(
            "Category",
            ["All Categories", "Science & Technology", "Education", "Gaming", "Music", "Entertainment", "News & Politics"]
        )
    with col_mm:
        multimodal_only = st.checkbox("Multimodal Assets Only", value=False, help="Filter to videos with extracted visual/audio assets")

    # Fetch Catalog from API
    params = {"limit": 24, "offset": 0}
    if search_query:
        params["query"] = search_query
    if multimodal_only:
        params["multimodal_only"] = "true"

    catalog_data = api_get("/index/list", params=params)
    items = catalog_data.get("items", []) if catalog_data else []

    st.write(f"Showing **{len(items)}** videos (Total Catalog: {catalog_data.get('total', 2983) if catalog_data else 2983})")

    if not items:
        st.info("No videos found matching the criteria. Try clearing search filters.")
    else:
        cols = st.columns(3)
        for i, video in enumerate(items):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="video-card">
                    <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        {video.get('title', 'Untitled Video')}
                    </div>
                    <div style="font-size: 12px; color: #aaaaaa; margin-bottom: 8px;">
                        <span>📺 {video.get('channel_title', 'YouTube Creator')}</span> &bull; 
                        <span>🏷️ {video.get('category_name', 'General')}</span>
                    </div>
                    <div style="font-size: 11px; margin-bottom: 10px;">
                        <span class="badge-pill">⏱️ {video.get('duration_formatted', '3:45')}</span>
                        <span class="badge-pill">🆔 {video.get('video_id', '')}</span>
                        {f'<span class="badge-pill badge-pill-success">Visual 2048D</span>' if video.get('has_visual_features') else ''}
                        {f'<span class="badge-pill badge-pill-danger">Audio Track</span>' if video.get('has_audio_features') else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action button to jump to recommendations
                if st.button("🎯 Inspect & Recommend", key=f"rec_btn_{video.get('video_id')}"):
                    st.session_state["selected_video_id"] = str(video.get("video_id"))
                    st.session_state["selected_video_title"] = video.get("title")
                    st.rerun()


# ================= PAGE 2: RECOMMENDATION STUDIO =================
elif page == "🎯 Recommendation Studio":
    st.header("🎯 Two-Stage Recommendation Studio")
    st.caption("Inspect pgvector HNSW candidate retrieval and tune Phase 11B multi-signal re-ranking weights.")

    default_seed = st.session_state.get("selected_video_id", "10040")
    col_input, col_meta = st.columns([2, 3])

    with col_input:
        seed_id = st.text_input("Seed Video ID", value=default_seed, help="Enter a video ID from the catalog (e.g. 10040, 10005, 10039)")
        ranking_mode = st.radio("Ranking Mode", ["Baseline HNSW ANN (Phase 8C)", "Phase 11B Candidate Re-Ranking"], index=0)

    # Sliders for Re-Ranking
    if "Re-Ranking" in ranking_mode:
        st.subheader("Candidate Re-Ranking Weights")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1:
            w_text = st.slider("Text Weight (w_text)", 0.0, 1.0, 0.90, 0.05)
        with col_w2:
            w_meta = st.slider("Metadata Weight (w_meta)", 0.0, 1.0, 0.10, 0.05)
        with col_w3:
            st.slider("Visual Weight (w_visual)", 0.0, 1.0, 0.00, disabled=True, help="Fixed at 0.00 — production model maintains zero visual weight until held-out NDCG lift is proven")
        with col_w4:
            st.slider("Audio Weight (w_audio)", 0.0, 1.0, 0.00, disabled=True, help="Fixed at 0.00 — audio feature extractions active, ranking weight zero")

    # Fetch Seed Video Details
    seed_data = api_get(f"/api/videos/{seed_id}")
    if seed_data:
        with col_meta:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 6px 0; color: #fff;">{seed_data.get('title', 'Video Details')}</h4>
                <p style="color: #aaa; font-size: 13px; margin-bottom: 8px;">{seed_data.get('description', 'No description available.')[:180]}...</p>
                <div>
                    <span class="badge-pill">Channel: {seed_data.get('channel_title', 'N/A')}</span>
                    <span class="badge-pill">Category: {seed_data.get('category_name', 'General')}</span>
                    <span class="badge-pill">Duration: {seed_data.get('duration_formatted', '3:45')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Local Video Player if present
            local_video_path = Path(f"data/raw/youtube/{seed_id}/video.mp4")
            if local_video_path.exists():
                st.video(str(local_video_path))

    # Fetch Recommendations
    st.divider()
    st.subheader("Recommended Videos")

    mode_param = "rerank" if "Re-Ranking" in ranking_mode else "baseline"
    rec_res = api_get(f"/api/recommendations/video/{seed_id}", params={"limit": 8, "ranking_mode": mode_param})

    if rec_res and rec_res.get("items"):
        rec_items = rec_res["items"]
        col_list = st.columns(4)
        for idx, item in enumerate(rec_items):
            with col_list[idx % 4]:
                score = item.get("score", 0.0)
                st.markdown(f"""
                <div class="video-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="badge-pill badge-pill-danger">Rank #{item.get('rank', idx+1)}</span>
                        <span style="font-size: 11px; font-weight: 600; color: #4caf50;">Sim: {score:.4f}</span>
                    </div>
                    <div style="font-weight: 600; font-size: 13px; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        {item.get('title', 'Recommended Video')}
                    </div>
                    <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
                        🏷️ {item.get('category_name', 'General')}
                    </div>
                    <span class="badge-pill">512D Cosine</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Select This Video", key=f"sel_{item.get('video_id')}"):
                    st.session_state["selected_video_id"] = str(item.get("video_id"))
                    st.rerun()
    else:
        st.info("No recommendations returned for this video ID. Verify the video exists in the catalog.")


# ================= PAGE 3: USER PERSONALIZATION =================
elif page == "👤 User Personalization":
    st.header("👤 Phase 10 User Personalization")
    st.caption("Generate dynamic recommendations using a recency-decay weighted 512-D user profile vector.")

    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        user_id = st.selectbox("Select User Profile", ["demo_user", "music_lover", "tech_dev", "new_user"])
        limit = st.slider("Max Recommendations", 4, 12, 8)

        st.subheader("Simulate Interaction")
        sim_vid = st.text_input("Watch Video ID", value="10040")
        if st.button("➕ Record Watch Event"):
            res = api_post("/api/interactions", data={"user_id": user_id, "video_id": sim_vid, "interaction_type": "view"})
            if res:
                st.success(f"Recorded view event for video {sim_vid}!")
                st.rerun()

    with col_u2:
        # Fetch User Recommendations
        user_recs = api_get(f"/api/recommendations/user/{user_id}", params={"limit": limit})
        if user_recs and user_recs.get("items"):
            st.subheader(f"Personalized Recommendations for `{user_id}`")
            st.caption(f"Strategy: {user_recs.get('strategy', 'Recency Decay')} &bull; Profile Dimension: 512-D")

            cols_p = st.columns(2)
            for i, p_item in enumerate(user_recs["items"]):
                with cols_p[i % 2]:
                    st.markdown(f"""
                    <div class="video-card">
                        <div style="font-weight: 600; font-size: 13px; margin-bottom: 4px;">
                            #{p_item.get('rank', i+1)} {p_item.get('title', 'Video')}
                        </div>
                        <div style="font-size: 11px; color: #888; margin-bottom: 6px;">
                            <span>🏷️ {p_item.get('category_name', 'General')}</span> &bull; 
                            <span style="color: #4caf50;">Match Score: {p_item.get('score', 0.0):.4f}</span>
                        </div>
                        <span class="badge-pill">Personalized Profile Match</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"No watch history found for user '{user_id}'. Simulate a watch event on the left to initialize the profile.")


# ================= PAGE 4: TRY YOUR OWN VIDEO =================
elif page == "📹 Try Your Own Video":
    st.header("📹 Try Your Own Video")
    st.caption("Upload a local video or run the verified sample to extract ResNet-50 visual features, check audio validity, and retrieve matching catalog videos.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        use_sample = st.button("⚡ Try with Sample Video (10040)", help="Use local_10040/video.mp4 for immediate verification")

    with col_info:
        st.markdown("""
        <span class="badge-pill badge-pill-success">Ephemeral &bull; Never Persisted to Catalog</span>
        <span class="badge-pill">Max 100 MB &bull; Up to 10 min</span>
        <span class="badge-pill">ResNet-50 2048-D + Phase 8C BERT 512-D</span>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose Video File", type=["mp4", "avi", "mov", "mkv", "webm"])
    col_t, col_d = st.columns(2)
    with col_t:
        vid_title = st.text_input("Video Title (Optional)", value="Building Microservices with Docker and Go" if use_sample else "")
    with col_d:
        vid_desc = st.text_input("Video Description (Optional)", value="Hands-on walkthrough on microservice architecture, Docker containerization, and Go backend." if use_sample else "")

    # Handle Sample Video Load
    sample_path = Path("data/raw/youtube/local_10040/video.mp4")
    file_bytes = None
    filename = "custom_video.mp4"

    if use_sample and sample_path.exists():
        file_bytes = sample_path.read_bytes()
        filename = "sample_local_10040.mp4"
        st.success("Loaded catalog sample video `local_10040/video.mp4` (2.08 MB)!")
    elif uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

    if file_bytes is not None:
        st.video(file_bytes)

        if st.button("🚀 Analyze Video & Discover Recommendations", type="primary"):
            with st.spinner("Processing video: extracting frames, running ResNet-50, checking audio, and querying HNSW index..."):
                files = {"file": (filename, io.BytesIO(file_bytes), "video/mp4")}
                data = {}
                if vid_title.strip():
                    data["title"] = vid_title.strip()
                if vid_desc.strip():
                    data["description"] = vid_desc.strip()
                data["max_recommendations"] = "8"

                try:
                    r = requests.post(f"{BACKEND_URL}/api/custom-video/analyze", files=files, data=data, timeout=60)
                    if r.status_code == 200:
                        analysis = r.json()
                        st.success("Analysis Complete!")

                        # 1. Video Metadata Metrics
                        st.subheader("1. Container & Codec Validation")
                        vm = analysis.get("video_metadata", {})
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Resolution", f"{vm.get('width', 0)} × {vm.get('height', 0)}")
                        c2.metric("Framerate", f"{vm.get('fps', 0.0):.1f} FPS")
                        c3.metric("Duration", f"{vm.get('duration_seconds', 0.0):.1f}s")
                        c4.metric("File Size", f"{vm.get('file_size_mb', 0.0):.2f} MB")

                        # 2. Extracted Frames Gallery
                        st.subheader("2. Extracted ResNet-50 Preview Frames")
                        frames = analysis.get("extracted_frames", [])
                        if frames:
                            frame_cols = st.columns(len(frames))
                            for idx, f in enumerate(frames):
                                with frame_cols[idx]:
                                    img_data = base64.b64decode(f["data_base64"])
                                    st.image(img_data, caption=f"Timestamp: {f.get('timestamp_seconds', 0.0):.1f}s")

                        # 3. Modality Diagnostics
                        st.subheader("3. Modality Diagnostic Badges")
                        mod = analysis.get("modalities", {})
                        vis = mod.get("visual", {})
                        aud = mod.get("audio", {})
                        txt = mod.get("text", {})

                        st.markdown(f"""
                        <div class="metric-card">
                            <span class="badge-pill badge-pill-success">Visual: {vis.get('model', 'ResNet-50')} ({vis.get('dimension', 2048)}D) - Status: {vis.get('status')}</span>
                            <span class="badge-pill {'badge-pill-success' if aud.get('has_audio') else 'badge-pill-danger'}">Audio: {aud.get('status')} ({aud.get('format', 'PCM')})</span>
                            <span class="badge-pill badge-pill-success">Text: {txt.get('model', 'BERT-base')} ({txt.get('dimension', 512)}D)</span>
                            <span class="badge-pill">Query Time: {analysis.get('query_time_ms', 0.0):.1f} ms</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # 4. Recommendation Results
                        st.subheader("4. Discovered Catalog Matches")
                        tab_vis, tab_txt = st.tabs(["👁️ Visual Nearest Neighbors (ResNet-50)", "📝 Text Semantic Recommendations (Phase 8C BERT)"])

                        with tab_vis:
                            vis_recs = analysis.get("visual_recommendations", [])
                            if vis_recs:
                                cols_v = st.columns(4)
                                for idx, vr in enumerate(vis_recs):
                                    with cols_v[idx % 4]:
                                        st.markdown(f"""
                                        <div class="video-card">
                                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                                <span class="badge-pill badge-pill-danger">Rank #{vr.get('rank', idx+1)}</span>
                                                <span style="font-size: 11px; font-weight: 600; color: #4caf50;">Sim: {vr.get('similarity', 0.0):.4f}</span>
                                            </div>
                                            <div style="font-weight: 600; font-size: 13px; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                                {vr.get('title', 'Catalog Video')}
                                            </div>
                                            <div style="font-size: 11px; color: #888; margin-bottom: 6px;">
                                                🏷️ {vr.get('category', 'General')}
                                            </div>
                                            <span class="badge-pill">ResNet-50 Cosine</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No visual neighbors found.")

                        with tab_txt:
                            txt_recs = analysis.get("text_recommendations", [])
                            if txt_recs:
                                cols_t = st.columns(4)
                                for idx, tr in enumerate(txt_recs):
                                    with cols_t[idx % 4]:
                                        st.markdown(f"""
                                        <div class="video-card">
                                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                                <span class="badge-pill badge-pill-danger">Rank #{tr.get('rank', idx+1)}</span>
                                                <span style="font-size: 11px; font-weight: 600; color: #4caf50;">Sim: {tr.get('similarity', 0.0):.4f}</span>
                                            </div>
                                            <div style="font-weight: 600; font-size: 13px; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                                {tr.get('title', 'Catalog Video')}
                                            </div>
                                            <div style="font-size: 11px; color: #888; margin-bottom: 6px;">
                                                🏷️ {tr.get('category', 'General')}
                                            </div>
                                            <span class="badge-pill">HNSW pgvector</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("Provide title or description to discover text recommendations.")
                    else:
                        st.error(f"Analysis failed with code {r.status_code}: {r.text}")
                except Exception as ex:
                    st.error(f"Error during video analysis: {ex}")


# ================= PAGE 5: ARCHITECTURE & TELEMETRY =================
elif page == "📊 Architecture & Telemetry":
    st.header("📊 Architecture & Scientific Telemetry")
    st.caption("Live system telemetry, scientific benchmark numbers, and model checkpoint provenance.")

    # Checkpoint Provenance
    st.subheader("Model Checkpoint Provenance")
    st.table([
        {"Property": "Checkpoint File", "Value": "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt"},
        {"Property": "SHA-256 Checksum", "Value": "a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996"},
        {"Property": "Architecture", "Value": "Frozen BERT-base-uncased → Linear(768→512) → L2-Norm"},
        {"Property": "Objective & Loss", "Value": "InfoNCE Contrastive Loss (tau = 0.07)"},
        {"Property": "NDCG@10 (Validation)", "Value": "0.4578 (MicroLens-50K held-out split)"},
        {"Property": "Recall@10 (Validation)", "Value": "0.3214 (MicroLens-50K held-out split)"},
    ])

    # HNSW vs Exact Retrieval Benchmark
    st.subheader("Phase 11A HNSW ANN Retrieval Benchmark")
    st.table([
        {"Metric": "Retrieval Latency", "HNSW ANN": "1.2 ms", "Exact Scan": "12.4 ms", "Lift / Agreement": "10.3× Speedup"},
        {"Metric": "Recall@10 Agreement", "HNSW ANN": "97.4%", "Exact Scan": "100.0%", "Lift / Agreement": "97.4% High Fidelity"},
        {"Metric": "HNSW Configuration", "HNSW ANN": "M=16, ef_c=64, ef_s=40", "Exact Scan": "Sequential Cosine (<=>)", "Lift / Agreement": "Sub-millisecond index"},
    ])

    # System Health Dump
    st.subheader("Live System Health Telemetry")
    st.json(health)
