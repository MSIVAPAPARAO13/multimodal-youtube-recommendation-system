"""
HTTP Client for YouTube Data API v3.
Encapsulates communication with Google APIs, quota error handling, and retries.
"""

import logging
import time
from typing import Any, Optional
import httpx

from src.integrations.youtube.exceptions import (
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeConfigurationError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
    YouTubeQuotaExceededError,
)

logger = logging.getLogger(__name__)


class YouTubeAPIClient:
    """
    Client for interacting with the YouTube Data API v3.

    Features:
        - Strict timeout enforcement
        - Retry logic for transient 5xx / connection errors
        - No retries on permanent 4xx errors
        - API key protection: Credentials are never logged or exposed in exceptions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://www.googleapis.com/youtube/v3",
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _get_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "MultimodalVideoRecommender/3.0",
        }

    def fetch_video_data(self, video_id: str) -> dict[str, Any]:
        """
        Fetch video metadata from YouTube Data API v3 videos.list endpoint.

        Args:
            video_id: 11-character video ID

        Returns:
            Dictionary representing the single video resource item from YouTube API.

        Raises:
            YouTubeConfigurationError: If API key is missing.
            YouTubeNotFoundError: If video does not exist.
            YouTubePrivateVideoError: If video is private.
            YouTubeQuotaExceededError: If daily quota is exhausted.
            YouTubeAuthenticationError: If API key is invalid.
            YouTubeAPIError: For other API failures.
        """
        if not self.api_key or not self.api_key.strip():
            raise YouTubeConfigurationError(
                "YouTube API key is missing. Set YOUTUBE_API_KEY environment variable."
            )

        endpoint = f"{self.base_url}/videos"
        # Minimum required parts per TASK 4
        params = {
            "part": "snippet,contentDetails,statistics,status",
            "id": video_id,
            "key": self.api_key,
        }

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(
                        endpoint,
                        params=params,
                        headers=self._get_headers()
                    )

                status_code = response.status_code

                # Handle successful response
                if status_code == 200:
                    try:
                        data = response.json()
                    except Exception as e:
                        raise YouTubeAPIError(f"Malformed JSON response from YouTube API: {e}", status_code=502)

                    items = data.get("items", [])
                    if not items:
                        raise YouTubeNotFoundError(video_id, f"Video '{video_id}' not found on YouTube.")

                    item = items[0]
                    # Check privacy / availability in status
                    status_info = item.get("status", {})
                    privacy_status = status_info.get("privacyStatus", "public")
                    if privacy_status == "private":
                        raise YouTubePrivateVideoError(video_id, f"Video '{video_id}' is private.")

                    return item

                # Handle 4xx Client Errors (Permanent: do not retry)
                elif 400 <= status_code < 500:
                    self._handle_client_error(response, video_id)

                # Handle 5xx Server Errors (Transient: retry with backoff)
                else:
                    last_error = YouTubeAPIError(
                        f"YouTube API server error ({status_code})",
                        status_code=status_code
                    )

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(
                    "Transient network error fetching YouTube video '%s' (attempt %d/%d): %s",
                    video_id, retries + 1, self.max_retries + 1, type(e).__name__
                )
                last_error = YouTubeAPIError(f"Network error communicating with YouTube API: {type(e).__name__}", status_code=504)

            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.backoff_factor * (2 ** (retries - 1)))

        # Retries exhausted
        if last_error:
            raise last_error
        raise YouTubeAPIError(f"Failed to fetch metadata for video '{video_id}' after retries.", status_code=500)

    def _handle_client_error(self, response: httpx.Response, video_id: str):
        """Parse error details and raise corresponding domain exception."""
        status_code = response.status_code
        try:
            error_data = response.json().get("error", {})
            errors_list = error_data.get("errors", [])
            reason = errors_list[0].get("reason", "") if errors_list else ""
            error_message = error_data.get("message", "")
        except Exception:
            reason = ""
            error_message = response.text[:200]

        # Quota exceeded
        if reason in ("quotaExceeded", "dailyLimitExceeded") or "quota" in error_message.lower():
            raise YouTubeQuotaExceededError("YouTube API quota exceeded. Please try again tomorrow.")

        # Authentication / API key issues
        if reason in ("keyInvalid", "badRequest", "forbidden") and (
            "key" in error_message.lower() or status_code in (400, 401, 403)
        ):
            if reason in ("keyInvalid", "apiNotActivated"):
                raise YouTubeAuthenticationError("YouTube API key is invalid or not authorized.")

        if status_code == 404:
            raise YouTubeNotFoundError(video_id, f"Video '{video_id}' not found.")

        if status_code == 403:
            raise YouTubePrivateVideoError(video_id, f"Access to video '{video_id}' is restricted or private.")

        raise YouTubeAPIError(
            f"YouTube API returned HTTP {status_code}: {error_message}",
            status_code=status_code
        )
