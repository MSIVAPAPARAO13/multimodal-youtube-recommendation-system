"""
YouTube Semantic Text Processor

Provides canonical data modeling, Unicode and whitespace normalization,
transcript artifact cleanup (timestamps, sound cues, duplicate caption lines),
and deterministic section-structured formatting for YouTube multimodal text.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Optional, Union
import unicodedata

logger = logging.getLogger(__name__)

# Documented maximum character constraints
DEFAULT_MAX_TITLE_CHARS = 300
DEFAULT_MAX_DESCRIPTION_CHARS = 1500
DEFAULT_MAX_TAGS_CHARS = 500
DEFAULT_MAX_TRANSCRIPT_CHARS = 20000


def normalize_unicode(text: Optional[str]) -> str:
    """Normalize text using Unicode NFKC form to unify composite characters."""
    if not text:
        return ""
    # Standard NFKC normalization handles compatibility and composition
    normalized = unicodedata.normalize("NFKC", str(text))
    # Remove control characters except standard whitespace (newlines, tabs)
    return "".join(ch for ch in normalized if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))


def normalize_whitespace(text: Optional[str]) -> str:
    """
    Collapse excessive spaces and tabs while preserving paragraph structure
    and meaningful punctuation.
    """
    if not text:
        return ""
    # Replace non-breaking spaces and carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    # Replace horizontal whitespace (tabs, multiple spaces) with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_tags(tags: Optional[Union[list[Any], str]]) -> tuple[list[str], str]:
    """
    Normalize list or string of tags into cleaned list and joined comma-separated string.
    """
    if tags is None:
        return [], ""

    clean_tags: list[str] = []
    if isinstance(tags, list):
        for t in tags:
            if t is not None:
                t_str = normalize_whitespace(normalize_unicode(str(t)))
                if t_str and t_str not in clean_tags:
                    clean_tags.append(t_str)
    elif isinstance(tags, str):
        # Could be comma-separated or space-separated tags
        raw = normalize_whitespace(normalize_unicode(tags))
        if "," in raw:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in raw.split() if p.strip()]
        for p in parts:
            if p and p not in clean_tags:
                clean_tags.append(p)
    else:
        s = normalize_whitespace(normalize_unicode(str(tags)))
        if s:
            clean_tags.append(s)

    joined = ", ".join(clean_tags)
    return clean_tags, joined


# Regular expressions for cleaning transcript artifacts
RE_VTT_TIMESTAMPS = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?",
    re.IGNORECASE
)
RE_BRACKETED_TIMESTAMPS = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]")
RE_PAREN_TIMESTAMPS = re.compile(r"\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)")
RE_STANDALONE_TIMESTAMPS = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$", re.MULTILINE)
RE_HTML_VTT_TAGS = re.compile(r"<\/?(?:c|v|lang|b|i|u|span|\d{1,2}:\d{2}[^>]*)>", re.IGNORECASE)
RE_SOUND_EFFECTS = re.compile(
    r"\[\s*(?:music|applause|laughter|cheering|silence|inaudible|whispering|sighs?|coughing|groans?|screams?)\s*\]",
    re.IGNORECASE
)
RE_SOUND_EFFECTS_PAREN = re.compile(
    r"\(\s*(?:music|applause|laughter|cheering|silence|inaudible|whispering|sighs?|coughing|groans?|screams?)\s*\)",
    re.IGNORECASE
)


def clean_transcript(
    transcript: Optional[str],
    max_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS
) -> str:
    """
    Clean YouTube transcript text:
    - Strips SRT / WebVTT timing cues and timestamp metadata
    - Strips audio description tags (e.g. [Music], [Applause])
    - Removes repeated / duplicate adjacent caption lines
    - Normalizes Unicode and whitespace
    - Preserves semantic text and meaningful punctuation
    - Bounded to max_chars without cutting words in half

    Args:
        transcript: Raw transcript or subtitle text
        max_chars: Maximum character limit for cleaned transcript

    Returns:
        Cleaned, readable transcript string. Returns "" if transcript is missing or empty.
    """
    if not transcript:
        return ""

    text = normalize_unicode(transcript)
    if not text.strip():
        return ""

    # 1. Remove SRT / WebVTT timestamp ranges
    text = RE_VTT_TIMESTAMPS.sub(" ", text)

    # 2. Remove bracketed / parenthesized timestamps e.g. [01:23] or (12:34)
    text = RE_BRACKETED_TIMESTAMPS.sub(" ", text)
    text = RE_PAREN_TIMESTAMPS.sub(" ", text)
    text = RE_STANDALONE_TIMESTAMPS.sub(" ", text)

    # 3. Strip HTML / WebVTT styling cues
    text = RE_HTML_VTT_TAGS.sub(" ", text)

    # 4. Remove sound effect annotations like [Music], (Applause)
    text = RE_SOUND_EFFECTS.sub(" ", text)
    text = RE_SOUND_EFFECTS_PAREN.sub(" ", text)

    # 5. Handle repeated caption lines (frequent in YouTube auto-generated captions)
    lines = [line.strip() for line in text.split("\n")]
    deduped_lines: list[str] = []

    for line in lines:
        if not line:
            continue
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        if not cleaned_line:
            continue

        if not deduped_lines:
            deduped_lines.append(cleaned_line)
            continue

        prev_line = deduped_lines[-1]

        # Exact repeat
        if cleaned_line.lower() == prev_line.lower():
            continue

        # Rolling caption prefix: prev_line is strict prefix of current line
        if cleaned_line.lower().startswith(prev_line.lower() + " "):
            # Replace previous with the more complete current line
            deduped_lines[-1] = cleaned_line
            continue

        # Or current line is strict prefix of previous line
        if prev_line.lower().startswith(cleaned_line.lower() + " "):
            continue

        deduped_lines.append(cleaned_line)

    reconstructed = " ".join(deduped_lines)
    reconstructed = normalize_whitespace(reconstructed)

    # 6. Truncate safely at word boundary if exceeding max_chars
    if len(reconstructed) > max_chars:
        cutoff = reconstructed[:max_chars]
        last_space = cutoff.rfind(" ")
        if last_space > int(max_chars * 0.8):
            reconstructed = cutoff[:last_space].strip()
        else:
            reconstructed = cutoff.strip()

    return reconstructed


@dataclass
class YouTubeTextInput:
    """
    Canonical data representation for YouTube video text metadata.

    Fields:
        title: Video title (required)
        description: Video description (optional)
        tags: List or string of tags (optional)
        transcript: Spoken captions / transcript (optional)
    """
    title: str
    description: Optional[str] = None
    tags: Optional[Union[list[str], str]] = None
    transcript: Optional[str] = None

    # Normalized internal representations
    normalized_title: str = field(init=False)
    normalized_description: str = field(init=False)
    normalized_tags: list[str] = field(init=False)
    normalized_tags_str: str = field(init=False)
    normalized_transcript: str = field(init=False)

    def __post_init__(self):
        # 1. Title normalization (preserve punctuation, collapse excessive whitespace)
        raw_title = str(self.title or "")
        self.normalized_title = normalize_whitespace(normalize_unicode(raw_title))
        if len(self.normalized_title) > DEFAULT_MAX_TITLE_CHARS:
            self.normalized_title = self.normalized_title[:DEFAULT_MAX_TITLE_CHARS].strip()

        # 2. Description normalization
        raw_desc = str(self.description or "") if self.description is not None else ""
        clean_desc = normalize_whitespace(normalize_unicode(raw_desc))
        if len(clean_desc) > DEFAULT_MAX_DESCRIPTION_CHARS:
            cutoff = clean_desc[:DEFAULT_MAX_DESCRIPTION_CHARS]
            last_space = cutoff.rfind(" ")
            if last_space > int(DEFAULT_MAX_DESCRIPTION_CHARS * 0.8):
                clean_desc = cutoff[:last_space].strip()
            else:
                clean_desc = cutoff.strip()
        self.normalized_description = clean_desc

        # 3. Tags normalization
        self.normalized_tags, self.normalized_tags_str = normalize_tags(self.tags)
        if len(self.normalized_tags_str) > DEFAULT_MAX_TAGS_CHARS:
            self.normalized_tags_str = self.normalized_tags_str[:DEFAULT_MAX_TAGS_CHARS].strip()

        # 4. Transcript normalization & artifact cleanup
        raw_trans = str(self.transcript or "") if self.transcript is not None else ""
        self.normalized_transcript = clean_transcript(raw_trans, max_chars=DEFAULT_MAX_TRANSCRIPT_CHARS)

    @classmethod
    def from_any(cls, item: Union[str, dict[str, Any], "YouTubeTextInput"]) -> "YouTubeTextInput":
        """
        Construct a YouTubeTextInput from a string (treated as title),
        a dictionary, or an existing YouTubeTextInput instance.
        """
        if isinstance(item, cls):
            return item
        if isinstance(item, str):
            return cls(title=item)
        if isinstance(item, dict):
            return cls(
                title=str(item.get("title", "")),
                description=item.get("description"),
                tags=item.get("tags"),
                transcript=item.get("transcript")
            )
        return cls(title=str(item or ""))

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dictionary payload."""
        return {
            "title": self.normalized_title,
            "description": self.normalized_description or None,
            "tags": self.normalized_tags if self.normalized_tags else None,
            "transcript": self.normalized_transcript or None,
        }


@dataclass
class TextQualityDiagnostics:
    """Metadata diagnostics for assessing text input richness and processing quality."""
    token_count: int = 0
    chunk_count: int = 0
    transcript_available: bool = False
    description_available: bool = False
    tags_available: bool = False
    transcript_used: bool = False
    title_length: int = 0
    description_length: int = 0
    tags_count: int = 0
    transcript_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "transcript_available": self.transcript_available,
            "description_available": self.description_available,
            "tags_available": self.tags_available,
            "transcript_used": self.transcript_used,
            "title_length": self.title_length,
            "description_length": self.description_length,
            "tags_count": self.tags_count,
            "transcript_length": self.transcript_length,
        }


class TextProcessor:
    """
    Constructs structured deterministic text representations with explicit section markers:

    TITLE:
    ...

    DESCRIPTION:
    ...

    TAGS:
    ...

    TRANSCRIPT:
    ...
    """

    def __init__(
        self,
        max_title_chars: int = DEFAULT_MAX_TITLE_CHARS,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        max_tags_chars: int = DEFAULT_MAX_TAGS_CHARS,
        max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS
    ):
        self.max_title_chars = max_title_chars
        self.max_description_chars = max_description_chars
        self.max_tags_chars = max_tags_chars
        self.max_transcript_chars = max_transcript_chars

    def process(
        self,
        item: Union[str, dict[str, Any], YouTubeTextInput]
    ) -> tuple[str, TextQualityDiagnostics, YouTubeTextInput]:
        """
        Process any text input into deterministic section-structured string,
         همراه with diagnostics and the canonical YouTubeTextInput.

        Returns:
            structured_text: Deterministically formatted string with explicit sections
            diagnostics: TextQualityDiagnostics object
            canonical_input: Normalized YouTubeTextInput
        """
        canonical = YouTubeTextInput.from_any(item)

        sections: list[str] = []

        # 1. TITLE (always preserved if available)
        if canonical.normalized_title:
            sections.append(f"TITLE:\n{canonical.normalized_title}")

        # 2. DESCRIPTION
        desc_available = bool(canonical.normalized_description)
        if desc_available:
            sections.append(f"DESCRIPTION:\n{canonical.normalized_description}")

        # 3. TAGS
        tags_available = bool(canonical.normalized_tags_str)
        if tags_available:
            sections.append(f"TAGS:\n{canonical.normalized_tags_str}")

        # 4. TRANSCRIPT
        trans_available = bool(canonical.normalized_transcript)
        if trans_available:
            sections.append(f"TRANSCRIPT:\n{canonical.normalized_transcript}")

        structured_text = "\n\n".join(sections) if sections else ""

        diagnostics = TextQualityDiagnostics(
            transcript_available=trans_available,
            description_available=desc_available,
            tags_available=tags_available,
            transcript_used=trans_available,
            title_length=len(canonical.normalized_title),
            description_length=len(canonical.normalized_description),
            tags_count=len(canonical.normalized_tags),
            transcript_length=len(canonical.normalized_transcript),
        )

        return structured_text, diagnostics, canonical

    def format_structured_text(
        self,
        item: Union[str, dict[str, Any], YouTubeTextInput]
    ) -> str:
        """Convenience method returning only the structured section text."""
        structured, _, _ = self.process(item)
        return structured
