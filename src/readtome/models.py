from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

SegmentKind = Literal["title", "heading", "paragraph", "figure", "reference"]
FigureStatus = Literal["pending", "analyzing", "ready", "error"]


class Segment(BaseModel):
    id: str
    index: int
    kind: SegmentKind
    section: str | None = None
    page: int | None = None
    display_text: str
    spoken_text: str
    skip_by_default: bool = False

    figure_id: str | None = None
    figure_label: str | None = None
    caption: str | None = None
    citing_context: str | None = None
    image_url: str | None = None
    image_path: str | None = Field(default=None, exclude=True)
    figure_status: FigureStatus | None = None
    vlm_model: str | None = None
    generation_ms: int | None = None


class DocumentSession(BaseModel):
    doc_id: str
    title: str
    source_type: Literal["pdf", "text"]
    filename: str | None = None
    page_count: int = 1
    figure_count: int = 0
    word_count: int = 0
    estimated_minutes: float = 0.0
    created_at: str
    segments: list[Segment]


class TextIngestRequest(BaseModel):
    title: str | None = None
    text: str
    strip_citations: bool = True


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=2.2)
    doc_id: str | None = None
    segment_id: str | None = None


class ReanalyzeFigureRequest(BaseModel):
    detail_level: Literal["concise", "detailed"] = "detailed"
    custom_question: str | None = None
