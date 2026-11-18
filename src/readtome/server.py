from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import shutil
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .models import DocumentSession, ReanalyzeFigureRequest, Segment, TextIngestRequest, TTSRequest
from .parser import new_doc_id, parse_pdf_document, parse_text_document
from .tts import CURATED_VOICES, LocalTTSEngine
from .vlm import LocalFigureAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("readtome.server")

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
STATIC_DIR = PACKAGE_DIR / "static"
MODELS_DIR = PROJECT_ROOT / "models" / "kokoro"
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
SAMPLE_PDF_PATH = PROJECT_ROOT / "samples" / "sample_paper.pdf"


class SessionStore:
    def __init__(self, sessions_dir: Path, analyzer: LocalFigureAnalyzer) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = analyzer
        self._docs: dict[str, DocumentSession] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict]]] = {}
        self._queue: deque[tuple[str, str]] = deque()
        self._queued_set: set[tuple[str, str]] = set()
        self._inflight_events: dict[tuple[str, str], asyncio.Event] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()
        self._load_existing_sessions()

    def _load_existing_sessions(self) -> None:
        for meta_file in sorted(self.sessions_dir.glob("*/session.json")):
            try:
                raw = json.loads(meta_file.read_text(encoding="utf-8"))
                doc = DocumentSession.model_validate(raw)
                fig_dir = meta_file.parent / "figures"
                for seg in doc.segments:
                    if seg.kind == "figure" and seg.figure_id:
                        candidate = fig_dir / f"{seg.figure_id}.png"
                        if candidate.exists():
                            seg.image_path = str(candidate)
                        if seg.figure_status == "analyzing":
                            seg.figure_status = "pending"
                self._docs[doc.doc_id] = doc
            except Exception as exc:
                logger.warning("Skipping unreadable session %s: %s", meta_file, exc)

    def save(self, doc: DocumentSession) -> None:
        self._docs[doc.doc_id] = doc
        sdir = self.sessions_dir / doc.doc_id
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "session.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    def get(self, doc_id: str) -> DocumentSession | None:
        return self._docs.get(doc_id)

    def list_recent(self, limit: int = 15) -> list[dict]:
        docs = sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)[:limit]
        return [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "source_type": d.source_type,
                "page_count": d.page_count,
                "figure_count": d.figure_count,
                "word_count": d.word_count,
                "estimated_minutes": d.estimated_minutes,
                "created_at": d.created_at,
            }
            for d in docs
        ]

    def subscribe(self, doc_id: str) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue()
        self._subscribers.setdefault(doc_id, set()).add(q)
        return q

    def unsubscribe(self, doc_id: str, q: asyncio.Queue[dict]) -> None:
        subs = self._subscribers.get(doc_id)
        if subs:
            subs.discard(q)

    def _broadcast(self, doc_id: str, payload: dict) -> None:
        for q in list(self._subscribers.get(doc_id, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run_worker(), name="figure-worker")
        for doc in self._docs.values():
            self.enqueue_document_figures(doc.doc_id)

    async def stop_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def enqueue_document_figures(self, doc_id: str) -> None:
        doc = self._docs.get(doc_id)
        if not doc:
            return
        added = False
        for seg in doc.segments:
            if seg.kind == "figure" and seg.figure_status == "pending" and seg.image_path:
                key = (doc_id, seg.id)
                if key not in self._queued_set:
                    self._queue.append(key)
                    self._queued_set.add(key)
                    self._inflight_events.setdefault(key, asyncio.Event())
                    added = True
        if added:
            self._wake_event.set()

    async def prioritize_and_wait(self, doc_id: str, segment_id: str, timeout: float = 90.0) -> Segment:
        doc = self._docs.get(doc_id)
        if not doc:
            raise KeyError(f"Unknown document {doc_id}")
        seg = next((s for s in doc.segments if s.id == segment_id), None)
        if not seg:
            raise KeyError(f"Unknown segment {segment_id}")
        if seg.kind != "figure" or seg.figure_status == "ready":
            return seg

        key = (doc_id, segment_id)
        if key in self._queued_set:
            try:
                self._queue.remove(key)
            except ValueError:
                pass
            self._queue.appendleft(key)
        elif seg.figure_status != "analyzing":
            seg.figure_status = "pending"
            self._queue.appendleft(key)
            self._queued_set.add(key)

        ev = self._inflight_events.setdefault(key, asyncio.Event())
        self._wake_event.set()
        await asyncio.wait_for(ev.wait(), timeout=timeout)
        return seg

    async def _analyze_single_segment(
        self,
        doc_id: str,
        seg: Segment,
        detail_level: str = "detailed",
        custom_question: str | None = None,
    ) -> tuple[str, str, int]:
        doc = self._docs[doc_id]
        if not custom_question:
            seg.figure_status = "analyzing"
            self._broadcast(doc_id, {"type": "segment_update", "segment": seg.model_dump()})

        try:
            display_txt, spoken_txt, elapsed_ms = await asyncio.to_thread(
                self.analyzer.describe_figure,
                seg.image_path or "",
                seg.figure_label or "Figure",
                seg.caption,
                seg.citing_context,
                detail_level,
                custom_question,
            )
            if not custom_question:
                seg.display_text = display_txt
                seg.spoken_text = spoken_txt
                seg.figure_status = "ready"
                seg.vlm_model = self.analyzer.backend_name
                seg.generation_ms = elapsed_ms
                self.save(doc)
                self._broadcast(doc_id, {"type": "segment_update", "segment": seg.model_dump()})
            return display_txt, spoken_txt, elapsed_ms
        except Exception as exc:
            logger.exception("Figure description failed for %s / %s: %s", doc_id, seg.id, exc)
            if not custom_question:
                fallback = seg.caption or f"{seg.figure_label} on page {seg.page}."
                seg.display_text = f"Caption: {fallback}"
                seg.figure_status = "error"
                self.save(doc)
                self._broadcast(doc_id, {"type": "segment_update", "segment": seg.model_dump()})
            raise

    async def _run_worker(self) -> None:
        logger.info("Figure worker started.")
        while True:
            if not self._queue:
                self._wake_event.clear()
                await self._wake_event.wait()
                continue

            doc_id, seg_id = self._queue.popleft()
            self._queued_set.discard((doc_id, seg_id))
            ev = self._inflight_events.get((doc_id, seg_id))

            doc = self._docs.get(doc_id)
            seg = next((s for s in doc.segments if s.id == seg_id), None) if doc else None
            if doc and seg and seg.kind == "figure" and seg.figure_status == "pending" and seg.image_path:
                try:
                    await self._analyze_single_segment(doc_id, seg)
                except Exception:
                    pass

            if ev:
                ev.set()
                self._inflight_events.pop((doc_id, seg_id), None)


tts_engine = LocalTTSEngine(models_dir=MODELS_DIR, cache_dir=AUDIO_CACHE_DIR)
figure_analyzer = LocalFigureAnalyzer()
session_store = SessionStore(sessions_dir=SESSIONS_DIR, analyzer=figure_analyzer)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.create_task(asyncio.to_thread(tts_engine.load))
    session_store.start_worker()
    yield
    await session_store.stop_worker()


app = FastAPI(title="ReadToMe", version="0.1.0", lifespan=lifespan)


@app.get("/api/status")
async def get_status() -> dict:
    voices = await asyncio.to_thread(tts_engine.list_voices) if tts_engine.is_downloaded() else CURATED_VOICES
    return {
        "local_only": True,
        "tts_engine": "Kokoro TTS",
        "tts_ready": tts_engine.is_downloaded(),
        "vlm_engine": figure_analyzer.backend_name,
        "vlm_loaded": figure_analyzer.is_loaded,
        "voices": voices,
        "has_sample_paper": SAMPLE_PDF_PATH.exists(),
    }


@app.get("/api/documents")
async def list_documents() -> dict:
    return {"documents": session_store.list_recent()}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str) -> DocumentSession:
    doc = session_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.post("/api/documents/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    strip_citations: bool = Form(True),
) -> DocumentSession:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    doc_id = new_doc_id()
    sdir = SESSIONS_DIR / doc_id
    sdir.mkdir(parents=True, exist_ok=True)
    pdf_path = sdir / "source.pdf"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    pdf_path.write_bytes(content)

    try:
        doc = await asyncio.to_thread(
            parse_pdf_document,
            pdf_path,
            doc_id,
            sdir,
            file.filename,
            strip_citations,
        )
    except Exception as exc:
        shutil.rmtree(sdir, ignore_errors=True)
        logger.exception("PDF parsing failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc

    session_store.save(doc)
    session_store.enqueue_document_figures(doc.doc_id)
    return doc


@app.post("/api/documents/sample")
async def load_sample_paper() -> DocumentSession:
    if not SAMPLE_PDF_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample paper not found.")

    doc_id = new_doc_id()
    sdir = SESSIONS_DIR / doc_id
    sdir.mkdir(parents=True, exist_ok=True)
    pdf_copy = sdir / "source.pdf"
    shutil.copyfile(SAMPLE_PDF_PATH, pdf_copy)

    doc = await asyncio.to_thread(
        parse_pdf_document,
        pdf_copy,
        doc_id,
        sdir,
        "sample_paper.pdf",
        True,
    )
    session_store.save(doc)
    session_store.enqueue_document_figures(doc.doc_id)
    return doc


@app.post("/api/documents/text")
async def ingest_text(req: TextIngestRequest) -> DocumentSession:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Please enter some text.")

    doc_id = new_doc_id()
    doc = await asyncio.to_thread(
        parse_text_document,
        req.text,
        doc_id,
        req.title,
        req.strip_citations,
    )
    session_store.save(doc)
    return doc


@app.get("/api/documents/{doc_id}/figures/{filename}")
async def serve_figure_image(doc_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    img_path = SESSIONS_DIR / doc_id / "figures" / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Figure image not found")
    return FileResponse(img_path, media_type="image/png")


@app.get("/api/documents/{doc_id}/events")
async def stream_document_events(doc_id: str) -> StreamingResponse:
    doc = session_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    queue = session_store.subscribe(doc_id)

    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            for seg in doc.segments:
                if seg.kind == "figure":
                    yield f"data: {json.dumps({'type': 'segment_update', 'segment': seg.model_dump()})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            session_store.unsubscribe(doc_id, queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/documents/{doc_id}/segments/{segment_id}/prioritize")
async def prioritize_figure_segment(doc_id: str, segment_id: str) -> Segment:
    try:
        return await session_store.prioritize_and_wait(doc_id, segment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Timed out waiting for figure description.") from exc


@app.post("/api/documents/{doc_id}/segments/{segment_id}/ask")
async def ask_or_reanalyze_figure(
    doc_id: str,
    segment_id: str,
    req: ReanalyzeFigureRequest,
) -> dict:
    doc = session_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    seg = next((s for s in doc.segments if s.id == segment_id), None)
    if not seg or seg.kind != "figure":
        raise HTTPException(status_code=400, detail="Segment is not a figure")

    display_txt, spoken_txt, elapsed_ms = await session_store._analyze_single_segment(
        doc_id=doc_id,
        seg=seg,
        detail_level=req.detail_level,
        custom_question=req.custom_question,
    )
    return {
        "segment_id": segment_id,
        "display_text": display_txt,
        "spoken_text": spoken_txt,
        "generation_ms": elapsed_ms,
        "segment": seg.model_dump(),
    }


@app.post("/api/tts")
async def synthesize_speech(req: TTSRequest) -> Response:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    wav_bytes = await asyncio.to_thread(
        tts_engine.synthesize_wav,
        req.text,
        req.voice,
        req.speed,
    )
    return Response(content=wav_bytes, media_type="audio/wav")


STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
