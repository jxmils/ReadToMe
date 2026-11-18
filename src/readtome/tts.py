from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
import threading
import urllib.request

from kokoro_onnx import Kokoro
import numpy as np
import soundfile as sf

logger = logging.getLogger("readtome.tts")

KOKORO_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"

CURATED_VOICES: list[dict[str, str]] = [
    {"id": "af_heart", "name": "Heart", "accent": "US English", "style": "Default US female"},
    {"id": "af_bella", "name": "Bella", "accent": "US English", "style": "US female"},
    {"id": "af_kore", "name": "Kore", "accent": "US English", "style": "US female"},
    {"id": "am_michael", "name": "Michael", "accent": "US English", "style": "US male"},
    {"id": "am_fenrir", "name": "Fenrir", "accent": "US English", "style": "US male"},
    {"id": "bf_emma", "name": "Emma", "accent": "UK English", "style": "UK female"},
    {"id": "bm_george", "name": "George", "accent": "UK English", "style": "UK male"},
    {"id": "bm_fable", "name": "Fable", "accent": "UK English", "style": "UK male"},
]


class LocalTTSEngine:
    def __init__(self, models_dir: Path, cache_dir: Path) -> None:
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._kokoro: Kokoro | None = None

    @property
    def model_path(self) -> Path:
        return self.models_dir / MODEL_FILENAME

    @property
    def voices_path(self) -> Path:
        return self.models_dir / VOICES_FILENAME

    def is_downloaded(self) -> bool:
        return self.model_path.exists() and self.voices_path.exists()

    def ensure_models(self) -> None:
        for fname in (VOICES_FILENAME, MODEL_FILENAME):
            target = self.models_dir / fname
            if target.exists() and target.stat().st_size > 1_000_000:
                continue
            url = f"{KOKORO_RELEASE_BASE}/{fname}"
            tmp_target = target.with_suffix(".part")
            logger.info("Downloading %s ...", fname)
            urllib.request.urlretrieve(url, str(tmp_target))
            tmp_target.replace(target)
            logger.info("Saved %s (%.1f MB)", fname, target.stat().st_size / 1e6)

    def load(self) -> Kokoro:
        if self._kokoro is not None:
            return self._kokoro
        with self._lock:
            if self._kokoro is None:
                self.ensure_models()
                logger.info("Loading Kokoro TTS model from %s", self.models_dir)
                self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        return self._kokoro

    def list_voices(self) -> list[dict[str, str]]:
        kokoro = self.load()
        available = set(kokoro.get_voices())
        result = [v for v in CURATED_VOICES if v["id"] in available]
        curated_ids = {v["id"] for v in result}
        for vid in sorted(available):
            if vid not in curated_ids and vid.startswith(("af_", "am_", "bf_", "bm_")):
                label = vid.split("_", 1)[1].capitalize()
                accent = "UK English" if vid.startswith("b") else "US English"
                result.append({"id": vid, "name": label, "accent": accent, "style": "Extra voice"})
        return result

    def synthesize_wav(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        cleaned = text.strip()
        if not cleaned:
            cleaned = " "

        speed = round(max(0.5, min(2.2, float(speed))), 2)
        cache_key = hashlib.sha256(f"v1|{voice}|{speed:.2f}|{cleaned}".encode("utf-8")).hexdigest()[:24]
        cached_wav = self.cache_dir / f"{cache_key}.wav"
        if cached_wav.exists() and cached_wav.stat().st_size > 128:
            return cached_wav.read_bytes()

        kokoro = self.load()
        available = set(kokoro.get_voices())
        chosen_voice = voice if voice in available else "af_heart"
        lang = "en-gb" if chosen_voice.startswith("b") else "en-us"

        with self._lock:
            samples, sample_rate = kokoro.create(
                cleaned,
                voice=chosen_voice,
                speed=speed,
                lang=lang,
                trim=True,
                sentence_pause=0.26,
                clause_pause=0.10,
            )

        tail_pad = np.zeros(int(sample_rate * 0.06), dtype=np.float32)
        padded = np.concatenate([samples.astype(np.float32), tail_pad])

        buf = io.BytesIO()
        sf.write(buf, padded, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        try:
            cached_wav.write_bytes(wav_bytes)
        except OSError:
            pass
        return wav_bytes
