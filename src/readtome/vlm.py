from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import threading
import time

import httpx
from PIL import Image

from .speech_cleaner import prepare_spoken_text

logger = logging.getLogger("readtome.vlm")

DEFAULT_MLX_MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
DEFAULT_TORCH_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

_OLLAMA_VISION_HINTS = ("vl", "llava", "vision", "minicpm-v", "moondream", "gemma3", "pixtral", "bakllava")


def _has_mlx_vlm() -> bool:
    return (
        sys.platform == "darwin"
        and platform.machine().lower() in ("arm64", "aarch64")
        and importlib.util.find_spec("mlx_vlm") is not None
    )


def _detect_local_ollama_vision_model(timeout: float = 0.45) -> tuple[str, str] | None:
    if os.environ.get("READTOME_VLM_BACKEND", "").strip().lower() in ("transformers", "torch", "mlx"):
        return None
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{ollama_host}/api/tags")
            if resp.status_code != 200:
                return None
            models = resp.json().get("models", [])
            for m in models:
                name = str(m.get("name", ""))
                if any(hint in name.lower() for hint in _OLLAMA_VISION_HINTS):
                    return f"{ollama_host}/v1", name
    except Exception:
        return None
    return None


def _strip_markdown_for_speech(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    lines = []
    for raw_line in text.splitlines():
        ln = raw_line.strip()
        if not ln:
            continue
        ln = re.sub(r"^(?:[-•*]|\d+[.)])\s+", "", ln)
        if ln and ln[-1] not in ".!?":
            ln += "."
        lines.append(ln)
    joined = " ".join(lines)
    return re.sub(r"\s{2,}", " ", joined).strip()


def _build_figure_prompt(
    figure_label: str,
    caption: str | None,
    citing_context: str | None,
    detail_level: str = "detailed",
    custom_question: str | None = None,
) -> str:
    parts = [
        "You are reading a research paper out loud to someone who cannot see the page.",
    ]
    if figure_label:
        parts.append(f"Figure: {figure_label}.")
    if caption:
        parts.append(f"Caption from the paper: {caption}")
    if citing_context:
        parts.append(f"Surrounding text from the paper: {citing_context}")

    is_table = figure_label.lower().startswith("table")
    if custom_question and custom_question.strip():
        parts.append(
            f"Answer this question based on what is shown in the image: "
            f"{custom_question.strip()}\n"
            f"Use 2 to 4 plain spoken sentences. Do not use markdown, bullets, or asterisks."
        )
    elif is_table:
        parts.append(
            f"Start naturally (like 'Looking at {figure_label}, ...'). In 3 or 4 plain spoken sentences, "
            "say what this table compares, call out the most notable numbers or differences across rows and columns, "
            "and state the main takeaway without reading every cell one by one. "
            "No markdown or bullet points."
        )
    elif detail_level == "concise":
        parts.append(
            f"Start naturally (like 'Looking at {figure_label}, ...'). In 2 or 3 plain spoken sentences, "
            "say what kind of chart or diagram this is, what is being compared, and the main takeaway. "
            "No markdown or bullet points, and don't just repeat the caption."
        )
    else:
        parts.append(
            f"Start naturally (like 'Looking at {figure_label}, ...'). In 4 or 5 plain spoken sentences: "
            "say what kind of chart or diagram this is and what the axes or parts represent, "
            "walk through the main trends or numbers visible in the image, "
            "and state the main point it makes for the paper. "
            "Write for spoken audio only: plain sentences, no markdown, no bullet points, and don't just read the caption word for word."
        )
    return "\n\n".join(parts)


class LocalFigureAnalyzer:
    def __init__(self, model_id: str | None = None) -> None:
        env_model = os.environ.get("READTOME_VLM_MODEL", "").strip()
        self.external_base_url = os.environ.get("READTOME_VLM_BASE_URL", "").strip()
        self._lock = threading.Lock()

        if self.external_base_url:
            self._mode = "openai_compat"
            self.model_id = model_id or env_model or DEFAULT_TORCH_MODEL
        elif _has_mlx_vlm() and os.environ.get("READTOME_VLM_BACKEND", "").strip().lower() not in ("transformers", "torch"):
            self._mode = "mlx"
            self.model_id = model_id or env_model or DEFAULT_MLX_MODEL
        else:
            ollama_detected = _detect_local_ollama_vision_model() if not (model_id or env_model) else None
            if ollama_detected is not None:
                self._mode = "openai_compat"
                self.external_base_url, self.model_id = ollama_detected
                logger.info("Using local Ollama vision model %s at %s", self.model_id, self.external_base_url)
            else:
                self._mode = "transformers"
                self.model_id = model_id or env_model or DEFAULT_TORCH_MODEL

        self._mlx_model = None
        self._mlx_processor = None
        self._mlx_config = None

        self._torch_model = None
        self._torch_processor = None
        self._torch_device: str | None = None
        self._torch_device_label: str | None = None

        self._loaded = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def backend_name(self) -> str:
        short_model = self.model_id.split("/")[-1]
        if self._mode == "openai_compat":
            host_label = "Ollama" if "11434" in self.external_base_url else "Local API"
            return f"{host_label} ({short_model})"
        if self._mode == "mlx":
            return f"MLX ({short_model})"
        if self._torch_device_label:
            return f"{self._torch_device_label} ({short_model})"
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu = torch.cuda.get_device_name(0)
                    return f"CUDA ({gpu}, {short_model})"
            except Exception:
                pass
        return f"PyTorch ({short_model})"

    @property
    def is_loaded(self) -> bool:
        return self._loaded or self._mode == "openai_compat"

    def _is_cpu_only(self) -> bool:
        if self._mode != "transformers":
            return False
        if self._torch_device is not None:
            return self._torch_device == "cpu"
        try:
            import torch

            return not torch.cuda.is_available() and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except Exception:
            return True

    def _ensure_mlx_loaded(self) -> None:
        if self._mlx_model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        logger.info("Loading %s with mlx-vlm...", self.model_id)
        t0 = time.perf_counter()
        self._mlx_model, self._mlx_processor = load(self.model_id)
        self._mlx_config = load_config(self.model_id)
        self._loaded = True
        logger.info("Loaded %s in %.2fs", self.model_id, time.perf_counter() - t0)

    def _ensure_torch_loaded(self) -> None:
        if self._torch_model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self._torch_device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
            self._torch_device_label = "MPS"
        else:
            device = "cpu"
            dtype = torch.bfloat16
            self._torch_device_label = "CPU"

        self._torch_device = device
        logger.info("Loading %s on %s...", self.model_id, self._torch_device_label)
        t0 = time.perf_counter()
        self._torch_processor = AutoProcessor.from_pretrained(self.model_id)
        model_kwargs: dict = {"dtype": dtype, "low_cpu_mem_usage": True}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
        if device != "cuda":
            model = model.to(device)
        model.eval()
        self._torch_model = model
        self._loaded = True
        logger.info("Loaded %s in %.2fs", self.model_id, time.perf_counter() - t0)

    def _load_resized_pil(self, image_path: Path, max_side: int) -> Image.Image:
        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size
            if max(w, h) <= max_side:
                return im_rgb.copy()
            scale = min(max_side / max(w, h), 1.0)
            new_w = max(int(w * scale), 64)
            new_h = max(int(h * scale), 64)
            return im_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _prepare_resized_file(self, image_path: Path, max_side: int = 1024) -> tuple[Path, bool]:
        with Image.open(image_path) as im:
            w, h = im.size
            if max(w, h) <= max_side and image_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return image_path, False
        resized = self._load_resized_pil(image_path, max_side=max_side)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        resized.save(tmp_path, format="PNG")
        return tmp_path, True

    def _generate_via_openai_compat(self, pil_img: Image.Image, prompt: str, max_tokens: int) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        url = f"{self.external_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _generate_via_torch(self, pil_img: Image.Image, prompt: str, max_tokens: int) -> str:
        import torch

        self._ensure_torch_loaded()
        assert self._torch_processor is not None and self._torch_model is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._torch_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._torch_model.device)

        with torch.inference_mode():
            generated_ids = self._torch_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded = self._torch_processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    def describe_figure(
        self,
        image_path: str | Path,
        figure_label: str = "Figure",
        caption: str | None = None,
        citing_context: str | None = None,
        detail_level: str = "detailed",
        custom_question: str | None = None,
    ) -> tuple[str, str, int]:
        t0 = time.perf_counter()
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Figure image not found: {img_p}")

        prompt_text = _build_figure_prompt(
            figure_label=figure_label,
            caption=caption,
            citing_context=citing_context,
            detail_level=detail_level,
            custom_question=custom_question,
        )
        max_tokens = 240 if detail_level == "detailed" else 150
        max_side = 768 if self._is_cpu_only() else 1024

        with self._lock:
            if self._mode == "openai_compat":
                pil_img = self._load_resized_pil(img_p, max_side=max_side)
                raw_out = self._generate_via_openai_compat(pil_img, prompt_text, max_tokens)
            elif self._mode == "mlx":
                vlm_img_path, is_temp = self._prepare_resized_file(img_p, max_side=max_side)
                try:
                    self._ensure_mlx_loaded()
                    from mlx_vlm import generate
                    from mlx_vlm.prompt_utils import apply_chat_template

                    formatted = apply_chat_template(
                        self._mlx_processor,
                        self._mlx_config,
                        prompt_text,
                        num_images=1,
                    )
                    res = generate(
                        self._mlx_model,
                        self._mlx_processor,
                        formatted,
                        image=[str(vlm_img_path)],
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                    raw_out = res.text
                finally:
                    if is_temp:
                        try:
                            vlm_img_path.unlink(missing_ok=True)
                        except OSError:
                            pass
            else:
                pil_img = self._load_resized_pil(img_p, max_side=max_side)
                raw_out = self._generate_via_torch(pil_img, prompt_text, max_tokens)

        clean_explanation = _strip_markdown_for_speech(raw_out)
        if not clean_explanation:
            clean_explanation = caption or f"{figure_label} diagram."

        if not custom_question and figure_label.lower() not in clean_explanation[:45].lower():
            clean_explanation = f"Looking at {figure_label}: {clean_explanation}"

        spoken = prepare_spoken_text(clean_explanation, strip_citations=True)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return clean_explanation, spoken, elapsed_ms
