from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from readtome.parser import parse_pdf_document, parse_text_document
from readtome.server import SAMPLE_PDF_PATH, app
from readtome.speech_cleaner import prepare_spoken_text
from readtome.vlm import DEFAULT_TORCH_MODEL, LocalFigureAnalyzer


class ReadToMePipelineTests(unittest.TestCase):
    def test_speech_cleanup(self) -> None:
        raw = (
            "Transformers [1, 4, 12] process 224 × 224 tiles in 180 μs (Vaswani et al., 2017), "
            "achieving a 3.4× speedup w.r.t. dense baselines (e.g., α = 0.25 ± 0.02, see Fig. 2)."
        )
        spoken = prepare_spoken_text(raw, strip_citations=True)
        self.assertNotIn("[1, 4, 12]", spoken)
        self.assertNotIn("Vaswani", spoken)
        self.assertIn("224 by 224", spoken)
        self.assertIn("180 microseconds", spoken)
        self.assertIn("3.4 times", spoken)
        self.assertIn("with respect to", spoken)
        self.assertIn("for example", spoken)
        self.assertIn("alpha", spoken)
        self.assertIn("plus or minus", spoken)
        self.assertIn("Figure 2", spoken)

    def test_sample_pdf_parsing_and_figures(self) -> None:
        self.assertTrue(SAMPLE_PDF_PATH.exists())
        tmp_dir = Path(tempfile.mkdtemp(prefix="readtome_test_"))
        try:
            doc = parse_pdf_document(SAMPLE_PDF_PATH, "unitdoc", tmp_dir, "sample_paper.pdf")
            self.assertEqual(doc.page_count, 2)
            self.assertEqual(doc.figure_count, 3)
            self.assertIn("SparseRouter-VL", doc.title)
            self.assertIn("Multimodal Reasoning", doc.title)

            fig_segments = [s for s in doc.segments if s.kind == "figure"]
            self.assertEqual([f.figure_label for f in fig_segments], ["Figure 1", "Figure 2", "Figure 3"])
            for fseg in fig_segments:
                self.assertTrue(Path(fseg.image_path or "").exists())
                prev_seg = doc.segments[fseg.index - 1]
                self.assertEqual(prev_seg.kind, "paragraph")
                self.assertIn(fseg.figure_label, prev_seg.spoken_text)

            ref_segments = [s for s in doc.segments if s.kind == "reference"]
            self.assertGreaterEqual(len(ref_segments), 2)
            self.assertTrue(all(r.skip_by_default for r in ref_segments))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_windows_vlm_selection(self) -> None:
        with (
            patch("readtome.vlm.sys.platform", "win32"),
            patch("readtome.vlm._detect_local_ollama_vision_model", return_value=None),
            patch.dict(os.environ, {"READTOME_VLM_BASE_URL": "", "READTOME_VLM_MODEL": ""}, clear=False),
        ):
            win_analyzer = LocalFigureAnalyzer()
            self.assertEqual(win_analyzer.mode, "transformers")
            self.assertEqual(win_analyzer.model_id, DEFAULT_TORCH_MODEL)

        with (
            patch("readtome.vlm.sys.platform", "win32"),
            patch(
                "readtome.vlm._detect_local_ollama_vision_model",
                return_value=("http://127.0.0.1:11434/v1", "qwen3-vl:2b"),
            ),
            patch.dict(os.environ, {"READTOME_VLM_BASE_URL": "", "READTOME_VLM_MODEL": ""}, clear=False),
        ):
            ollama_analyzer = LocalFigureAnalyzer()
            self.assertEqual(ollama_analyzer.mode, "openai_compat")
            self.assertEqual(ollama_analyzer.model_id, "qwen3-vl:2b")

    def test_text_ingest_and_tts_endpoint(self) -> None:
        doc = parse_text_document(
            "# My Paper\n\n## Abstract\nOur method achieves 2.5× throughput [2, 5].",
            "txtdoc",
        )
        self.assertEqual(doc.title, "My Paper")
        self.assertGreaterEqual(len(doc.segments), 3)

        with TestClient(app) as client:
            st = client.get("/api/status")
            self.assertEqual(st.status_code, 200)
            self.assertTrue(st.json()["local_only"])

            tts_resp = client.post(
                "/api/tts",
                json={"text": "Testing local speech output.", "voice": "af_heart", "speed": 1.0},
            )
            self.assertEqual(tts_resp.status_code, 200)
            self.assertEqual(tts_resp.headers["content-type"], "audio/wav")
            self.assertTrue(tts_resp.content.startswith(b"RIFF"))
            self.assertGreater(len(tts_resp.content), 10_000)


if __name__ == "__main__":
    unittest.main()
