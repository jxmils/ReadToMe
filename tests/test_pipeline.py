from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pymupdf

from readtome.parser import parse_pdf_document, parse_text_document
from readtome.server import SAMPLE_PDF_PATH, app
from readtome.speech_cleaner import clean_display_text, prepare_spoken_text
from readtome.vlm import DEFAULT_TORCH_MODEL, LocalFigureAnalyzer


def _build_single_column_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    css = "body { font-family: serif; font-size: 10.5pt; line-height: 1.35; }"
    page.insert_htmlbox(
        pymupdf.Rect(72, 54, 540, 92),
        '<div style="font-family:sans-serif;font-size:16pt;font-weight:bold;">Single Column Preprint on Convex Optimization</div>',
    )
    page.insert_htmlbox(pymupdf.Rect(72, 108, 220, 128), '<div style="font-family:sans-serif;font-size:12pt;font-weight:bold;">1 Introduction</div>')
    page.insert_htmlbox(
        pymupdf.Rect(72, 132, 540, 210),
        "<div class='p'>First paragraph in a single column layout explaining gradient convergence rates across smooth objectives.</div>",
        css=css,
    )
    # Narrow centered equation + right-aligned equation tag "(1)" — must NOT split page into 2 columns
    page.insert_htmlbox(pymupdf.Rect(230, 222, 370, 242), "<div>f(x) = L ||x||^2</div>", css=css)
    page.insert_htmlbox(pymupdf.Rect(510, 222, 540, 242), "<div>(1)</div>", css=css)
    page.insert_htmlbox(
        pymupdf.Rect(72, 256, 540, 340),
        "<div class='p'>Second paragraph right below Equation 1 confirming that Figure 1 illustrates the step-size schedule.</div>",
        css=css,
    )
    # Simple vector figure + caption below
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(156, 360, 456, 470))
    shape.finish(color=(0.1, 0.3, 0.7), fill=(0.92, 0.95, 1.0), width=1.2)
    shape.commit()
    page.insert_htmlbox(
        pymupdf.Rect(120, 476, 492, 512),
        '<div style="font-family:sans-serif;font-size:9pt;"><b>Figure 1:</b> Cosine step-size decay over training iterations.</div>',
    )
    page.insert_htmlbox(
        pymupdf.Rect(72, 530, 540, 610),
        "<div class='p'>Final concluding paragraph at the bottom of the single-column page.</div>",
        css=css,
    )
    doc.save(str(path))
    doc.close()


def _build_three_column_cacm_pdf(path: Path) -> None:
    """Build a 3-page PDF reproducing CACM-style 3-column layouts, vertical divider rules,
    asymmetric title page, middle-column pull-quote, and 2-of-3-column figures/tables with top header bars.
    """
    doc = pymupdf.open()
    W, H = 612.0, 792.0
    c1 = (42.0, 208.0)
    c2 = (223.0, 389.0)
    c3 = (404.0, 570.0)
    css = "body { font-family: serif; font-size: 9.2pt; line-height: 1.3; }"

    # --- Page 1: Asymmetric right-two-thirds header + stacked hyphenated title + 2 bottom sub-columns ---
    p1 = doc.new_page(width=W, height=H)
    p1.insert_htmlbox(pymupdf.Rect(400, 24, 570, 40), '<div style="font-size:8pt;">DOI:10.1145/3360307</div>')
    p1.insert_htmlbox(
        pymupdf.Rect(c2[0], 62, c3[1], 125),
        '<div style="font-family:sans-serif;font-size:12pt;font-weight:bold;color:#007396;">Custom supercomputers train deep networks 50x faster on production workloads.</div>',
    )
    p1.insert_htmlbox(
        pymupdf.Rect(c2[0], 132, c3[1], 165),
        '<div style="font-family:sans-serif;font-size:8.5pt;font-weight:bold;">BY NORMAN P. JOUPPI AND DAVID PATTERSON</div>',
    )
    # Stacked multi-line hyphenated title
    title_lines = ["A Domain-", "Specific", "Supercomputer", "for Training", "Deep Networks"]
    y_cursor = 182.0
    for ln in title_lines:
        p1.insert_htmlbox(
            pymupdf.Rect(c2[0], y_cursor, c3[1], y_cursor + 34),
            f'<div style="font-family:sans-serif;font-size:24pt;font-weight:bold;">{ln}</div>',
        )
        y_cursor += 30.0

    p1.insert_htmlbox(
        pymupdf.Rect(c2[0], 360, c3[1], 450),
        "<div>THE RECENT SUCCESS of deep neural networks has inspired domain specific architectures due to the slowing of Moore's Law.17</div>",
        css=css,
    )
    # Vertical hairline divider between sub-column 2 and sub-column 3
    s1 = p1.new_shape()
    s1.draw_line((396.5, 465), (396.5, 725))
    s1.finish(color=(0.4, 0.4, 0.4), width=0.5)
    s1.commit()

    p1.insert_htmlbox(
        pymupdf.Rect(c2[0], 468, c2[1], 680),
        "<div>Alpha subcolumn text comes first on page one before moving across the vertical divider rule.</div>",
        css=css,
    )
    p1.insert_htmlbox(
        pymupdf.Rect(c3[0], 468, c3[1], 680),
        "<div>Beta subcolumn text comes second on page one right after the left subcolumn finishes.</div>",
        css=css,
    )
    p1.insert_htmlbox(pymupdf.Rect(320, 755, 570, 775), '<div style="font-size:7.5pt;">JULY 2020 | COMMUNICATIONS OF THE ACM 67</div>')

    # --- Page 2: 3 columns with vertical divider lines + middle-column pull-quote interrupting Col 1 -> Col 3 sentence ---
    p2 = doc.new_page(width=W, height=H)
    p2.insert_htmlbox(pymupdf.Rect(42, 22, 220, 40), '<div style="font-size:10pt;color:#666;">contributed articles</div>')
    s2 = p2.new_shape()
    s2.draw_line((215.5, 56), (215.5, 735))
    s2.draw_line((396.5, 56), (396.5, 735))
    s2.finish(color=(0.4, 0.4, 0.4), width=0.5)
    s2.commit()

    p2.insert_htmlbox(
        pymupdf.Rect(c1[0], 60, c1[1], 420),
        "<div>Column one on page two explains stochastic gradient descent, where forward propagation is functionally similar to inference,</div>",
        css=css,
    )
    # Oversized pull-quote in middle column
    p2.insert_htmlbox(
        pymupdf.Rect(c2[0], 140, c2[1], 290),
        '<div style="font-family:sans-serif;font-size:15pt;font-weight:bold;color:#006688;">DNN wisdom is that bigger machines lead to bigger breakthroughs.</div>',
    )
    p2.insert_htmlbox(
        pymupdf.Rect(c3[0], 60, c3[1], 420),
        "<div>and if we were building an inference accelerator, we could stop there in column three.</div>",
        css=css,
    )

    # --- Page 3: 2-of-3 column Table 1 + Figure 1 (captions in top header bars!) + 2 lower columns + full-height Col 3 ---
    p3 = doc.new_page(width=W, height=H)
    p3.insert_htmlbox(pymupdf.Rect(42, 22, 220, 40), '<div style="font-size:10pt;color:#666;">contributed articles</div>')
    s3 = p3.new_shape()
    s3.draw_line((396.5, 56), (396.5, 735))
    s3.draw_line((215.5, 430), (215.5, 735))
    s3.finish(color=(0.4, 0.4, 0.4), width=0.5)
    s3.commit()

    # Table 1 spanning Cols 1-2 at top-left (caption bar on top + shaded table body below)
    tshape = p3.new_shape()
    tshape.draw_rect(pymupdf.Rect(c1[0], 60, c2[1], 80))
    tshape.finish(fill=(0.0, 0.45, 0.62), color=None)
    tshape.draw_rect(pymupdf.Rect(c1[0], 80, c2[1], 165))
    tshape.finish(fill=(0.90, 0.95, 0.98), color=(0.2, 0.2, 0.2), width=0.6)
    tshape.commit()
    p3.insert_htmlbox(
        pymupdf.Rect(c1[0] + 4, 63, c2[1] - 4, 79),
        '<div style="font-family:sans-serif;font-size:8.5pt;font-weight:bold;color:#ffffff;">Table 1. Batch sizes across three scaling regions.</div>',
    )
    p3.insert_htmlbox(
        pymupdf.Rect(c1[0] + 6, 90, c2[1] - 6, 155),
        '<div style="font-family:sans-serif;font-size:8pt;">Model  Perfect  Diminishing  Maximum<br/>ResNet-50  8192  65536</div>',
    )

    # Figure 1 spanning Cols 1-2 right below Table 1 (caption bar on top + diagram box below)
    fshape = p3.new_shape()
    fshape.draw_rect(pymupdf.Rect(c1[0], 185, c2[1], 208))
    fshape.finish(fill=(0.0, 0.45, 0.62), color=None)
    fshape.draw_rect(pymupdf.Rect(c1[0], 208, c2[1], 405))
    fshape.finish(fill=(0.91, 0.95, 0.98), color=(0.2, 0.4, 0.6), width=0.8)
    fshape.draw_rect(pymupdf.Rect(c1[0] + 35, 240, c1[0] + 135, 340))
    fshape.finish(fill=(0.82, 0.88, 0.96), color=(0.1, 0.2, 0.5), width=1.2)
    fshape.draw_rect(pymupdf.Rect(c2[0] + 15, 240, c2[0] + 125, 340))
    fshape.finish(fill=(0.85, 0.82, 0.95), color=(0.3, 0.1, 0.5), width=1.2)
    fshape.commit()
    p3.insert_htmlbox(
        pymupdf.Rect(c1[0] + 4, 188, c2[1] - 4, 206),
        '<div style="font-family:sans-serif;font-size:8.5pt;font-weight:bold;color:#ffffff;">Figure 1. Block diagram of a TensorCore node.</div>',
    )
    p3.insert_htmlbox(pymupdf.Rect(c1[0] + 45, 275, c1[0] + 125, 315), '<div style="font-size:8pt;">HBM Memory</div>')
    p3.insert_htmlbox(pymupdf.Rect(c2[0] + 25, 275, c2[0] + 115, 315), '<div style="font-size:8pt;">MXU Array</div>')

    # Lower Columns 1 & 2 underneath Figure 1, plus full-height Column 3 on the right
    p3.insert_htmlbox(
        pymupdf.Rect(c1[0], 435, c1[1], 710),
        "<div>First lower-left column paragraph on page three discusses synchronous all-reduce scaling across torus links.</div>",
        css=css,
    )
    p3.insert_htmlbox(
        pymupdf.Rect(c2[0], 435, c2[1], 710),
        "<div>Second lower-middle column paragraph on page three examines batch size regions summarized in Table 1 for a 2D torus topol-</div>",
        css=css,
    )
    # Shaded "key insights" callout box at the top of Column 3 (must not break "topol-" + "ogy" mid-word)
    kshape = p3.new_shape()
    kshape.draw_rect(pymupdf.Rect(c3[0], 60, c3[1], 195))
    kshape.finish(fill=(0.93, 0.96, 0.98), color=(0.7, 0.8, 0.9), width=0.6)
    kshape.commit()
    p3.insert_htmlbox(
        pymupdf.Rect(c3[0] + 6, 68, c3[1] - 6, 188),
        '<div style="font-family:sans-serif;font-size:8pt;"><b>key insights</b><br/>Co-design of compiler, interconnect, and custom bfloat16 units lets ML workloads scale linearly across hundreds of chips.</div>',
    )
    p3.insert_htmlbox(
        pymupdf.Rect(c3[0], 215, c3[1], 710),
        "<div>ogy. Third right column on page three introduces the six major blocks shown in Figure 1 for the TensorCore chip.</div>",
        css=css,
    )

    doc.save(str(path))
    doc.close()


class ReadToMePipelineTests(unittest.TestCase):
    def test_speech_cleanup_and_superscript_citations(self) -> None:
        raw = clean_display_text(
            "Transformers [1, 4, 12] process 224 × 224 tiles in 180 μs (Vaswani et al., 2017), "
            "achieving a 50x speedup after Moore's Law.17 Also see Google's Pro\ufffdcessing Unit.19,20 "
            "(e.g., α = 0.25 ± 0.02, see Fig. 2)."
        )
        self.assertIn("Processing Unit", raw)
        spoken = prepare_spoken_text(raw, strip_citations=True)
        self.assertNotIn("[1, 4, 12]", spoken)
        self.assertNotIn("17", spoken)
        self.assertNotIn("19,20", spoken)
        self.assertIn("224 by 224", spoken)
        self.assertIn("180 microseconds", spoken)
        self.assertIn("50 times", spoken)
        self.assertIn("alpha", spoken)
        self.assertIn("plus or minus", spoken)
        self.assertIn("Figure 2", spoken)

    def test_single_column_pdf_layout(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="readtome_1col_"))
        try:
            pdf_path = tmp_dir / "single_col.pdf"
            _build_single_column_pdf(pdf_path)
            doc = parse_pdf_document(pdf_path, "doc1col", tmp_dir / "session", "single_col.pdf")
            self.assertEqual(doc.title, "Single Column Preprint on Convex Optimization")
            self.assertEqual(doc.figure_count, 1)

            paragraphs = [s.display_text for s in doc.segments if s.kind == "paragraph"]
            joined = " ".join(paragraphs)
            self.assertLess(joined.index("First paragraph"), joined.index("Second paragraph"))
            self.assertLess(joined.index("Second paragraph"), joined.index("Final concluding paragraph"))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_two_column_sample_pdf_layout(self) -> None:
        self.assertTrue(SAMPLE_PDF_PATH.exists())
        tmp_dir = Path(tempfile.mkdtemp(prefix="readtome_2col_"))
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

    def test_three_column_cacm_style_pdf_layout(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="readtome_3col_"))
        try:
            pdf_path = tmp_dir / "cacm_3col.pdf"
            _build_three_column_cacm_pdf(pdf_path)
            doc = parse_pdf_document(pdf_path, "doc3col", tmp_dir / "session", "cacm_3col.pdf")

            # 1. Stacked hyphenated 5-line title merged cleanly
            self.assertEqual(doc.title, "A Domain-Specific Supercomputer for Training Deep Networks")

            # 2. Both Table 1 and Figure 1 (with top header-bar captions!) extracted without fusing across vertical rules
            visual_labels = [s.figure_label for s in doc.segments if s.kind == "figure"]
            self.assertIn("Table 1", visual_labels)
            self.assertIn("Figure 1", visual_labels)

            all_prose = "\n".join(s.display_text for s in doc.segments if s.kind == "paragraph")

            # 3. Running headers/footers ("contributed articles", "DOI:...") and diagram interior boxes ("MXU Array") excluded
            self.assertNotIn("contributed articles", all_prose)
            self.assertNotIn("MXU Array", all_prose)
            self.assertNotIn("ResNet-50 8192", all_prose)

            # 4. Page 1 asymmetric sub-columns read in order (Alpha -> Beta)
            self.assertLess(all_prose.index("Alpha subcolumn"), all_prose.index("Beta subcolumn"))

            # 5. Page 2 Col 1 -> Col 3 sentence merged seamlessly across the middle-column pull-quote
            self.assertNotIn("bigger breakthroughs", all_prose)
            self.assertIn("similar to inference, and if we were building an inference accelerator", all_prose)

            # 6. Page 3 3-column order + word rejoined smoothly across the "key insights" callout box
            self.assertIn("2D torus topology.", all_prose)
            self.assertLess(
                all_prose.index("First lower-left column"),
                all_prose.index("Second lower-middle column"),
            )
            self.assertLess(
                all_prose.index("Second lower-middle column"),
                all_prose.index("Third right column"),
            )
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
