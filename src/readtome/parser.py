from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

import pymupdf

from .models import DocumentSession, Segment
from .speech_cleaner import clean_display_text, prepare_spoken_text

# Matches actual figure captions like "Figure 1: ...", "Fig. 2. ...", "FIGURE 3 — ..."
_CAPTION_START_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|FIGURE|FIG\.?)\s*(\d+[A-Za-z]?)\s*(?:[:.|—–\-]|\s+(?=[A-Z(]))",
    re.DOTALL,
)
# Body sentences starting with "Figure 1 shows..." (no punctuation separator + lowercase verb)
_FALSE_CAPTION_VERB_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?)\s*\d+[A-Za-z]?\s+"
    r"(?:shows?|illustrates?|demonstrates?|presents?|compares?|depicts?|plots?|"
    r"summarizes?|is|are|was|were|reveals?|highlights?|confirms?|provides?|gives?|reports?)\b",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:Table|Tab\.?|TABLE)\s*(?:\d+|[IVX]+)\b")

_SECTION_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,2}|[IVX]+\.)\s+[A-Z][^\n]{1,75}|"
    r"(?:Abstract|Introduction|Background|Related\s+Work|Preliminaries|Method(?:s|ology)?|"
    r"Approach|Architecture|Experiments?|Experimental\s+Setup|Results?(?:\s+and\s+Discussion)?|"
    r"Discussion|Ablation\s+Stud(?:y|ies)|Limitations?|Conclusion(?:s)?|Acknowledg(?:e)?ments?|"
    r"Appendix|References|Bibliography))$",
    re.IGNORECASE,
)

_REFERENCES_HEADING_RE = re.compile(r"^(?:\d+\.?\s*)?(?:References|Bibliography|Works\s+Cited)$", re.IGNORECASE)
_ARXIV_STAMP_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}", re.IGNORECASE)


@dataclass
class _TextBlock:
    page_num: int  # 1-based
    bbox: pymupdf.Rect
    raw_text: str
    clean_text: str
    max_font_size: float
    avg_font_size: float
    is_bold: bool
    first_span_bold: bool


@dataclass
class _ExtractedFigure:
    figure_id: str
    figure_num: str | None
    figure_label: str
    page_num: int
    crop_rect: pymupdf.Rect
    exclusion_rects: list[pymupdf.Rect] = field(default_factory=list)
    caption: str = ""
    image_filename: str = ""
    image_path: str = ""
    anchor_y: float = 0.0


def _rect_overlap_ratio(inner: pymupdf.Rect, outer: pymupdf.Rect) -> float:
    inter = inner & outer
    if inter.is_empty or inner.is_empty:
        return 0.0
    inner_area = max(inner.width * inner.height, 1e-6)
    return (inter.width * inter.height) / inner_area


def _horiz_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    left = max(a.x0, b.x0)
    right = min(a.x1, b.x1)
    if right <= left:
        return 0.0
    min_w = max(min(a.width, b.width), 1e-6)
    return (right - left) / min_w


def _separated_by_barrier(a: pymupdf.Rect, b: pymupdf.Rect, barriers: list[pymupdf.Rect]) -> bool:
    top_y = min(a.y1, b.y1)
    bot_y = max(a.y0, b.y0)
    if bot_y <= top_y:
        return False
    union_x0 = min(a.x0, b.x0)
    union_x1 = max(a.x1, b.x1)
    for bar in barriers:
        if bar.y0 >= top_y - 4 and bar.y1 <= bot_y + 4:
            if min(bar.x1, union_x1) - max(bar.x0, union_x0) > 20:
                return True
    return False


def _merge_visual_boxes(boxes: list[pymupdf.Rect], barriers: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Merge nearby sub-panels/drawings belonging to the same figure unless a caption separates them."""
    if not boxes:
        return []
    merged = [pymupdf.Rect(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        next_list: list[pymupdf.Rect] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = pymupdf.Rect(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                if _separated_by_barrier(cur, other, barriers):
                    continue
                # Check intersection or close proximity (side-by-side panels or stacked subplots)
                expanded = pymupdf.Rect(cur.x0 - 26, cur.y0 - 20, cur.x1 + 26, cur.y1 + 20)
                if not (expanded & other).is_empty:
                    cur |= other
                    used[j] = True
                    changed = True
            next_list.append(cur)
        merged = next_list
    return merged


def _extract_page_text_blocks(page: pymupdf.Page, page_num: int) -> list[_TextBlock]:
    page_rect = page.rect
    w, h = page_rect.width, page_rect.height
    data = page.get_text("dict")
    blocks: list[_TextBlock] = []

    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        bbox = pymupdf.Rect(b["bbox"])
        if bbox.is_empty or bbox.width < 4 or bbox.height < 4:
            continue

        lines_text: list[str] = []
        font_sizes: list[float] = []
        char_weights: list[int] = []
        bold_chars = 0
        total_chars = 0
        first_span_bold = False
        saw_first_span = False

        for line in b.get("lines", []):
            span_strs: list[str] = []
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if not txt.strip():
                    if txt:
                        span_strs.append(txt)
                    continue
                span_strs.append(txt)
                sz = float(span.get("size", 10.0))
                font_name = span.get("font", "").lower()
                flags = int(span.get("flags", 0))
                is_span_bold = bool(flags & 16) or ("bold" in font_name) or ("medi" in font_name) or ("demi" in font_name)
                if not saw_first_span:
                    first_span_bold = is_span_bold
                    saw_first_span = True
                n_chars = len(txt.strip())
                font_sizes.append(sz)
                char_weights.append(n_chars)
                total_chars += n_chars
                if is_span_bold:
                    bold_chars += n_chars
            if span_strs:
                lines_text.append("".join(span_strs))

        raw_text = "\n".join(lines_text).strip()
        clean_text = clean_display_text(raw_text)
        if not clean_text:
            continue

        # Filter vertical margin stamps (e.g., arXiv sidebars)
        if (bbox.x1 < 0.08 * w or bbox.x0 > 0.92 * w) and bbox.height > 2.2 * bbox.width:
            continue
        if _ARXIV_STAMP_RE.search(clean_text) and len(clean_text) < 120:
            continue

        # Filter running headers, footers, and standalone page numbers
        if (bbox.y1 < 0.055 * h or bbox.y0 > 0.942 * h) and len(clean_text) < 115:
            if not _CAPTION_START_RE.match(clean_text):
                continue
        if re.fullmatch(r"\d{1,4}", clean_text) and (bbox.y0 > 0.89 * h or bbox.y1 < 0.08 * h):
            continue

        max_sz = max(font_sizes) if font_sizes else 10.0
        avg_sz = (
            sum(s * c for s, c in zip(font_sizes, char_weights)) / max(sum(char_weights), 1)
            if font_sizes
            else 10.0
        )
        is_bold = (bold_chars / max(total_chars, 1)) > 0.55

        blocks.append(
            _TextBlock(
                page_num=page_num,
                bbox=bbox,
                raw_text=raw_text,
                clean_text=clean_text,
                max_font_size=max_sz,
                avg_font_size=avg_sz,
                is_bold=is_bold,
                first_span_bold=first_span_bold,
            )
        )
    return blocks


def _is_figure_caption(tb: _TextBlock, body_font_size: float) -> str | None:
    """Return figure number string (e.g. '1', '2a') if block is a figure caption, else None."""
    m = _CAPTION_START_RE.match(tb.clean_text)
    if not m:
        return None
    fig_num = m.group(1)
    # Check if it's actually a prose paragraph starting with "Figure 1 shows..."
    has_explicit_sep = bool(
        re.match(r"^\s*(?:Figure|Fig\.?)\s*\d+[A-Za-z]?\s*[:.|—–\-]", tb.clean_text, re.IGNORECASE)
    )
    if not has_explicit_sep and not tb.first_span_bold:
        if _FALSE_CAPTION_VERB_RE.match(tb.clean_text):
            return None
        if tb.avg_font_size > body_font_size + 0.5:
            return None
    return fig_num


def _collect_visual_regions(page: pymupdf.Page, caption_rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    page_rect = page.rect
    w, h = page_rect.width, page_rect.height
    raw_boxes: list[pymupdf.Rect] = []

    # 1. Embedded raster images
    seen_xrefs: set[int] = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        for r in page.get_image_rects(xref):
            r = r & page_rect
            if r.is_empty or r.width < 42 or r.height < 42:
                continue
            if r.width > 0.96 * w and r.height > 0.96 * h:
                continue
            raw_boxes.append(r)

    # 2. Vector graphics clusters (plots, charts, architecture diagrams)
    try:
        drawing_clusters = page.cluster_drawings(x_tolerance=12, y_tolerance=12)
    except Exception:
        drawing_clusters = []

    for dr in drawing_clusters:
        r = pymupdf.Rect(dr) & page_rect
        if r.is_empty:
            continue
        # Ignore thin rules/lines (underlines, table borders, header bars)
        if r.width < 48 or r.height < 38:
            continue
        if r.width > 0.96 * w and r.height > 0.96 * h:
            continue
        raw_boxes.append(r)

    return _merge_visual_boxes(raw_boxes, caption_rects)


def _extract_figures_on_page(
    page: pymupdf.Page,
    page_num: int,
    text_blocks: list[_TextBlock],
    body_font_size: float,
    figures_dir: Path,
    start_fig_counter: int,
) -> tuple[list[_ExtractedFigure], set[int]]:
    """Detect & crop figures on a page, returning (figures, excluded_text_block_indices)."""
    page_rect = page.rect
    w, h = page_rect.width, page_rect.height

    caption_entries: list[tuple[int, str, _TextBlock]] = []
    table_caption_rects: list[pymupdf.Rect] = []
    for idx, tb in enumerate(text_blocks):
        fig_num = _is_figure_caption(tb, body_font_size)
        if fig_num is not None:
            caption_entries.append((idx, fig_num, tb))
        elif _TABLE_CAPTION_RE.match(tb.clean_text):
            table_caption_rects.append(tb.bbox)

    barriers = [tb.bbox for _, _, tb in caption_entries] + table_caption_rects
    visual_boxes = _collect_visual_regions(page, barriers)
    used_visuals: set[int] = set()
    excluded_block_idxs: set[int] = set()
    extracted: list[_ExtractedFigure] = []
    fig_counter = start_fig_counter

    for cap_idx, fig_num, cap_tb in caption_entries:
        cap_rect = cap_tb.bbox
        excluded_block_idxs.add(cap_idx)

        # Find visual boxes above caption first (standard academic layout)
        above_candidates: list[int] = []
        for v_idx, vr in enumerate(visual_boxes):
            if v_idx in used_visuals:
                continue
            # Visual is above caption and not separated by another caption/long prose block
            if vr.y1 <= cap_rect.y0 + 24 and (cap_rect.y0 - vr.y1) <= 165:
                if _horiz_overlap(vr, cap_rect) >= 0.25 or vr.width > 0.65 * w or cap_rect.width > 0.65 * w:
                    if not _separated_by_barrier(vr, cap_rect, [b for b in barriers if b != cap_rect]):
                        above_candidates.append(v_idx)

        chosen_vis_idxs = above_candidates
        if not chosen_vis_idxs:
            # Check visual boxes immediately below caption (captions-on-top style)
            for v_idx, vr in enumerate(visual_boxes):
                if v_idx in used_visuals:
                    continue
                if vr.y0 >= cap_rect.y1 - 18 and (vr.y0 - cap_rect.y1) <= 110:
                    if _horiz_overlap(vr, cap_rect) >= 0.25 or vr.width > 0.65 * w:
                        if not _separated_by_barrier(cap_rect, vr, [b for b in barriers if b != cap_rect]):
                            chosen_vis_idxs.append(v_idx)

        if not chosen_vis_idxs:
            # Fallback: if a diagram consists purely of arranged text boxes + thin arrows missed by cluster_drawings,
            # look for a non-prose zone directly above the caption.
            fallback_top = max(page_rect.y0 + 28, cap_rect.y0 - 210)
            for idx_other, other_tb in enumerate(text_blocks):
                if idx_other == cap_idx:
                    continue
                if other_tb.bbox.y1 <= cap_rect.y0 and len(other_tb.clean_text) > 180:
                    if _horiz_overlap(other_tb.bbox, cap_rect) > 0.25:
                        fallback_top = max(fallback_top, other_tb.bbox.y1 + 8)
            if cap_rect.y0 - fallback_top < 45:
                continue
            vis_union = pymupdf.Rect(cap_rect.x0, fallback_top, cap_rect.x1, cap_rect.y0 - 2)
        else:
            vis_union = pymupdf.Rect(visual_boxes[chosen_vis_idxs[0]])
            for v_idx in chosen_vis_idxs[1:]:
                vis_union |= visual_boxes[v_idx]
            for v_idx in chosen_vis_idxs:
                used_visuals.add(v_idx)

        # Expand slightly to include outer plot axis labels, titles, subfigure (a)/(b) markers, and tick numbers
        expanded_vis = pymupdf.Rect(
            max(page_rect.x0 + 8, vis_union.x0 - 30),
            max(page_rect.y0 + 8, vis_union.y0 - 22),
            min(page_rect.x1 - 8, vis_union.x1 + 22),
            min(page_rect.y1 - 8, vis_union.y1 + 24),
        )
        # Do not intrude into the caption itself or long body paragraphs
        if expanded_vis.y0 < cap_rect.y0 and vis_union.y1 <= cap_rect.y0 + 16:
            expanded_vis.y1 = min(expanded_vis.y1, cap_rect.y0 - 2)

        for idx_other, other_tb in enumerate(text_blocks):
            if idx_other in excluded_block_idxs:
                continue
            inter = expanded_vis & other_tb.bbox
            if inter.is_empty:
                continue
            # Short labels (axis titles, ticks, legend entries) belong inside the figure crop
            if len(other_tb.clean_text) <= 115 and not _SECTION_HEADING_RE.match(other_tb.clean_text):
                vis_union |= other_tb.bbox
                excluded_block_idxs.add(idx_other)
            elif _rect_overlap_ratio(other_tb.bbox, vis_union) > 0.55:
                excluded_block_idxs.add(idx_other)

        # Add crisp white padding around the final figure crop without slicing into the caption
        crop_y0 = max(page_rect.y0 + 4, vis_union.y0 - 10)
        crop_y1 = min(page_rect.y1 - 4, vis_union.y1 + 10)
        if vis_union.y1 <= cap_rect.y0 + 16 and crop_y0 < cap_rect.y0:
            crop_y1 = min(crop_y1, cap_rect.y0 - 2)
        elif vis_union.y0 >= cap_rect.y1 - 16 and crop_y1 > cap_rect.y1:
            crop_y0 = max(crop_y0, cap_rect.y1 + 2)

        crop_rect = pymupdf.Rect(
            max(page_rect.x0 + 4, vis_union.x0 - 10),
            crop_y0,
            min(page_rect.x1 - 4, vis_union.x1 + 10),
            crop_y1,
        )
        if crop_rect.width < 45 or crop_rect.height < 40:
            continue

        fig_counter += 1
        fig_id = f"fig_{fig_counter}"
        img_filename = f"{fig_id}.png"
        img_path = figures_dir / img_filename
        pix = page.get_pixmap(clip=crop_rect, dpi=220, alpha=False)
        pix.save(str(img_path))

        extracted.append(
            _ExtractedFigure(
                figure_id=fig_id,
                figure_num=fig_num,
                figure_label=f"Figure {fig_num}",
                page_num=page_num,
                crop_rect=crop_rect,
                exclusion_rects=[crop_rect, cap_rect],
                caption=cap_tb.clean_text,
                image_filename=img_filename,
                image_path=str(img_path),
                anchor_y=min(crop_rect.y0, cap_rect.y0),
            )
        )

    # Also capture standalone large visual figures without a formal "Figure N:" caption (slides, reports, diagrams)
    for v_idx, vr in enumerate(visual_boxes):
        if v_idx in used_visuals:
            continue
        if vr.width < 125 or vr.height < 95 or (vr.width * vr.height) < 18000:
            continue
        # Skip if right next to a Table caption (likely a boxed table)
        near_table = any(
            abs(vr.y0 - tr.y1) < 35 or abs(tr.y0 - vr.y1) < 35 for tr in table_caption_rects
        )
        if near_table:
            continue

        crop_rect = pymupdf.Rect(
            max(page_rect.x0 + 4, vr.x0 - 10),
            max(page_rect.y0 + 4, vr.y0 - 10),
            min(page_rect.x1 - 4, vr.x1 + 10),
            min(page_rect.y1 - 4, vr.y1 + 10),
        )
        for idx_other, other_tb in enumerate(text_blocks):
            if idx_other in excluded_block_idxs:
                continue
            if _rect_overlap_ratio(other_tb.bbox, crop_rect) > 0.55 and len(other_tb.clean_text) <= 130:
                excluded_block_idxs.add(idx_other)

        fig_counter += 1
        fig_id = f"fig_{fig_counter}"
        img_filename = f"{fig_id}.png"
        img_path = figures_dir / img_filename
        pix = page.get_pixmap(clip=crop_rect, dpi=220, alpha=False)
        pix.save(str(img_path))

        extracted.append(
            _ExtractedFigure(
                figure_id=fig_id,
                figure_num=None,
                figure_label=f"Figure (Page {page_num})",
                page_num=page_num,
                crop_rect=crop_rect,
                exclusion_rects=[crop_rect],
                caption="",
                image_filename=img_filename,
                image_path=str(img_path),
                anchor_y=crop_rect.y0,
            )
        )

    extracted.sort(key=lambda f: f.anchor_y)
    return extracted, excluded_block_idxs


def _order_blocks_academic(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    """Order blocks cleanly across full-width banners and two-column layouts."""
    if len(blocks) <= 1:
        return blocks

    mid_x = page_width * 0.5

    def is_spanning(b: _TextBlock) -> bool:
        return b.bbox.x0 < page_width * 0.46 and b.bbox.x1 > page_width * 0.54

    rough = sorted(blocks, key=lambda b: (round(b.bbox.y0, 1), round(b.bbox.x0, 1)))
    ordered: list[_TextBlock] = []
    col_buffer: list[_TextBlock] = []

    def flush_columns() -> None:
        if not col_buffer:
            return
        left = [b for b in col_buffer if (b.bbox.x0 + b.bbox.x1) * 0.5 < mid_x]
        right = [b for b in col_buffer if (b.bbox.x0 + b.bbox.x1) * 0.5 >= mid_x]
        left.sort(key=lambda b: (round(b.bbox.y0, 1), round(b.bbox.x0, 1)))
        right.sort(key=lambda b: (round(b.bbox.y0, 1), round(b.bbox.x0, 1)))
        ordered.extend(left)
        ordered.extend(right)
        col_buffer.clear()

    for b in rough:
        if is_spanning(b):
            flush_columns()
            ordered.append(b)
        else:
            col_buffer.append(b)
    flush_columns()
    return ordered


def _estimate_body_font_size(all_page_blocks: list[list[_TextBlock]]) -> float:
    weighted_sizes: list[tuple[float, int]] = []
    for pblocks in all_page_blocks:
        for b in pblocks:
            if len(b.clean_text) >= 60:
                weighted_sizes.append((round(b.avg_font_size, 1), len(b.clean_text)))
    if not weighted_sizes:
        return 10.0
    buckets: dict[float, int] = {}
    for sz, weight in weighted_sizes:
        buckets[sz] = buckets.get(sz, 0) + weight
    return max(buckets.items(), key=lambda kv: kv[1])[0]


def parse_pdf_document(
    pdf_path: Path,
    doc_id: str,
    session_dir: Path,
    filename: str | None = None,
    strip_citations: bool = True,
) -> DocumentSession:
    """Parse an academic or general PDF into ordered narrative segments + high-res extracted figures."""
    figures_dir = session_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)

    raw_page_blocks: list[list[_TextBlock]] = []
    for p_idx in range(page_count):
        raw_page_blocks.append(_extract_page_text_blocks(doc[p_idx], p_idx + 1))

    body_font_size = _estimate_body_font_size(raw_page_blocks)

    all_figures: list[_ExtractedFigure] = []
    ordered_text_blocks: list[_TextBlock] = []
    fig_counter = 0

    for p_idx in range(page_count):
        page = doc[p_idx]
        p_blocks = raw_page_blocks[p_idx]
        page_figs, excluded_idxs = _extract_figures_on_page(
            page=page,
            page_num=p_idx + 1,
            text_blocks=p_blocks,
            body_font_size=body_font_size,
            figures_dir=figures_dir,
            start_fig_counter=fig_counter,
        )
        fig_counter += len(page_figs)
        all_figures.extend(page_figs)

        kept_blocks = [b for i, b in enumerate(p_blocks) if i not in excluded_idxs]
        ordered_text_blocks.extend(_order_blocks_academic(kept_blocks, page.rect.width))

    doc.close()

    # Identify Document Title on page 1 (and merge multi-line title blocks sharing the same large font)
    doc_title = (filename or "Untitled Paper").rsplit(".", 1)[0]
    title_block_indices: set[int] = set()
    page1_candidates = [
        (i, b)
        for i, b in enumerate(ordered_text_blocks[:8])
        if b.page_num == 1
        and b.bbox.y0 < 320
        and 4 <= len(b.clean_text) <= 260
        and not _SECTION_HEADING_RE.match(b.clean_text)
    ]
    if page1_candidates:
        best_i, best_b = max(page1_candidates, key=lambda item: item[1].max_font_size)
        if best_b.max_font_size >= body_font_size + 1.5:
            title_parts = [best_b.clean_text]
            title_block_indices.add(best_i)
            prev_b = best_b
            for nxt_i in range(best_i + 1, min(best_i + 4, len(ordered_text_blocks))):
                nxt_b = ordered_text_blocks[nxt_i]
                if (
                    nxt_b.page_num == 1
                    and abs(nxt_b.max_font_size - best_b.max_font_size) <= 0.6
                    and -8 <= (nxt_b.bbox.y0 - prev_b.bbox.y1) <= 18
                    and not _SECTION_HEADING_RE.match(nxt_b.clean_text)
                ):
                    title_parts.append(nxt_b.clean_text)
                    title_block_indices.add(nxt_i)
                    prev_b = nxt_b
                else:
                    break
            doc_title = " ".join(title_parts)

    # Classify blocks into title / headings / paragraphs / references and merge split paragraphs
    draft_items: list[dict] = []
    current_section = "Overview"
    in_references = False
    emitted_title = False

    for idx, tb in enumerate(ordered_text_blocks):
        txt = tb.clean_text
        if not txt:
            continue

        if idx in title_block_indices:
            if not emitted_title:
                emitted_title = True
                draft_items.append(
                    {
                        "kind": "title",
                        "section": "Title",
                        "page": tb.page_num,
                        "display_text": doc_title,
                        "skip_by_default": False,
                    }
                )
            continue

        # Skip author affiliation / email blocks right under the title if short/noisy, or keep concise author line
        if _REFERENCES_HEADING_RE.match(txt) and len(txt) <= 40:
            in_references = True
            current_section = "References"
            draft_items.append(
                {
                    "kind": "heading",
                    "section": current_section,
                    "page": tb.page_num,
                    "display_text": txt,
                    "skip_by_default": True,
                }
            )
            continue

        is_heading = False
        if not in_references and len(txt) <= 95 and not txt.endswith("."):
            if _SECTION_HEADING_RE.match(txt):
                is_heading = True
            elif (tb.is_bold or tb.max_font_size >= body_font_size + 1.2) and len(txt.split()) <= 11:
                # Ensure it's not an inline run-in paragraph or equation fragment
                if re.match(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z]", txt):
                    is_heading = True

        if is_heading:
            current_section = txt
            draft_items.append(
                {
                    "kind": "heading",
                    "section": current_section,
                    "page": tb.page_num,
                    "display_text": txt,
                    "skip_by_default": False,
                }
            )
            continue

        if in_references:
            draft_items.append(
                {
                    "kind": "reference",
                    "section": "References",
                    "page": tb.page_num,
                    "display_text": txt,
                    "skip_by_default": True,
                }
            )
            continue

        # Merge paragraphs split mid-sentence across column/page boundaries
        if (
            draft_items
            and draft_items[-1]["kind"] == "paragraph"
            and not re.search(r"[.!?:;\"')\]\d]$", draft_items[-1]["display_text"])
            and txt[0].islower()
        ):
            prev_txt = draft_items[-1]["display_text"]
            if prev_txt.endswith("-"):
                draft_items[-1]["display_text"] = prev_txt[:-1] + txt
            else:
                draft_items[-1]["display_text"] = f"{prev_txt} {txt}"
            continue

        draft_items.append(
            {
                "kind": "paragraph",
                "section": current_section,
                "page": tb.page_num,
                "display_text": txt,
                "skip_by_default": False,
            }
        )

    # Match each figure to the paragraph that references it
    insert_after_map: dict[int, list[_ExtractedFigure]] = {}
    citing_contexts: dict[str, str] = {}

    for fig in all_figures:
        matched_indices: list[int] = []
        if fig.figure_num:
            # Match "Figure 2", "Fig. 2", "Figures 1 and 2", "Figs. 2-4"
            num_esc = re.escape(fig.figure_num)
            cite_pat = re.compile(
                rf"\b(?:Figures?|Figs?\.?)\s*(?:\d+[A-Za-z]?\s*(?:,|and|&|–|-)\s*)*{num_esc}(?![0-9A-Za-z])",
                re.IGNORECASE,
            )
            for d_idx, item in enumerate(draft_items):
                if item["kind"] == "paragraph" and cite_pat.search(item["display_text"]):
                    matched_indices.append(d_idx)

        if matched_indices:
            anchor_idx = matched_indices[0]
            ctx_snippets = [draft_items[i]["display_text"] for i in matched_indices[:2]]
            citing_contexts[fig.figure_id] = " ".join(ctx_snippets)[:950]
        else:
            # Place at the end of the non-reference paragraphs on the figure's page (or nearest preceding paragraph)
            page_candidates = [
                i
                for i, item in enumerate(draft_items)
                if item["page"] is not None and item["page"] <= fig.page_num and not item["skip_by_default"]
            ]
            anchor_idx = page_candidates[-1] if page_candidates else max(len(draft_items) - 1, 0)
            # Gather local page context for the VLM
            nearby = [
                item["display_text"]
                for item in draft_items[max(0, anchor_idx - 1) : anchor_idx + 2]
                if item["kind"] == "paragraph"
            ]
            citing_contexts[fig.figure_id] = " ".join(nearby)[:950]

        insert_after_map.setdefault(anchor_idx, []).append(fig)

    # Assemble final Segment stream
    segments: list[Segment] = []

    def append_figure_segment(fig: _ExtractedFigure, section_name: str | None) -> None:
        seg_idx = len(segments)
        cap_clean = fig.caption or f"{fig.figure_label} from page {fig.page_num}."
        placeholder_spoken = prepare_spoken_text(
            f"Now looking at {fig.figure_label}. {cap_clean}",
            strip_citations=strip_citations,
        )
        segments.append(
            Segment(
                id=f"seg_{seg_idx:04d}",
                index=seg_idx,
                kind="figure",
                section=section_name,
                page=fig.page_num,
                display_text=f"Reading {fig.figure_label}...",
                spoken_text=placeholder_spoken,
                skip_by_default=False,
                figure_id=fig.figure_id,
                figure_label=fig.figure_label,
                caption=fig.caption,
                citing_context=citing_contexts.get(fig.figure_id, ""),
                image_url=f"/api/documents/{doc_id}/figures/{fig.image_filename}",
                image_path=fig.image_path,
                figure_status="pending",
            )
        )

    if not draft_items and all_figures:
        for fig in all_figures:
            append_figure_segment(fig, "Figures")
    else:
        for d_idx, item in enumerate(draft_items):
            seg_idx = len(segments)
            disp = item["display_text"]
            spoken = prepare_spoken_text(disp, strip_citations=strip_citations)
            if item["kind"] == "heading":
                spoken = f"Section: {spoken}."
            if spoken:
                segments.append(
                    Segment(
                        id=f"seg_{seg_idx:04d}",
                        index=seg_idx,
                        kind=item["kind"],
                        section=item["section"],
                        page=item["page"],
                        display_text=disp,
                        spoken_text=spoken,
                        skip_by_default=item["skip_by_default"],
                    )
                )
            for fig in insert_after_map.get(d_idx, []):
                append_figure_segment(fig, item["section"])

    active_words = sum(len(s.spoken_text.split()) for s in segments if not s.skip_by_default)
    est_minutes = round(max(active_words / 155.0, 0.2), 1)

    return DocumentSession(
        doc_id=doc_id,
        title=doc_title,
        source_type="pdf",
        filename=filename,
        page_count=page_count,
        figure_count=len(all_figures),
        word_count=active_words,
        estimated_minutes=est_minutes,
        created_at=datetime.now(timezone.utc).isoformat(),
        segments=segments,
    )


def parse_text_document(
    raw_text: str,
    doc_id: str,
    title: str | None = None,
    strip_citations: bool = True,
) -> DocumentSession:
    """Parse pasted academic text, Markdown, or notes into clean spoken narrative segments."""
    lines = raw_text.replace("\r\n", "\n").split("\n")
    paragraphs: list[tuple[str, str]] = []  # (kind, text)
    buf: list[str] = []

    def flush_buf() -> None:
        if not buf:
            return
        joined = clean_display_text("\n".join(buf))
        buf.clear()
        if not joined:
            return
        if len(joined) <= 90 and not joined.endswith(".") and _SECTION_HEADING_RE.match(joined):
            paragraphs.append(("heading", joined))
        else:
            paragraphs.append(("paragraph", joined))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_buf()
            continue
        md_heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if md_heading:
            flush_buf()
            lvl = len(md_heading.group(1))
            heading_txt = clean_display_text(md_heading.group(2))
            paragraphs.append(("title" if lvl == 1 and not paragraphs else "heading", heading_txt))
            continue
        buf.append(stripped)
    flush_buf()

    inferred_title = title.strip() if title and title.strip() else ""
    if not inferred_title:
        if paragraphs and paragraphs[0][0] in ("title", "heading"):
            inferred_title = paragraphs[0][1]
        elif paragraphs:
            first_words = " ".join(paragraphs[0][1].split()[:9])
            inferred_title = first_words + ("…" if len(paragraphs[0][1].split()) > 9 else "")
        else:
            inferred_title = "Pasted Text"

    segments: list[Segment] = []
    current_section = "Text"
    in_references = False

    if title and title.strip() and (not paragraphs or paragraphs[0][1] != title.strip()):
        segments.append(
            Segment(
                id="seg_0000",
                index=0,
                kind="title",
                section="Title",
                page=None,
                display_text=title.strip(),
                spoken_text=prepare_spoken_text(title.strip(), strip_citations=strip_citations),
                skip_by_default=False,
            )
        )

    for kind, txt in paragraphs:
        if _REFERENCES_HEADING_RE.match(txt):
            in_references = True
            current_section = "References"
            kind = "heading"
        elif kind in ("title", "heading"):
            current_section = txt

        actual_kind = "reference" if (in_references and kind == "paragraph") else kind
        spoken = prepare_spoken_text(txt, strip_citations=strip_citations)
        if actual_kind == "heading":
            spoken = f"Section: {spoken}."
        if not spoken:
            continue

        seg_idx = len(segments)
        segments.append(
            Segment(
                id=f"seg_{seg_idx:04d}",
                index=seg_idx,
                kind=actual_kind,  # type: ignore[arg-type]
                section=current_section,
                page=None,
                display_text=txt,
                spoken_text=spoken,
                skip_by_default=in_references,
            )
        )

    active_words = sum(len(s.spoken_text.split()) for s in segments if not s.skip_by_default)
    return DocumentSession(
        doc_id=doc_id,
        title=inferred_title,
        source_type="text",
        filename=None,
        page_count=1,
        figure_count=0,
        word_count=active_words,
        estimated_minutes=round(max(active_words / 155.0, 0.1), 1),
        created_at=datetime.now(timezone.utc).isoformat(),
        segments=segments,
    )


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]
