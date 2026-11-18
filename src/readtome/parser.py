from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

import pymupdf

from .models import DocumentSession, Segment
from .speech_cleaner import clean_display_text, prepare_spoken_text

_FIG_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|FIGURE|FIG\.?)\s*(\d+[A-Za-z]?)\s*(?:[:.|—–\-]|\s+(?=[A-Z(]))",
    re.DOTALL,
)
_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:Table|Tab\.?|TABLE)\s*(\d+[A-Za-z]?|[IVX]+)\s*(?:[:.|—–\-]|\s+(?=[A-Z(]))",
    re.DOTALL,
)
_FALSE_CAPTION_VERB_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table|Tab\.?)\s*(?:\d+[A-Za-z]?|[IVX]+)\s+"
    r"(?:shows?|illustrates?|demonstrates?|presents?|compares?|depicts?|plots?|lists?|"
    r"summarizes?|is|are|was|were|reveals?|highlights?|confirms?|provides?|gives?|reports?)\b",
    re.IGNORECASE,
)

_SECTION_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,2}|[IVX]+\.)\s+[A-Z][^\n]{1,75}|"
    r"(?:Abstract|Introduction|Background|Related\s+Work|Preliminaries|Method(?:s|ology)?|"
    r"Approach|Architecture|Performance\s+Evaluation|Experiments?|Experimental\s+Setup|"
    r"Results?(?:\s+and\s+Discussion)?|Discussion|Ablation\s+Stud(?:y|ies)|Limitations?|"
    r"Conclusion(?:s)?|Acknowledg(?:e)?ments?|Appendix|References|Bibliography))$",
    re.IGNORECASE,
)

_REFERENCES_HEADING_RE = re.compile(r"^(?:\d+\.?\s*)?(?:References|Bibliography|Works\s+Cited)$", re.IGNORECASE)
_ARXIV_STAMP_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}", re.IGNORECASE)
_DOI_HEADER_RE = re.compile(r"^DOI:\s*10\.\d{4,9}/\S+$", re.IGNORECASE)


@dataclass
class _TextBlock:
    page_num: int
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
    kind_name: str  # "Figure" or "Table"
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
    """Merge overlapping or tightly touching visual sub-elements without crossing column gutters or captions."""
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
                # Keep horizontal merge tight (7pt) so figures in adjacent 2/3-col columns stay separate
                expanded = pymupdf.Rect(cur.x0 - 7, cur.y0 - 12, cur.x1 + 7, cur.y1 + 12)
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
            spans = line.get("spans", [])
            line_max_sz = max((float(s.get("size", 10.0)) for s in spans if s.get("text", "").strip()), default=10.0)
            span_strs: list[str] = []
            prev_nonempty_txt = ""
            prev_origin_y: float | None = None

            for span in spans:
                txt = span.get("text", "")
                if not txt.strip():
                    if txt:
                        span_strs.append(txt)
                    continue

                sz = float(span.get("size", 10.0))
                flags = int(span.get("flags", 0))
                origin = span.get("origin")
                origin_y = float(origin[1]) if origin and len(origin) >= 2 else None
                stripped_t = txt.strip()

                # Detect superscript numeric citations (e.g. "Law.17", "weights.27,29", "studies5")
                is_sup_span = bool(flags & 1) or (
                    sz < line_max_sz * 0.77
                    and prev_origin_y is not None
                    and origin_y is not None
                    and origin_y < prev_origin_y - 1.2
                )
                if (
                    is_sup_span
                    and re.fullmatch(r"\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*", stripped_t)
                    and prev_nonempty_txt
                    and re.search(r"[A-Za-z.,;:!?)\”\"']$", prev_nonempty_txt)
                    and not (
                        stripped_t in ("2", "3")
                        and re.search(r"\b(?:mm|cm|km|m|s|in|ft)$", prev_nonempty_txt)
                    )
                ):
                    span_strs.append(f" [{stripped_t}] ")
                    continue

                span_strs.append(txt)
                font_name = span.get("font", "").lower()
                is_span_bold = (
                    bool(flags & 16) or ("bold" in font_name) or ("medi" in font_name) or ("demi" in font_name)
                )
                if not saw_first_span:
                    first_span_bold = is_span_bold
                    saw_first_span = True
                n_chars = len(stripped_t)
                font_sizes.append(sz)
                char_weights.append(n_chars)
                total_chars += n_chars
                if is_span_bold:
                    bold_chars += n_chars
                prev_nonempty_txt = stripped_t
                if origin_y is not None:
                    prev_origin_y = origin_y

            if span_strs:
                lines_text.append("".join(span_strs))

        raw_text = "\n".join(lines_text).strip()
        clean_text = clean_display_text(raw_text)
        if not clean_text:
            continue

        # Ignore vertical margin watermarks (e.g. arXiv sidebars)
        if (bbox.x1 < 0.08 * w or bbox.x0 > 0.92 * w) and bbox.height > 2.2 * bbox.width:
            continue
        if _ARXIV_STAMP_RE.search(clean_text) and len(clean_text) < 120:
            continue
        if _DOI_HEADER_RE.match(clean_text):
            continue

        max_sz = max(font_sizes) if font_sizes else 10.0

        # Ignore running headers, magazine section slugs ("contributed articles"), footers, and page numbers
        if (bbox.y1 < 0.068 * h or bbox.y0 > 0.935 * h) and len(clean_text) < 125 and max_sz < 15.0:
            if not _FIG_CAPTION_RE.match(clean_text) and not _TABLE_CAPTION_RE.match(clean_text):
                continue
        if re.fullmatch(r"\d{1,4}", clean_text) and (bbox.y0 > 0.89 * h or bbox.y1 < 0.08 * h):
            continue

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


def _classify_caption(tb: _TextBlock, body_font_size: float) -> tuple[str, str] | None:
    """Return ('Figure' | 'Table', num_str) if `tb` is a figure or table caption, else None."""
    for kind_name, pat in (("Figure", _FIG_CAPTION_RE), ("Table", _TABLE_CAPTION_RE)):
        m = pat.match(tb.clean_text)
        if not m:
            continue
        num_str = m.group(1)
        has_explicit_sep = bool(
            re.match(
                r"^\s*(?:Figure|Fig\.?|Table|Tab\.?)\s*(?:\d+[A-Za-z]?|[IVX]+)\s*[:.|—–\-]",
                tb.clean_text,
                re.IGNORECASE,
            )
        )
        if not has_explicit_sep and not tb.first_span_bold:
            if _FALSE_CAPTION_VERB_RE.match(tb.clean_text):
                return None
            if tb.avg_font_size > body_font_size + 0.5:
                return None
        return kind_name, num_str
    return None


def _collect_visual_regions(
    page: pymupdf.Page,
    caption_rects: list[pymupdf.Rect],
    table_caption_rects: list[pymupdf.Rect],
) -> list[pymupdf.Rect]:
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
            if r.is_empty or r.width < 38 or r.height < 35:
                continue
            if r.width > 0.96 * w and r.height > 0.96 * h:
                continue
            raw_boxes.append(r)

    # 2. Vector graphics clusters — filter out column divider lines & header/footer rules BEFORE clustering
    try:
        all_drawings = page.get_drawings()
    except Exception:
        all_drawings = []

    filtered_drawings = []
    horiz_rules: list[pymupdf.Rect] = []
    for d in all_drawings:
        dr = pymupdf.Rect(d.get("rect", (0, 0, 0, 0))) & page_rect
        if dr.is_empty:
            continue
        has_fill = d.get("fill") is not None
        items = d.get("items") or []

        # Skip stroke-only paths made purely of vertical column dividers or margin header/footer rules
        # (Adobe InDesign / CACM / IEEE often combine multiple column divider rules into one compound path)
        if not has_fill and items:
            all_dividers = True
            for it in items:
                op = it[0]
                if op == "l" and len(it) >= 3:
                    p1, p2 = it[1], it[2]
                    dx = abs(float(p1.x) - float(p2.x))
                    dy = abs(float(p1.y) - float(p2.y))
                    is_vert_col_rule = dx < 3.5 and dy > 85
                    is_margin_horiz_rule = (
                        dy < 3.5
                        and dx > 0.45 * w
                        and (max(float(p1.y), float(p2.y)) < 0.08 * h or min(float(p1.y), float(p2.y)) > 0.92 * h)
                    )
                    if not (is_vert_col_rule or is_margin_horiz_rule):
                        all_dividers = False
                        break
                else:
                    all_dividers = False
                    break
            if all_dividers:
                continue

        if dr.width < 3.5 and dr.height > 85 and not has_fill:
            continue
        if dr.height < 3.5 and dr.width > 0.50 * w and (dr.y1 < 0.08 * h or dr.y0 > 0.92 * h) and not has_fill:
            continue
        if dr.width < 9 and dr.height < 9:
            continue
        if dr.height < 4.0 and dr.width >= 90:
            horiz_rules.append(dr)
        filtered_drawings.append(d)

    try:
        drawing_clusters = (
            page.cluster_drawings(drawings=filtered_drawings, x_tolerance=10, y_tolerance=10)
            if filtered_drawings
            else []
        )
    except Exception:
        drawing_clusters = []

    for dr in drawing_clusters:
        r = pymupdf.Rect(dr) & page_rect
        if r.is_empty:
            continue
        if r.width < 46 or r.height < 34:
            continue
        if r.width > 0.96 * w and r.height > 0.96 * h:
            continue
        raw_boxes.append(r)

    # 3. Unboxed LaTeX booktabs tables (horizontal rules aligned under/above a Table caption)
    if table_caption_rects and len(horiz_rules) >= 2:
        for tcap in table_caption_rects:
            aligned_rules = [
                hr
                for hr in horiz_rules
                if _horiz_overlap(hr, tcap) >= 0.45 and -260 <= (hr.y0 - tcap.y1) <= 260
            ]
            if len(aligned_rules) >= 2:
                u = pymupdf.Rect(aligned_rules[0])
                for hr in aligned_rules[1:]:
                    u |= hr
                if u.height >= 28:
                    raw_boxes.append(u)

    return _merge_visual_boxes(raw_boxes, caption_rects)


def _extract_figures_on_page(
    page: pymupdf.Page,
    page_num: int,
    text_blocks: list[_TextBlock],
    body_font_size: float,
    figures_dir: Path,
    start_fig_counter: int,
) -> tuple[list[_ExtractedFigure], set[int], set[int]]:
    """Detect & crop both Figures and Tables, returning (figures, excluded_block_idxs, callout_box_block_idxs)."""
    page_rect = page.rect

    caption_entries: list[tuple[int, str, str, _TextBlock]] = []  # (block_idx, kind_name, num_str, tb)
    table_caption_rects: list[pymupdf.Rect] = []
    for idx, tb in enumerate(text_blocks):
        classified = _classify_caption(tb, body_font_size)
        if classified is not None:
            kind_name, num_str = classified
            caption_entries.append((idx, kind_name, num_str, tb))
            if kind_name == "Table":
                table_caption_rects.append(tb.bbox)

    barriers = [tb.bbox for _, _, _, tb in caption_entries]
    visual_boxes = _collect_visual_regions(page, barriers, table_caption_rects)
    used_visuals: set[int] = set()
    excluded_block_idxs: set[int] = set()
    extracted: list[_ExtractedFigure] = []
    fig_counter = start_fig_counter

    for cap_idx, kind_name, fig_num, cap_tb in caption_entries:
        cap_rect = cap_tb.bbox
        excluded_block_idxs.add(cap_idx)
        other_barriers = [b for b in barriers if b != cap_rect]

        # 1. Check if a shaded container box directly wraps BOTH the caption bar and the diagram/table
        wrapping_idxs: list[int] = []
        for v_idx, vr in enumerate(visual_boxes):
            if v_idx in used_visuals:
                continue
            if _rect_overlap_ratio(cap_rect, vr) >= 0.55 and vr.height >= cap_rect.height + 28:
                wrapping_idxs.append(v_idx)

        above_candidates: list[tuple[float, int]] = []
        below_candidates: list[tuple[float, int]] = []

        if not wrapping_idxs:
            for v_idx, vr in enumerate(visual_boxes):
                if v_idx in used_visuals:
                    continue
                if _horiz_overlap(vr, cap_rect) < 0.25:
                    continue

                # Candidate above caption (classic arXiv / LaTeX style)
                if vr.y1 <= cap_rect.y0 + 20 and (cap_rect.y0 - vr.y1) <= 155:
                    if not _separated_by_barrier(vr, cap_rect, other_barriers):
                        gap = max(0.0, cap_rect.y0 - vr.y1)
                        above_candidates.append((gap, v_idx))

                # Candidate below caption (CACM / IEEE / Nature header-bar style)
                if vr.y0 >= cap_rect.y0 - 6 and (vr.y0 - cap_rect.y1) <= 155 and vr.y1 > cap_rect.y1 + 20:
                    if not _separated_by_barrier(cap_rect, vr, other_barriers):
                        gap = max(0.0, vr.y0 - cap_rect.y1)
                        below_candidates.append((gap, v_idx))

        if wrapping_idxs:
            chosen_vis_idxs = wrapping_idxs
        elif above_candidates and below_candidates:
            # Pick whichever side sits closer to the caption (0pt touching header bar beats 22pt gap to prior table)
            min_above = min(g for g, _ in above_candidates)
            min_below = min(g for g, _ in below_candidates)
            if min_below + 4.0 < min_above:
                chosen_vis_idxs = [i for _, i in below_candidates]
            else:
                chosen_vis_idxs = [i for _, i in above_candidates]
        elif above_candidates:
            chosen_vis_idxs = [i for _, i in above_candidates]
        elif below_candidates:
            chosen_vis_idxs = [i for _, i in below_candidates]
        else:
            chosen_vis_idxs = []

        if not chosen_vis_idxs:
            if kind_name == "Table":
                continue
            fallback_top = max(page_rect.y0 + 28, cap_rect.y0 - 200)
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

        # Absorb sub-caption notes sitting right between a top caption bar and the visual (e.g., CACM Figure 3 / 4)
        caption_full = cap_tb.clean_text
        between_zone = pymupdf.Rect(
            min(cap_rect.x0, vis_union.x0),
            min(cap_rect.y0, vis_union.y0),
            max(cap_rect.x1, vis_union.x1),
            max(cap_rect.y1, vis_union.y1),
        )
        for idx_other, other_tb in enumerate(text_blocks):
            if idx_other in excluded_block_idxs:
                continue
            if (
                other_tb.bbox.y0 >= cap_rect.y1 - 3
                and other_tb.bbox.y1 <= vis_union.y0 + 6
                and _horiz_overlap(other_tb.bbox, between_zone) >= 0.45
            ):
                caption_full = f"{caption_full} {other_tb.clean_text}"
                excluded_block_idxs.add(idx_other)

        # Expand moderately for axis labels/ticks without crossing narrow 3-column gutters into neighbor prose
        expanded_vis = pymupdf.Rect(
            max(page_rect.x0 + 6, vis_union.x0 - 14),
            max(page_rect.y0 + 6, vis_union.y0 - 18),
            min(page_rect.x1 - 6, vis_union.x1 + 14),
            min(page_rect.y1 - 6, vis_union.y1 + 20),
        )
        if expanded_vis.y0 < cap_rect.y0 and vis_union.y1 <= cap_rect.y0 + 16:
            expanded_vis.y1 = min(expanded_vis.y1, cap_rect.y0 - 2)

        for idx_other, other_tb in enumerate(text_blocks):
            if idx_other in excluded_block_idxs:
                continue
            # All text inside a figure/table's visual box (axis ticks, table cells, diagram boxes) is excluded from prose
            if _rect_overlap_ratio(other_tb.bbox, vis_union) >= 0.50:
                excluded_block_idxs.add(idx_other)
                continue
            inter = expanded_vis & other_tb.bbox
            if inter.is_empty:
                continue
            # Only absorb short outer axis labels that align horizontally with this figure (not adjacent column prose)
            if (
                len(other_tb.clean_text) <= 90
                and _horiz_overlap(other_tb.bbox, vis_union) >= 0.25
                and not _SECTION_HEADING_RE.match(other_tb.clean_text)
            ):
                vis_union |= other_tb.bbox
                excluded_block_idxs.add(idx_other)

        # Compute clean crop bounds without slicing through external captions
        crop_y0 = max(page_rect.y0 + 4, vis_union.y0 - 8)
        crop_y1 = min(page_rect.y1 - 4, vis_union.y1 + 8)
        if vis_union.y1 <= cap_rect.y0 + 16 and crop_y0 < cap_rect.y0:
            crop_y1 = min(crop_y1, cap_rect.y0 - 2)
        elif vis_union.y0 >= cap_rect.y1 - 12 and crop_y1 > cap_rect.y1 + 24:
            crop_y0 = max(crop_y0, cap_rect.y1 + 2)
        elif _rect_overlap_ratio(cap_rect, vis_union) >= 0.55 and (vis_union.y1 - cap_rect.y1) >= 35:
            # Caption sits in a top header bar inside the shaded container: start crop cleanly below the header bar
            crop_y0 = cap_rect.y1 + 1

        crop_rect = pymupdf.Rect(
            max(page_rect.x0 + 4, vis_union.x0 - 8),
            crop_y0,
            min(page_rect.x1 - 4, vis_union.x1 + 8),
            crop_y1,
        )
        if crop_rect.width < 45 or crop_rect.height < 32:
            continue

        fig_counter += 1
        prefix = "fig" if kind_name == "Figure" else "tab"
        fig_id = f"{prefix}_{fig_counter}"
        img_filename = f"{fig_id}.png"
        img_path = figures_dir / img_filename
        pix = page.get_pixmap(clip=crop_rect, dpi=220, alpha=False)
        pix.save(str(img_path))

        extracted.append(
            _ExtractedFigure(
                figure_id=fig_id,
                kind_name=kind_name,
                figure_num=fig_num,
                figure_label=f"{kind_name} {fig_num}",
                page_num=page_num,
                crop_rect=crop_rect,
                exclusion_rects=[crop_rect, cap_rect],
                caption=caption_full,
                image_filename=img_filename,
                image_path=str(img_path),
                anchor_y=min(crop_rect.y0, cap_rect.y0),
            )
        )

    # Also capture standalone uncaptioned figures, while separating shaded text callout boxes (e.g. "key insights")
    callout_block_idxs: set[int] = set()
    for v_idx, vr in enumerate(visual_boxes):
        if v_idx in used_visuals:
            continue
        inside_pairs = [
            (i, tb)
            for i, tb in enumerate(text_blocks)
            if i not in excluded_block_idxs and _rect_overlap_ratio(tb.bbox, vr) >= 0.55
        ]
        inside_chars = sum(len(tb.clean_text) for _, tb in inside_pairs)
        if inside_chars >= 60 and (any(len(tb.clean_text) > 55 for _, tb in inside_pairs) or len(inside_pairs) >= 2):
            # Shaded prose sidebar / callout box: defer its blocks past inline columns so split words don't break
            for i, _ in inside_pairs:
                callout_block_idxs.add(i)
            continue

        if vr.width < 130 or vr.height < 100 or (vr.width * vr.height) < 20000:
            continue

        crop_rect = pymupdf.Rect(
            max(page_rect.x0 + 4, vr.x0 - 8),
            max(page_rect.y0 + 4, vr.y0 - 8),
            min(page_rect.x1 - 4, vr.x1 + 8),
            min(page_rect.y1 - 4, vr.y1 + 8),
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
                kind_name="Figure",
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
    return extracted, excluded_block_idxs, callout_block_idxs


def _split_by_x_gutters(blocks: list[_TextBlock], page_width: float) -> list[list[_TextBlock]] | None:
    """Split `blocks` left-to-right across vertical column gutters (supports 1, 2, 3, or N columns)."""
    if len(blocks) <= 1:
        return None

    min_gutter = max(7.0, page_width * 0.011)
    min_col_width = page_width * 0.115

    by_x = sorted(blocks, key=lambda b: (b.bbox.x0, b.bbox.x1))
    raw_groups: list[list[_TextBlock]] = [[by_x[0]]]
    cur_x1 = by_x[0].bbox.x1

    for b in by_x[1:]:
        if b.bbox.x0 >= cur_x1 + min_gutter:
            raw_groups.append([b])
            cur_x1 = b.bbox.x1
        else:
            raw_groups[-1].append(b)
            cur_x1 = max(cur_x1, b.bbox.x1)

    if len(raw_groups) <= 1:
        return None

    def is_substantial_col(grp: list[_TextBlock]) -> bool:
        grp_x0 = min(x.bbox.x0 for x in grp)
        grp_x1 = max(x.bbox.x1 for x in grp)
        chars = sum(len(x.clean_text) for x in grp)
        return (grp_x1 - grp_x0) >= min_col_width and chars >= 18

    # Merge tiny margin artifacts (like equation tags "(1)" or stray bullets) into their adjacent group
    merged_groups: list[list[_TextBlock]] = []
    for grp in raw_groups:
        if not merged_groups:
            merged_groups.append(grp)
            continue
        if not is_substantial_col(grp):
            merged_groups[-1].extend(grp)
        elif not is_substantial_col(merged_groups[-1]):
            merged_groups[-1].extend(grp)
        else:
            merged_groups.append(grp)

    if len(merged_groups) <= 1:
        return None
    return merged_groups


def _split_by_y_bands(blocks: list[_TextBlock]) -> list[list[_TextBlock]] | None:
    """Split `blocks` top-to-bottom across horizontal whitespace bands."""
    if len(blocks) <= 1:
        return None

    by_y = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.y1))
    bands: list[list[_TextBlock]] = [[by_y[0]]]
    cur_y1 = by_y[0].bbox.y1

    for b in by_y[1:]:
        # Allow tiny 1.5pt ascender/descender box touch when lines are clearly stacked
        if b.bbox.y0 >= cur_y1 - 1.5:
            bands.append([b])
            cur_y1 = b.bbox.y1
        else:
            bands[-1].append(b)
            cur_y1 = max(cur_y1, b.bbox.y1)

    return bands if len(bands) >= 2 else None


def _order_blocks_xy_cut(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    """Recursive X-first XY-Cut reading order for 1-column, 2-column, 3-column, and mixed-spanning pages."""
    if len(blocks) <= 1:
        return list(blocks)

    # 1. Prefer vertical column cuts whenever clean vertical gutters run through this zone
    x_cols = _split_by_x_gutters(blocks, page_width)
    if x_cols is not None:
        ordered: list[_TextBlock] = []
        for col in x_cols:
            ordered.extend(_order_blocks_xy_cut(col, page_width))
        return ordered

    # 2. Otherwise split horizontally across full-width / multi-column spanning bands
    y_bands = _split_by_y_bands(blocks)
    if y_bands is not None:
        ordered = []
        for band in y_bands:
            ordered.extend(_order_blocks_xy_cut(band, page_width))
        return ordered

    # 3. Irreducible L-shaped wrap: peel off the topmost wide block and recurse on the remainder
    top_idx = min(range(len(blocks)), key=lambda i: (round(blocks[i].bbox.y0, 1), round(blocks[i].bbox.x0, 1)))
    rest = [b for i, b in enumerate(blocks) if i != top_idx]
    return [blocks[top_idx]] + _order_blocks_xy_cut(rest, page_width)


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
    figures_dir = session_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    page1_height = doc[0].rect.height if page_count > 0 else 792.0

    raw_page_blocks: list[list[_TextBlock]] = []
    for p_idx in range(page_count):
        raw_page_blocks.append(_extract_page_text_blocks(doc[p_idx], p_idx + 1))

    body_font_size = _estimate_body_font_size(raw_page_blocks)

    all_figures: list[_ExtractedFigure] = []
    ordered_text_blocks: list[_TextBlock] = []
    deferred_callouts: list[_TextBlock] = []
    fig_counter = 0

    for p_idx in range(page_count):
        page = doc[p_idx]
        p_blocks = raw_page_blocks[p_idx]
        page_figs, excluded_idxs, callout_idxs = _extract_figures_on_page(
            page=page,
            page_num=p_idx + 1,
            text_blocks=p_blocks,
            body_font_size=body_font_size,
            figures_dir=figures_dir,
            start_fig_counter=fig_counter,
        )
        fig_counter += len(page_figs)
        all_figures.extend(page_figs)

        kept_blocks: list[_TextBlock] = []
        page_callouts: list[_TextBlock] = []
        for i, b in enumerate(p_blocks):
            if i in excluded_idxs:
                continue
            if i in callout_idxs:
                page_callouts.append(b)
                continue
            # Skip oversized magazine pull-quotes on body pages (e.g. CACM middle-column callouts)
            if (
                b.page_num >= 2
                and b.avg_font_size >= body_font_size + 3.2
                and len(b.clean_text.split()) >= 7
                and not _SECTION_HEADING_RE.match(b.clean_text)
            ):
                continue
            kept_blocks.append(b)

        ordered_text_blocks.extend(_order_blocks_xy_cut(kept_blocks, page.rect.width))
        if page_callouts:
            deferred_callouts.extend(_order_blocks_xy_cut(page_callouts, page.rect.width))

    doc.close()

    # Identify Document Title on page 1 (and merge multi-line stacked titles up to 8 lines)
    doc_title = (filename or "Untitled Paper").rsplit(".", 1)[0]
    title_block_indices: set[int] = set()
    page1_candidates = [
        (i, b)
        for i, b in enumerate(ordered_text_blocks[:12])
        if b.page_num == 1
        and b.bbox.y0 < page1_height * 0.52
        and 3 <= len(b.clean_text) <= 260
        and not _SECTION_HEADING_RE.match(b.clean_text)
    ]
    if page1_candidates:
        best_i, best_b = max(page1_candidates, key=lambda item: item[1].max_font_size)
        if best_b.max_font_size >= body_font_size + 1.5:
            title_parts = [best_b.clean_text]
            title_block_indices.add(best_i)
            prev_b = best_b
            for nxt_i in range(best_i + 1, min(best_i + 8, len(ordered_text_blocks))):
                nxt_b = ordered_text_blocks[nxt_i]
                if (
                    nxt_b.page_num == 1
                    and abs(nxt_b.max_font_size - best_b.max_font_size) <= 0.8
                    and -12 <= (nxt_b.bbox.y0 - prev_b.bbox.y1) <= 22
                    and not _SECTION_HEADING_RE.match(nxt_b.clean_text)
                ):
                    title_parts.append(nxt_b.clean_text)
                    title_block_indices.add(nxt_i)
                    prev_b = nxt_b
                else:
                    break
            doc_title = clean_display_text("\n".join(title_parts))

    draft_items: list[dict] = []
    current_section = "Overview"
    in_references = False

    if title_block_indices:
        draft_items.append(
            {
                "kind": "title",
                "section": "Title",
                "page": 1,
                "display_text": doc_title,
                "skip_by_default": False,
            }
        )

    for cb in deferred_callouts:
        if cb.clean_text:
            draft_items.append(
                {
                    "kind": "paragraph",
                    "section": "Key Insights",
                    "page": cb.page_num,
                    "display_text": cb.clean_text,
                    "skip_by_default": False,
                }
            )

    for idx, tb in enumerate(ordered_text_blocks):
        txt = tb.clean_text
        if not txt or idx in title_block_indices:
            continue

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
                if re.match(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z]", txt) and not txt.upper().startswith("BY "):
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

        # Merge sentences split mid-clause across column bottoms or page breaks
        if (
            draft_items
            and draft_items[-1]["kind"] == "paragraph"
            and not re.search(r"[.!?:;\"')\]\d]$", draft_items[-1]["display_text"])
            and txt[0].islower()
        ):
            prev_txt = draft_items[-1]["display_text"]
            if prev_txt.endswith("-"):
                joined_split = clean_display_text(f"{prev_txt}\n{txt}")
                draft_items[-1]["display_text"] = joined_split
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

    # Match each Figure / Table to the first paragraph that cites it
    insert_after_map: dict[int, list[_ExtractedFigure]] = {}
    citing_contexts: dict[str, str] = {}

    for fig in all_figures:
        matched_indices: list[int] = []
        if fig.figure_num:
            num_esc = re.escape(fig.figure_num)
            if fig.kind_name == "Table":
                cite_pat = re.compile(
                    rf"\b(?:Tables?|Tabs?\.?)\s*(?:(?:\d+[A-Za-z]?|[IVX]+)\s*(?:,|and|&|–|-)\s*)*{num_esc}(?![0-9A-Za-z])",
                    re.IGNORECASE,
                )
            else:
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
            page_candidates = [
                i
                for i, item in enumerate(draft_items)
                if item["page"] is not None and item["page"] <= fig.page_num and not item["skip_by_default"]
            ]
            anchor_idx = page_candidates[-1] if page_candidates else max(len(draft_items) - 1, 0)
            nearby = [
                item["display_text"]
                for item in draft_items[max(0, anchor_idx - 1) : anchor_idx + 2]
                if item["kind"] == "paragraph"
            ]
            citing_contexts[fig.figure_id] = " ".join(nearby)[:950]

        insert_after_map.setdefault(anchor_idx, []).append(fig)

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
    lines = raw_text.replace("\r\n", "\n").split("\n")
    paragraphs: list[tuple[str, str]] = []
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
            inferred_title = first_words + ("..." if len(paragraphs[0][1].split()) > 9 else "")
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
