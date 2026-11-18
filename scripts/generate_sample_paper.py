from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "readtome_mpl"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pymupdf


def _fig_to_png_bytes(fig: plt.Figure, dpi: int = 180) -> bytes:
    raw = io.BytesIO()
    fig.savefig(raw, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    raw.seek(0)
    with Image.open(raw) as im:
        out = io.BytesIO()
        im.convert("RGB").save(out, format="PNG", optimize=True)
        return out.getvalue()


def _make_fig1_architecture_png() -> bytes:
    fig, ax = plt.subplots(figsize=(4.6, 2.35), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        (0.3, 1.35, 2.0, 1.7, "#E0F2FE", "#0284C7", "Document\nPatches\n(224 × 224)"),
        (2.85, 1.35, 2.15, 1.7, "#FEF3C7", "#D97706", "Top-k Sparse\nRouter\n(α = 0.25)"),
        (5.55, 2.35, 2.0, 1.35, "#DCFCE7", "#16A34A", "Local Window\nAttention"),
        (5.55, 0.65, 2.0, 1.35, "#F3E8FF", "#9333EA", "Global Memory\nTokens (64)"),
        (8.05, 1.35, 1.7, 1.7, "#FFE4E6", "#E11D48", "Fused Decoder\nOutput"),
    ]
    for x, y, w, h, fc, ec, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.6,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.6, fontweight="bold", color="#0F172A")

    arrows = [
        ((2.42, 2.2), (2.78, 2.2)),
        ((5.12, 2.55), (5.48, 2.95)),
        ((5.12, 1.85), (5.48, 1.35)),
        ((7.65, 2.95), (7.98, 2.45)),
        ((7.65, 1.35), (7.98, 1.95)),
    ]
    for start, end in arrows:
        arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.4, color="#334155")
        ax.add_patch(arr)

    ax.text(5.0, 4.12, "Adaptive Token Routing Pipeline", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1E293B")
    fig.tight_layout(pad=0.25)
    return _fig_to_png_bytes(fig)


def _make_fig2_scaling_png() -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.55), dpi=180)

    ctx = [4, 8, 16, 32, 64, 128]
    dense_ppl = [5.4, 5.6, 6.3, 7.9, 10.8, 15.2]
    window_ppl = [5.5, 5.9, 6.8, 8.4, 11.1, 14.0]
    ours_ppl = [5.4, 5.45, 5.55, 5.72, 5.95, 6.18]

    ax1.plot(ctx, dense_ppl, "o--", color="#94A3B8", linewidth=1.8, markersize=4.5, label="Dense Attention")
    ax1.plot(ctx, window_ppl, "^:", color="#F59E0B", linewidth=1.8, markersize=4.5, label="Sliding Window")
    ax1.plot(ctx, ours_ppl, "s-", color="#2563EB", linewidth=2.2, markersize=5, label="SparseRouter (Ours)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ctx, [f"{c}K" for c in ctx])
    ax1.set_xlabel("Context Length (Tokens)", fontsize=8)
    ax1.set_ylabel("Validation Perplexity (↓ lower is better)", fontsize=8)
    ax1.set_title("(a) Long-Context Perplexity Scaling", fontsize=8.8, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=6.8, frameon=True)
    ax1.tick_params(labelsize=7.2)

    models = ["Dense\nAttn", "Flash\nWindow", "Linear\nRNN", "SparseRouter\n(Ours)"]
    tps = [42, 78, 96, 144]
    colors = ["#CBD5E1", "#FCD34D", "#A7F3D0", "#2563EB"]
    bars = ax2.bar(models, tps, color=colors, edgecolor="#1E293B", linewidth=0.8, width=0.56)
    for bar, val in zip(bars, tps):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 3.5, f"{val} tok/s", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax2.set_ylim(0, 170)
    ax2.set_ylabel("Decode Throughput (tokens / sec ↑)", fontsize=8)
    ax2.set_title("(b) On-Device Throughput at 64K Context", fontsize=8.8, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.tick_params(labelsize=7.2)

    fig.tight_layout(pad=0.55)
    return _fig_to_png_bytes(fig)


def _make_fig3_ablation_png() -> bytes:
    fig, ax = plt.subplots(figsize=(4.4, 2.35), dpi=180)
    ratios = np.array([0.10, 0.15, 0.25, 0.50, 1.00])
    sciqa = np.array([78.2, 84.6, 89.4, 89.9, 90.1])
    docvqa = np.array([74.0, 81.3, 87.8, 88.4, 88.6])

    x_pos = np.arange(len(ratios))
    w = 0.34
    b1 = ax.bar(x_pos - w / 2, sciqa, width=w, label="ScienceQA (%)", color="#0EA5E9", edgecolor="#0F172A", linewidth=0.7)
    b2 = ax.bar(x_pos + w / 2, docvqa, width=w, label="DocVQA (%)", color="#6366F1", edgecolor="#0F172A", linewidth=0.7)

    ax.axvline(2, color="#DC2626", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.text(2.08, 72.0, "Optimal α=0.25\n(3.4× faster)", fontsize=6.6, color="#DC2626", fontweight="bold")

    ax.set_xticks(x_pos, [f"{r:.2f}" for r in ratios])
    ax.set_ylim(68, 95)
    ax.set_xlabel("Active Token Routing Ratio (α)", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.set_title("Accuracy vs. Active Routing Budget", fontsize=8.6, fontweight="bold")
    ax.legend(fontsize=6.8, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=7.2)

    for b in (b1[2], b2[2]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout(pad=0.4)
    return _fig_to_png_bytes(fig)


def generate_sample_pdf(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    W, H = 612.0, 792.0  # US Letter

    css = """
    body { font-family: serif; font-size: 9.5pt; line-height: 1.32; color: #111827; margin: 0; padding: 0; }
    .title { font-family: sans-serif; font-size: 15pt; font-weight: bold; text-align: center; line-height: 1.22; color: #0F172A; }
    .authors { font-family: sans-serif; font-size: 9.2pt; text-align: center; color: #374151; }
    .h1 { font-family: sans-serif; font-size: 11.2pt; font-weight: bold; color: #0F172A; }
    .caption { font-family: sans-serif; font-size: 8.2pt; line-height: 1.26; color: #1F2937; }
    .p { text-align: justify; }
    .ref { font-size: 8.1pt; line-height: 1.25; color: #374151; }
    """

    # ---------------- PAGE 1 ----------------
    p1 = doc.new_page(width=W, height=H)
    p1.insert_htmlbox(pymupdf.Rect(54, 20, W - 54, 38), '<div style="font-family:sans-serif;font-size:7.8pt;color:#6B7280;">Preprint • Conference on Efficient Multimodal Systems (2026)</div>')
    p1.insert_htmlbox(pymupdf.Rect(290, 756, 322, 774), '<div style="font-family:sans-serif;font-size:8.5pt;text-align:center;color:#6B7280;">1</div>')

    # Full-width Title, Authors, Abstract
    p1.insert_htmlbox(
        pymupdf.Rect(54, 46, W - 54, 96),
        '<div class="title">SparseRouter-VL: Sub-Linear Context Scaling for On-Device Multimodal Reasoning</div>',
        css=css,
    )
    p1.insert_htmlbox(
        pymupdf.Rect(72, 98, W - 72, 120),
        '<div class="authors">Elena Vance, Marcus Chen, and Priya Nair &bull; Systems &amp; Machine Intelligence Lab</div>',
        css=css,
    )
    p1.insert_htmlbox(pymupdf.Rect(72, 126, 220, 144), '<div class="h1">Abstract</div>', css=css)
    p1.insert_htmlbox(
        pymupdf.Rect(72, 145, W - 72, 232),
        '<div class="p">Deploying vision-language models on consumer laptops remains bottlenecked by quadratic self-attention over dense visual patches and long document contexts [1, 4, 9]. '
        "We introduce SparseRouter-VL, a lightweight routing architecture that dynamically selects only &alpha; = 0.25 of informative visual and text tokens for global memory fusion while processing local neighborhoods in linear time (e.g., 224 &times; 224 document tiles in 180 &mu;s). "
        "Across long-context document benchmarks up to 128K tokens, SparseRouter-VL achieves a 3.4&times; decoding speedup on Apple Silicon unified memory while matching dense baselines within &plusmn;0.7% accuracy.</div>",
        css=css,
    )

    col_left_x0, col_left_x1 = 54.0, 294.0
    col_right_x0, col_right_x1 = 318.0, 558.0

    # Left Column
    p1.insert_htmlbox(pymupdf.Rect(col_left_x0, 242, col_left_x1, 260), '<div class="h1">1 Introduction</div>', css=css)
    p1.insert_htmlbox(
        pymupdf.Rect(col_left_x0, 262, col_left_x1, 422),
        '<div class="p">Modern scientific workflows increasingly rely on local multimodal assistants capable of reading dense technical papers, charts, and architectural diagrams without transmitting private data to external servers (Vaswani et al., 2017; Dao et al., 2024). '
        "However, standard dense transformers compute pairwise attention over every visual patch and text token, causing latency and memory footprint to grow rapidly beyond 16K context lengths [2, 7].</div>",
        css=css,
    )
    p1.insert_htmlbox(
        pymupdf.Rect(col_left_x0, 424, col_left_x1, 566),
        '<div class="p">To overcome this limitation, we propose an adaptive two-branch routing mechanism illustrated in Figure 1. '
        "Instead of treating background whitespace and redundant prose tokens equally, a learned lightweight gate routes high-salience tokens into a compact 64-slot global memory bank while keeping fine-grained spatial detail inside fast local window attention.</div>",
        css=css,
    )
    p1.insert_htmlbox(pymupdf.Rect(col_left_x0, 576, col_left_x1, 594), '<div class="h1">2 Architecture</div>', css=css)
    p1.insert_htmlbox(
        pymupdf.Rect(col_left_x0, 596, col_left_x1, 742),
        '<div class="p">Given an input stream of 224 &times; 224 image patches and tokenized text, our router scores each token embedding using a single linear projection. '
        "Only the top fraction &alpha; of tokens participate in global cross-modal attention, whereas all remaining tokens attend within fixed windows of 512 tokens, i.e., reducing global complexity by more than an order of magnitude.</div>",
        css=css,
    )

    # Right Column: Figure 1 + prose
    p1.insert_image(pymupdf.Rect(col_right_x0, 242, col_right_x1, 368), stream=_make_fig1_architecture_png())
    p1.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 372, col_right_x1, 434),
        '<div class="caption"><b>Figure 1:</b> Overview of the SparseRouter-VL pipeline. Input 224 &times; 224 patches pass through Top-k Sparse Router (&alpha; = 0.25), splitting into Local Window Attention and Global Memory Tokens before Fused Decoder Output.</div>',
        css=css,
    )
    p1.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 444, col_right_x1, 592),
        '<div class="p">During decoding, the global memory bank caches compressed key-value states across pages so cross-section references&mdash;such as linking an equation in Section 2 to a plot in Section 3&mdash;remain preserved without recomputing full prefix attention [11, 14].</div>',
        css=css,
    )
    p1.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 596, col_right_x1, 742),
        '<div class="p">We train the router end-to-end with an auxiliary load-balancing loss weighted by &lambda; = 0.01 so that visual chart labels, axis ticks, and mathematical symbols consistently receive priority allocation in the global branch.</div>',
        css=css,
    )

    # ---------------- PAGE 2 ----------------
    p2 = doc.new_page(width=W, height=H)
    p2.insert_htmlbox(pymupdf.Rect(54, 20, W - 54, 38), '<div style="font-family:sans-serif;font-size:7.8pt;color:#6B7280;">Vance et al. • SparseRouter-VL</div>')
    p2.insert_htmlbox(pymupdf.Rect(290, 756, 322, 774), '<div style="font-family:sans-serif;font-size:8.5pt;text-align:center;color:#6B7280;">2</div>')

    # Full-width Figure 2
    p2.insert_image(pymupdf.Rect(58, 46, W - 58, 222), stream=_make_fig2_scaling_png())
    p2.insert_htmlbox(
        pymupdf.Rect(58, 226, W - 58, 268),
        '<div class="caption"><b>Figure 2:</b> (a) Validation perplexity across context lengths from 4K to 128K tokens comparing Dense Attention, Sliding Window, and SparseRouter. (b) On-device decoding throughput at 64K context on Apple Silicon.</div>',
        css=css,
    )

    # Page 2 Left Column
    p2.insert_htmlbox(pymupdf.Rect(col_left_x0, 282, col_left_x1, 300), '<div class="h1">3 Experimental Results</div>', css=css)
    p2.insert_htmlbox(
        pymupdf.Rect(col_left_x0, 302, col_left_x1, 462),
        '<div class="p">We evaluate SparseRouter-VL on long-document language modeling and scientific visual question answering. '
        "As shown in Figure 2, both Dense Attention and Sliding Window baselines suffer sharp perplexity degradation past 32K tokens, whereas SparseRouter-VL stays nearly flat up to 128K tokens while reaching 144 tokens per second&mdash;a 3.4&times; speedup over dense attention.</div>",
        css=css,
    )
    p2.insert_htmlbox(
        pymupdf.Rect(col_left_x0, 466, col_left_x1, 608),
        '<div class="p">Next, we examine sensitivity to the active routing ratio &alpha; in Figure 3. '
        "Increasing &alpha; from 0.10 to 0.25 yields substantial accuracy gains on both ScienceQA and DocVQA, reaching 89.4% and 87.8% respectively, while values above &alpha; = 0.25 show diminishing returns of less than 0.8%.</div>",
        css=css,
    )
    p2.insert_htmlbox(pymupdf.Rect(col_left_x0, 616, col_left_x1, 634), '<div class="h1">4 Conclusion</div>', css=css)
    p2.insert_htmlbox(
        pymupdf.Rect(col_left_x0, 636, col_left_x1, 742),
        '<div class="p">SparseRouter-VL demonstrates that selective cross-modal routing enables fast, fully private research paper understanding and figure analysis directly on everyday laptops without cloud GPUs.</div>',
        css=css,
    )

    # Page 2 Right Column: Figure 3 + References
    p2.insert_image(pymupdf.Rect(col_right_x0, 282, col_right_x1, 412), stream=_make_fig3_ablation_png())
    p2.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 416, col_right_x1, 472),
        '<div class="caption"><b>Figure 3:</b> Ablation of active token routing ratio &alpha; from 0.10 to 1.00 on ScienceQA and DocVQA accuracy, highlighting the optimal operating point at &alpha; = 0.25.</div>',
        css=css,
    )
    p2.insert_htmlbox(pymupdf.Rect(col_right_x0, 486, col_right_x1, 504), '<div class="h1">References</div>', css=css)
    p2.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 506, col_right_x1, 558),
        '<div class="ref">[1] A. Vaswani et al. Attention is all you need. NeurIPS, 2017.</div>',
        css=css,
    )
    p2.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 560, col_right_x1, 616),
        '<div class="ref">[2] T. Dao et al. FlashAttention-2: Faster attention with better parallelism. ICLR, 2024.</div>',
        css=css,
    )
    p2.insert_htmlbox(
        pymupdf.Rect(col_right_x0, 618, col_right_x1, 686),
        '<div class="ref">[4] J. Bai et al. Qwen-VL: A versatile vision-language model for understanding and localization. arXiv:2308.12966, 2023.</div>',
        css=css,
    )

    doc.subset_fonts()
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    print(f"Generated sample academic paper: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_sample_pdf(root / "samples" / "sample_paper.pdf")
