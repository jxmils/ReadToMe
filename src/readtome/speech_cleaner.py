from __future__ import annotations

import re

_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00ad": "",
        "\u2212": "-",
        "\u2010": "-",
        "\u2011": "-",
    }
)

_KEEP_HYPHEN_PREFIXES = {
    "self",
    "multi",
    "cross",
    "pre",
    "post",
    "non",
    "sub",
    "inter",
    "intra",
    "semi",
    "anti",
    "high",
    "low",
    "long",
    "short",
    "two",
    "three",
    "four",
    "zero",
    "few",
    "one",
    "real",
    "open",
    "closed",
    "fine",
    "large",
    "small",
    "co",
    "re",
    "over",
    "under",
    "full",
    "half",
    "end",
    "state",
    "vision",
    "audio",
    "text",
    "data",
    "model",
    "single",
    "double",
    "well",
    "ill",
    "domain",
    "layer",
    "peer",
    "time",
    "all",
    "on",
    "off",
    "in",
    "up",
    "back",
    "run",
    "hand",
    "hard",
    "soft",
    "best",
    "world",
    "first",
}

_GREEK_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ϵ": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "ϕ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Δ": "Delta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Σ": "Sigma",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
}

_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _dehyphenate_match(match: re.Match[str]) -> str:
    prefix = match.group(1)
    suffix = match.group(2)
    if prefix.lower() in _KEEP_HYPHEN_PREFIXES or (prefix and prefix[0].isupper() and suffix and suffix[0].isupper()):
        return f"{prefix}-{suffix}"
    return f"{prefix}{suffix}"


def clean_display_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.translate(_LIGATURES)
    # Strip private-use font bullet glyphs common in magazine/ACM PDFs
    text = re.sub(r"[\ue000-\uf8ff˲˽]+", "• ", text)
    # Fix InDesign / custom Type1 soft-hyphen replacement chars (e.g., "Pro\ufffdcessing")
    text = re.sub(r"([A-Za-z]{2,})\ufffd\s*\n?\s*([A-Za-z]{2,})", _dehyphenate_match, text)
    text = text.replace("\ufffd", "")
    # Join words split across lines with a hyphen
    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,})", _dehyphenate_match, text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def prepare_spoken_text(display_text: str, *, strip_citations: bool = True) -> str:
    if not display_text:
        return ""

    text = display_text

    # Remove bullet symbols so TTS reads bulleted lines cleanly
    text = re.sub(r"[•▸►▪▫◦‣»]+", " ", text)

    # Drop bare parenthetical links and shorten inline URLs
    text = re.sub(r"\(\s*https?://[^\s)]+\s*\)", "", text)
    text = re.sub(r"https?://\S+", "see link", text)

    if strip_citations:
        # Bracketed numeric citations: [1], [2, 5, 11], [3-7]
        text = re.sub(r"\[\s*\d+(?:\s*[,;–\-]\s*\d+)*\s*\]", "", text)
        # Trailing superscript-style citations attached after punctuation: "Law.17", "inference.19,20", "(SGD).31"
        text = re.sub(r"([A-Za-z)\”\"'][.,;:])\d{1,3}(?:[,–\-]\d{1,3})*(?=\s|$|[)\]\"”'])", r"\1", text)
        # Multi-author parenthetical citations
        text = re.sub(
            r"\((?:[A-Z][A-Za-z'\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\-]+))?,?\s*\d{4}[a-z]?\s*[;,]\s*)+"
            r"[A-Z][A-Za-z'\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\-]+))?,?\s*\d{4}[a-z]?\)",
            "",
            text,
        )
        text = re.sub(
            r"\([A-Z][A-Za-z'\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\-]+))?,\s*\d{4}[a-z]?\)",
            "",
            text,
        )

    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*[μµ]s\b", r"\1 microseconds", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*[μµ]m\b", r"\1 micrometers", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*[μµ]L\b", r"\1 microliters", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*[μµ]M\b", r"\1 micromolar", text)

    # Read "224 × 224" or "128x128" as "by", and "3.4×" or "50x" as "times"
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\b", r"\1 by \2", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[×x]\b", r"\1 times", text)
    text = text.replace("×", " times ")

    def _replace_unicode_sup(m: re.Match[str]) -> str:
        base = m.group(1)
        exp = m.group(2).translate(_SUPERSCRIPT_DIGITS)
        if exp == "2":
            return f"{base} squared"
        if exp == "3":
            return f"{base} cubed"
        exp_spoken = exp.replace("-", "minus ").replace("+", "plus ")
        return f"{base} to the {exp_spoken}"

    text = re.sub(r"([A-Za-z0-9]+)([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", _replace_unicode_sup, text)
    text = re.sub(r"\b10\^(-?\d+)\b", lambda m: f"10 to the {m.group(1).replace('-', 'minus ')}", text)
    text = re.sub(r"\b([A-Za-z])\^2\b", r"\1 squared", text)

    abbrev_rules = [
        (r"\be\.g\.,?", "for example,"),
        (r"\bi\.e\.,?", "that is,"),
        (r"\bw\.r\.t\.?", "with respect to"),
        (r"\bc\.f\.?", "compare"),
        (r"\bvs\.", "versus"),
        (r"\bFigs\.\s*", "Figures "),
        (r"\bFig\.\s*", "Figure "),
        (r"\bEqs\.\s*", "Equations "),
        (r"\bEq\.\s*", "Equation "),
        (r"\bSecs\.\s*", "Sections "),
        (r"\bSec\.\s*", "Section "),
        (r"\bTabs\.\s*", "Tables "),
        (r"\bTab\.\s*", "Table "),
        (r"\bAlg\.\s*", "Algorithm "),
        (r"\bet\s+al\.", "et al"),
        (r"\bapprox\.", "approximately"),
    ]
    for pattern, repl in abbrev_rules:
        text = re.sub(pattern, repl, text)

    symbol_rules = [
        ("±", " plus or minus "),
        ("≈", " approximately "),
        ("∼", " around "),
        ("≠", " not equal to "),
        ("≤", " less than or equal to "),
        ("≥", " greater than or equal to "),
        ("→", " to "),
        ("←", " from "),
        ("↔", " between "),
        ("∞", " infinity "),
    ]
    for sym, spoken in symbol_rules:
        text = text.replace(sym, spoken)

    for greek_char, greek_name in _GREEK_MAP.items():
        if greek_char in text:
            text = re.sub(rf"(?<![A-Za-z]){re.escape(greek_char)}(?![A-Za-z])", f" {greek_name} ", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
