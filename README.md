# ReadToMe

A local tool that reads PDFs and text out loud, mainly built for research papers.

Most PDF readers mess up two-column papers, read out every citation bracket, and either skip over figures or read random axis numbers as gibberish. ReadToMe fixes the reading order, crops out each figure as an image, and uses a small local vision model to tell you what the chart or diagram actually shows before moving on with the paper.

Everything runs locally on your own machine (Kokoro for speech and Qwen3-VL for figures). Nothing gets sent to external APIs.

## Setup and Running

### Mac (Apple Silicon) and Linux

You'll need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jxmils/ReadToMe.git
cd ReadToMe
./run.sh
```

Then open `http://127.0.0.1:8000` in your browser.

### Windows

Double-click `run.bat`, or run it from Command Prompt / PowerShell:

```bat
git clone https://github.com/jxmils/ReadToMe.git
cd ReadToMe
run.bat
```

Or in PowerShell:

```powershell
.\run.ps1
```

On the first run, the script sets up `uv` and Python 3.12 if you don't already have them, checks for an NVIDIA GPU so it can grab the CUDA build of PyTorch, and opens `http://127.0.0.1:8000`. If you don't have an NVIDIA GPU, it falls back to CPU (or uses local Ollama automatically if you have Ollama running with a vision model like `qwen3-vl:2b`).

The first time you run it, it downloads the voice files (~340 MB into `models/kokoro/`) and the vision model (~1.8 GB on Mac MLX, ~4 GB on Windows/Linux into your Hugging Face cache). After that it works offline.

## What it does

- Reads two-column academic PDFs in the right order while ignoring headers, footers, and page numbers.
- Pulls out both normal images and vector plots/diagrams, then explains what they show in plain English right after the paragraph that mentions them.
- Cleans up inline citations like `[1, 4, 12]` and reads common math symbols (`α`, `3.4×`, `±`, `μs`) naturally.
- Lets you type a follow-up question on any figure card and hear the answer read back.
- Skips the bibliography by default unless you turn it on in the sidebar.
- Includes a built-in sample paper on the start screen if you just want to test it out quickly.

## Optional environment variables

- `READTOME_VLM_MODEL`: Override the Hugging Face model name used for figure descriptions.
- `READTOME_VLM_BASE_URL`: Point to a local OpenAI-compatible server (like Ollama at `http://127.0.0.1:11434/v1` or LM Studio at `http://127.0.0.1:1234/v1`) instead of loading the vision model directly in Python.
- `READTOME_VLM_BACKEND`: Set to `mlx` or `transformers` if you want to force a specific backend.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
