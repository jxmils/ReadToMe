const SAMPLE_EXCERPT = `# Scaling Laws for Sparse Multimodal Routing

## Abstract
Vision-language transformers typically process every image tile with dense quadratic self-attention [1, 3, 8]. We show that routing only α = 0.25 of visual patches into global cross-attention cuts memory bandwidth by 3.2× on unified memory hardware while keeping 99.3% accuracy (±0.4%) on chart and document tasks (Vance et al., 2026).

## 1 Key Findings
Across 224 × 224 plot benchmarks, sparse routing keeps axis labels, legends, and curve changes while skipping blank background tiles in under 150 μs.`;

const state = {
  doc: null,
  eventSource: null,
  voices: [],
  selectedVoice: localStorage.getItem("readtome_voice") || "af_heart",
  speed: parseFloat(localStorage.getItem("readtome_speed") || "1.0"),
  explainFigures: true,
  distinctFigureVoice: true,
  readReferences: false,

  isPlaying: false,
  currentIndex: 0,
  playbackToken: 0,
  currentAudio: null,
  audioBlobCache: new Map(),
};

const el = {
  brandBtn: document.getElementById("brandBtn"),
  newDocBtn: document.getElementById("newDocBtn"),
  localStatusText: document.getElementById("localStatusText"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  themeIconSun: document.getElementById("themeIconSun"),
  themeIconMoon: document.getElementById("themeIconMoon"),

  ingestView: document.getElementById("ingestView"),
  pdfDropzone: document.getElementById("pdfDropzone"),
  pdfFileInput: document.getElementById("pdfFileInput"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  fillSampleTextBtn: document.getElementById("fillSampleTextBtn"),
  pasteTitleInput: document.getElementById("pasteTitleInput"),
  pasteTextInput: document.getElementById("pasteTextInput"),
  submitTextBtn: document.getElementById("submitTextBtn"),
  stripCitationsCheck: document.getElementById("stripCitationsCheck"),
  ingestBusyBanner: document.getElementById("ingestBusyBanner"),
  ingestBusyText: document.getElementById("ingestBusyText"),
  recentSection: document.getElementById("recentSection"),
  recentList: document.getElementById("recentList"),

  readerView: document.getElementById("readerView"),
  docSourceBadge: document.getElementById("docSourceBadge"),
  sidebarDocTitle: document.getElementById("sidebarDocTitle"),
  sidebarDocStats: document.getElementById("sidebarDocStats"),
  explainFiguresToggle: document.getElementById("explainFiguresToggle"),
  distinctFigureVoiceToggle: document.getElementById("distinctFigureVoiceToggle"),
  readReferencesToggle: document.getElementById("readReferencesToggle"),
  sidebarFiguresGroup: document.getElementById("sidebarFiguresGroup"),
  figureReadyCounter: document.getElementById("figureReadyCounter"),
  sidebarFiguresList: document.getElementById("sidebarFiguresList"),
  sidebarOutlineList: document.getElementById("sidebarOutlineList"),
  segmentsContainer: document.getElementById("segmentsContainer"),

  playerDock: document.getElementById("playerDock"),
  prevSegBtn: document.getElementById("prevSegBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  playIcon: document.getElementById("playIcon"),
  pauseIcon: document.getElementById("pauseIcon"),
  nextSegBtn: document.getElementById("nextSegBtn"),
  nowPlayingBadge: document.getElementById("nowPlayingBadge"),
  nowPlayingLabel: document.getElementById("nowPlayingLabel"),
  nowPlayingProgress: document.getElementById("nowPlayingProgress"),
  docProgressFill: document.getElementById("docProgressFill"),
  voiceSelect: document.getElementById("voiceSelect"),
  speedSelect: document.getElementById("speedSelect"),

  figureLightbox: document.getElementById("figureLightbox"),
  closeLightboxBtn: document.getElementById("closeLightboxBtn"),
  lightboxImg: document.getElementById("lightboxImg"),
  lightboxCaption: document.getElementById("lightboxCaption"),
};

function initTheme() {
  const saved = localStorage.getItem("readtome_theme") || "light";
  setTheme(saved);
  el.themeToggleBtn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    setTheme(cur === "dark" ? "light" : "dark");
  });
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("readtome_theme", theme);
  el.themeIconSun.classList.toggle("hidden", theme !== "dark");
  el.themeIconMoon.classList.toggle("hidden", theme === "dark");
}

async function fetchStatusAndRecent() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    el.localStatusText.textContent = `Local: ${data.tts_engine} + ${data.vlm_engine}`;
    state.voices = data.voices || [];
    populateVoiceSelector();
  } catch {
    el.localStatusText.textContent = "Server offline";
  }

  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    renderRecentDocuments(data.documents || []);
  } catch {
    // ignore
  }
}

function populateVoiceSelector() {
  el.voiceSelect.innerHTML = "";
  for (const v of state.voices) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = `${v.name} (${v.accent.replace(" English", "")})`;
    opt.title = v.style;
    if (v.id === state.selectedVoice) opt.selected = true;
    el.voiceSelect.appendChild(opt);
  }
  el.speedSelect.value = String(state.speed);
  if (el.speedSelect.selectedIndex === -1) {
    el.speedSelect.value = "1";
    state.speed = 1.0;
  }
}

function getFigureVoice() {
  if (!state.distinctFigureVoice) return state.selectedVoice;
  if (state.selectedVoice === "af_heart") return "bm_george";
  if (state.selectedVoice.startsWith("af_")) return "am_michael";
  if (state.selectedVoice.startsWith("am_")) return "bf_emma";
  if (state.selectedVoice.startsWith("bf_")) return "am_michael";
  return "af_heart";
}

function renderRecentDocuments(docs) {
  if (!docs.length) {
    el.recentSection.classList.add("hidden");
    return;
  }
  el.recentSection.classList.remove("hidden");
  el.recentList.innerHTML = "";
  for (const d of docs) {
    const btn = document.createElement("button");
    btn.className = "recent-item";
    const figTxt = d.figure_count ? `, ${d.figure_count} fig${d.figure_count === 1 ? "" : "s"}` : "";
    btn.innerHTML = `
      <div class="recent-item-title">${escapeHtml(d.title)}</div>
      <div class="recent-item-meta">${d.source_type.toUpperCase()} (${d.page_count}p${figTxt}, ~${d.estimated_minutes} min)</div>
    `;
    btn.addEventListener("click", () => openExistingDocument(d.doc_id));
    el.recentList.appendChild(btn);
  }
}

function setIngestBusy(busy, msg = "") {
  el.ingestBusyBanner.classList.toggle("hidden", !busy);
  if (msg) el.ingestBusyText.textContent = msg;
  el.loadSampleBtn.disabled = busy;
  el.submitTextBtn.disabled = busy;
}

async function uploadPdfFile(file) {
  if (!file) return;
  setIngestBusy(true, `Reading "${file.name}" and pulling out figures...`);
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("strip_citations", String(el.stripCitationsCheck.checked));
    const res = await fetch("/api/documents/pdf", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Could not read PDF");
    }
    const doc = await res.json();
    loadDocument(doc);
  } catch (err) {
    alert(err.message);
  } finally {
    setIngestBusy(false);
  }
}

async function loadSamplePaper() {
  setIngestBusy(true, "Loading sample paper...");
  try {
    const res = await fetch("/api/documents/sample", { method: "POST" });
    if (!res.ok) throw new Error("Could not load sample paper");
    const doc = await res.json();
    loadDocument(doc);
  } catch (err) {
    alert(err.message);
  } finally {
    setIngestBusy(false);
  }
}

async function submitPastedText() {
  const text = el.pasteTextInput.value.trim();
  if (!text) {
    el.pasteTextInput.focus();
    return;
  }
  setIngestBusy(true, "Preparing text...");
  try {
    const res = await fetch("/api/documents/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: el.pasteTitleInput.value.trim() || null,
        text,
        strip_citations: el.stripCitationsCheck.checked,
      }),
    });
    if (!res.ok) throw new Error("Failed to load text");
    const doc = await res.json();
    loadDocument(doc);
  } catch (err) {
    alert(err.message);
  } finally {
    setIngestBusy(false);
  }
}

async function openExistingDocument(docId) {
  setIngestBusy(true, "Opening document...");
  try {
    const res = await fetch(`/api/documents/${docId}`);
    if (!res.ok) throw new Error("Document not found");
    const doc = await res.json();
    loadDocument(doc);
  } catch (err) {
    alert(err.message);
  } finally {
    setIngestBusy(false);
  }
}

function showIngestScreen() {
  stopPlayback();
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  el.readerView.classList.add("hidden");
  el.playerDock.classList.add("hidden");
  el.newDocBtn.classList.add("hidden");
  el.ingestView.classList.remove("hidden");
  fetchStatusAndRecent();
}

function loadDocument(doc) {
  stopPlayback();
  state.doc = doc;
  state.currentIndex = 0;

  el.ingestView.classList.add("hidden");
  el.readerView.classList.remove("hidden");
  el.playerDock.classList.remove("hidden");
  el.newDocBtn.classList.remove("hidden");

  el.docSourceBadge.textContent = doc.source_type === "pdf" ? "PDF" : "TEXT";
  el.sidebarDocTitle.textContent = doc.title;
  const figStat = doc.figure_count ? `, ${doc.figure_count} figure${doc.figure_count === 1 ? "" : "s"}` : "";
  el.sidebarDocStats.textContent = `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}${figStat} (~${doc.estimated_minutes} min)`;

  renderOutline();
  renderSidebarFigures();
  renderSegmentsStream();
  updateNowPlayingUI();

  prefetchUpcomingFrom(0);

  if (state.eventSource) state.eventSource.close();
  if (doc.figure_count > 0) {
    connectFigureSSE(doc.doc_id);
  }
}

function connectFigureSSE(docId) {
  const es = new EventSource(`/api/documents/${docId}/events`);
  state.eventSource = es;
  es.onmessage = (evt) => {
    if (!state.doc || state.doc.doc_id !== docId) return;
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "segment_update" && msg.segment) {
        const updated = msg.segment;
        const idx = state.doc.segments.findIndex((s) => s.id === updated.id);
        if (idx !== -1) {
          state.doc.segments[idx] = updated;
          updateFigureCardDOM(updated);
          renderSidebarFigures();
          if (updated.figure_status === "ready" && Math.abs(idx - state.currentIndex) <= 3) {
            prefetchSegmentAudio(updated);
          }
        }
      }
    } catch {
      // ignore keepalive
    }
  };
}

function renderOutline() {
  el.sidebarOutlineList.innerHTML = "";
  if (!state.doc) return;
  const headings = state.doc.segments.filter((s) => s.kind === "title" || s.kind === "heading");
  for (const h of headings) {
    const btn = document.createElement("button");
    btn.className = "outline-btn";
    btn.dataset.segIndex = String(h.index);
    btn.textContent = h.display_text;
    btn.addEventListener("click", () => jumpToSegment(h.index, true));
    el.sidebarOutlineList.appendChild(btn);
  }
}

function renderSidebarFigures() {
  if (!state.doc) return;
  const figs = state.doc.segments.filter((s) => s.kind === "figure");
  if (!figs.length) {
    el.sidebarFiguresGroup.classList.add("hidden");
    return;
  }
  el.sidebarFiguresGroup.classList.remove("hidden");
  const readyCount = figs.filter((f) => f.figure_status === "ready").length;
  el.figureReadyCounter.textContent = `${readyCount} / ${figs.length}`;

  el.sidebarFiguresList.innerHTML = "";
  for (const fig of figs) {
    const card = document.createElement("button");
    card.className = "sidebar-fig-card" + (state.currentIndex === fig.index ? " active" : "");
    let statusTxt = "Waiting...";
    let statusCls = "";
    if (fig.figure_status === "analyzing") statusTxt = "Reading figure...";
    else if (fig.figure_status === "ready") {
      statusTxt = "Ready";
      statusCls = "ready";
    } else if (fig.figure_status === "error") statusTxt = "Caption only";

    card.innerHTML = `
      <img class="sidebar-fig-thumb" src="${fig.image_url}" alt="${escapeHtml(fig.figure_label || "Figure")}" />
      <div class="sidebar-fig-info">
        <div class="sidebar-fig-label">${escapeHtml(fig.figure_label || "Figure")} ${fig.page ? `<span style="font-weight:400;opacity:0.7">(p.${fig.page})</span>` : ""}</div>
        <div class="sidebar-fig-status ${statusCls}">${statusTxt}</div>
      </div>
    `;
    card.addEventListener("click", () => jumpToSegment(fig.index, true));
    el.sidebarFiguresList.appendChild(card);
  }
}

function renderSegmentsStream() {
  el.segmentsContainer.innerHTML = "";
  if (!state.doc) return;

  for (const seg of state.doc.segments) {
    if (seg.kind === "figure") {
      el.segmentsContainer.appendChild(createFigureCardElement(seg));
      continue;
    }

    const div = document.createElement("div");
    const shouldDim = isSegmentSkipped(seg);
    div.className = `seg-block seg-${seg.kind}${shouldDim ? " skipped" : ""}`;
    div.id = `dom_${seg.id}`;
    div.dataset.index = String(seg.index);

    const pageChip = seg.page && seg.kind === "heading" ? `<span class="seg-page-chip">p.${seg.page}</span>` : "";
    div.innerHTML = `${pageChip}${escapeHtml(seg.display_text)}`;
    div.addEventListener("click", () => jumpToSegment(seg.index, true));
    el.segmentsContainer.appendChild(div);
  }
}

function createFigureCardElement(seg) {
  const wrap = document.createElement("article");
  wrap.className = "figure-card";
  wrap.id = `dom_${seg.id}`;
  wrap.dataset.index = String(seg.index);

  wrap.innerHTML = buildFigureCardInnerHTML(seg);
  bindFigureCardEvents(wrap, seg);
  return wrap;
}

function buildFigureCardInnerHTML(seg) {
  const isReady = seg.figure_status === "ready";
  const statusLabel = isReady
    ? `${seg.vlm_model || "Local model"}${seg.generation_ms ? ` (${(seg.generation_ms / 1000).toFixed(1)}s)` : ""}`
    : seg.figure_status === "analyzing"
      ? "Looking at chart details..."
      : "Queued...";

  const bodyCls = isReady || seg.figure_status === "error" ? "figure-narration-text" : "figure-narration-text analyzing-pulse";

  return `
    <div class="figure-card-header" data-role="play-figure">
      <div class="figure-header-left">
        <span class="figure-badge">${escapeHtml(seg.figure_label || "Figure")}${seg.page ? ` (Page ${seg.page})` : ""}</span>
        <span class="figure-status-tag">${escapeHtml(statusLabel)}</span>
      </div>
      <button class="figure-play-chip" type="button" data-role="play-btn">Play figure</button>
    </div>
    <div class="figure-image-wrap" data-role="zoom-image" title="Click to enlarge">
      <img src="${seg.image_url}" alt="${escapeHtml(seg.figure_label || "Figure")}" loading="lazy" />
    </div>
    <div class="figure-narration-body">
      <div class="figure-narration-label">What this figure shows</div>
      <div class="${bodyCls}" data-role="narration-text">${escapeHtml(seg.display_text)}</div>
      ${seg.caption ? `<div class="figure-original-caption"><strong>Original caption:</strong> ${escapeHtml(seg.caption)}</div>` : ""}
    </div>
    <form class="figure-ask-bar" data-role="ask-form">
      <input
        class="figure-ask-input"
        type="text"
        placeholder="Ask a question about ${escapeHtml(seg.figure_label || "this figure")}..."
      />
      <button class="btn btn-secondary" type="submit">Ask</button>
    </form>
    <div class="figure-qa-answer hidden" data-role="qa-answer"></div>
  `;
}

function bindFigureCardEvents(wrap, seg) {
  const playTrigger = () => jumpToSegment(seg.index, true);
  wrap.querySelector('[data-role="play-btn"]')?.addEventListener("click", (e) => {
    e.stopPropagation();
    playTrigger();
  });
  wrap.querySelector('[data-role="narration-text"]')?.addEventListener("click", playTrigger);

  wrap.querySelector('[data-role="zoom-image"]')?.addEventListener("click", () => {
    el.lightboxImg.src = seg.image_url;
    el.lightboxCaption.textContent = seg.caption || seg.display_text;
    el.figureLightbox.classList.remove("hidden");
  });

  const form = wrap.querySelector('[data-role="ask-form"]');
  const input = form?.querySelector("input");
  const answerBox = wrap.querySelector('[data-role="qa-answer"]');
  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || !state.doc) return;
    stopPlayback();
    answerBox.classList.remove("hidden");
    answerBox.textContent = "Checking the figure...";
    input.disabled = true;
    try {
      const res = await fetch(`/api/documents/${state.doc.doc_id}/segments/${seg.id}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ detail_level: "detailed", custom_question: q }),
      });
      if (!res.ok) throw new Error("Could not answer question");
      const data = await res.json();
      answerBox.innerHTML = `<strong>Answer:</strong> ${escapeHtml(data.display_text)}`;
      await speakAdhocText(data.spoken_text, getFigureVoice(), `${seg.figure_label} answer`);
    } catch (err) {
      answerBox.textContent = `Error: ${err.message}`;
    } finally {
      input.disabled = false;
    }
  });
}

function updateFigureCardDOM(seg) {
  const existing = document.getElementById(`dom_${seg.id}`);
  if (!existing) return;
  const wasActive = existing.classList.contains("active");
  existing.innerHTML = buildFigureCardInnerHTML(seg);
  if (wasActive) existing.classList.add("active");
  bindFigureCardEvents(existing, seg);
}

function isSegmentSkipped(seg) {
  if (seg.kind === "figure" && !state.explainFigures) return true;
  if ((seg.kind === "reference" || seg.skip_by_default) && !state.readReferences) return true;
  return false;
}

// Split off the first sentence on longer paragraphs so audio starts quickly while the rest generates
function splitForFastStart(spokenText) {
  const trimmed = (spokenText || "").trim();
  if (trimmed.length <= 210) return [trimmed];
  const match = trimmed.match(/^(.{45,220}?[.!?])\s+(.+)$/s);
  if (match) {
    return [match[1].trim(), match[2].trim()];
  }
  return [trimmed];
}

function voiceForSegment(seg) {
  return seg.kind === "figure" ? getFigureVoice() : state.selectedVoice;
}

function fetchChunkAudioUrl(text, voice) {
  const key = `${voice}|${text}`;
  if (state.audioBlobCache.has(key)) {
    return state.audioBlobCache.get(key);
  }
  const promise = fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed: 1.0 }),
  })
    .then((res) => {
      if (!res.ok) throw new Error("TTS request failed");
      return res.blob();
    })
    .then((blob) => URL.createObjectURL(blob))
    .catch((err) => {
      state.audioBlobCache.delete(key);
      throw err;
    });

  state.audioBlobCache.set(key, promise);
  return promise;
}

function prefetchSegmentAudio(seg) {
  if (!seg || isSegmentSkipped(seg)) return;
  if (seg.kind === "figure" && seg.figure_status !== "ready" && seg.figure_status !== "error") {
    return;
  }
  const voice = voiceForSegment(seg);
  const parts = splitForFastStart(seg.spoken_text);
  for (const p of parts) {
    if (p) fetchChunkAudioUrl(p, voice).catch(() => {});
  }
}

function prefetchUpcomingFrom(startIdx) {
  if (!state.doc) return;
  let count = 0;
  for (let i = startIdx; i < state.doc.segments.length && count < 2; i++) {
    const seg = state.doc.segments[i];
    if (!isSegmentSkipped(seg)) {
      prefetchSegmentAudio(seg);
      count++;
    }
  }
}

function playSingleAudioUrl(url, token) {
  return new Promise((resolve, reject) => {
    if (token !== state.playbackToken) return resolve(false);
    const audio = new Audio(url);
    audio.preservesPitch = true;
    audio.playbackRate = state.speed;
    state.currentAudio = audio;

    audio.onended = () => {
      if (state.currentAudio === audio) state.currentAudio = null;
      resolve(true);
    };
    audio.onerror = (err) => {
      if (state.currentAudio === audio) state.currentAudio = null;
      reject(err);
    };
    audio.play().catch(reject);
  });
}

async function speakAdhocText(spokenText, voice, badgeLabel) {
  const token = ++state.playbackToken;
  state.isPlaying = true;
  syncPlayPauseButton();
  el.nowPlayingBadge.textContent = "ANSWER";
  el.nowPlayingBadge.classList.add("figure-mode");
  el.nowPlayingLabel.textContent = badgeLabel;

  const parts = splitForFastStart(spokenText);
  const urls = parts.map((p) => fetchChunkAudioUrl(p, voice));
  for (const urlPromise of urls) {
    const url = await urlPromise;
    if (token !== state.playbackToken || !state.isPlaying) return;
    await playSingleAudioUrl(url, token);
  }
  if (token === state.playbackToken) {
    state.isPlaying = false;
    syncPlayPauseButton();
    updateNowPlayingUI();
  }
}

function stopPlayback() {
  state.playbackToken++;
  state.isPlaying = false;
  if (state.currentAudio) {
    state.currentAudio.pause();
    state.currentAudio = null;
  }
  syncPlayPauseButton();
}

function syncPlayPauseButton() {
  el.playIcon.classList.toggle("hidden", state.isPlaying);
  el.pauseIcon.classList.toggle("hidden", !state.isPlaying);
}

function highlightActiveSegment() {
  document.querySelectorAll(".seg-block.active, .figure-card.active").forEach((n) => n.classList.remove("active"));
  if (!state.doc) return;
  const seg = state.doc.segments[state.currentIndex];
  if (!seg) return;

  const domEl = document.getElementById(`dom_${seg.id}`);
  if (domEl) {
    domEl.classList.add("active");
    domEl.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  document.querySelectorAll(".outline-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent === seg.section || Number(btn.dataset.segIndex) === seg.index);
  });
  renderSidebarFigures();
  updateNowPlayingUI();
}

function updateNowPlayingUI() {
  if (!state.doc || !state.doc.segments.length) return;
  const total = state.doc.segments.length;
  const idx = Math.min(state.currentIndex, total - 1);
  const seg = state.doc.segments[idx];

  if (seg.kind === "figure") {
    el.nowPlayingBadge.textContent = "FIGURE";
    el.nowPlayingBadge.classList.add("figure-mode");
    el.nowPlayingLabel.textContent = `${seg.figure_label}: ${seg.display_text}`;
  } else {
    el.nowPlayingBadge.textContent = (seg.section || seg.kind).toUpperCase().slice(0, 22);
    el.nowPlayingBadge.classList.remove("figure-mode");
    el.nowPlayingLabel.textContent = seg.display_text;
  }

  el.nowPlayingProgress.textContent = `${idx + 1} / ${total}`;
  const pct = total > 1 ? ((idx + 1) / total) * 100 : 100;
  el.docProgressFill.style.width = `${pct.toFixed(1)}%`;
}

async function runPlaybackLoop(startIdx) {
  if (!state.doc) return;
  const token = ++state.playbackToken;
  state.isPlaying = true;
  state.currentIndex = startIdx;
  syncPlayPauseButton();

  while (state.isPlaying && token === state.playbackToken && state.currentIndex < state.doc.segments.length) {
    let seg = state.doc.segments[state.currentIndex];
    if (isSegmentSkipped(seg)) {
      state.currentIndex++;
      continue;
    }

    highlightActiveSegment();

    if (seg.kind === "figure" && seg.figure_status !== "ready" && seg.figure_status !== "error") {
      el.nowPlayingLabel.textContent = `Reading ${seg.figure_label}...`;
      try {
        const res = await fetch(`/api/documents/${state.doc.doc_id}/segments/${seg.id}/prioritize`, {
          method: "POST",
        });
        if (res.ok) {
          seg = await res.json();
          state.doc.segments[state.currentIndex] = seg;
          updateFigureCardDOM(seg);
          updateNowPlayingUI();
        }
      } catch {
        // fall back to caption
      }
    }

    if (token !== state.playbackToken || !state.isPlaying) return;

    prefetchUpcomingFrom(state.currentIndex + 1);

    const voice = voiceForSegment(seg);
    const parts = splitForFastStart(seg.spoken_text).filter(Boolean);
    const partPromises = parts.map((p) => fetchChunkAudioUrl(p, voice));

    try {
      for (const pUrlPromise of partPromises) {
        const url = await pUrlPromise;
        if (token !== state.playbackToken || !state.isPlaying) return;
        await playSingleAudioUrl(url, token);
      }
    } catch (err) {
      console.error("Playback error on segment", seg.id, err);
    }

    if (token !== state.playbackToken || !state.isPlaying) return;
    state.currentIndex++;
  }

  if (token === state.playbackToken) {
    state.isPlaying = false;
    if (state.doc && state.currentIndex >= state.doc.segments.length) {
      state.currentIndex = 0;
    }
    syncPlayPauseButton();
  }
}

function togglePlayPause() {
  if (!state.doc) return;
  if (state.isPlaying) {
    stopPlayback();
  } else {
    runPlaybackLoop(state.currentIndex);
  }
}

function jumpToSegment(index, autoPlay = true) {
  if (!state.doc) return;
  const bounded = Math.max(0, Math.min(index, state.doc.segments.length - 1));
  stopPlayback();
  state.currentIndex = bounded;
  highlightActiveSegment();
  if (autoPlay) {
    runPlaybackLoop(bounded);
  }
}

function stepSegment(delta) {
  if (!state.doc) return;
  let next = state.currentIndex + delta;
  while (next >= 0 && next < state.doc.segments.length && isSegmentSkipped(state.doc.segments[next])) {
    next += delta > 0 ? 1 : -1;
  }
  next = Math.max(0, Math.min(next, state.doc.segments.length - 1));
  jumpToSegment(next, state.isPlaying);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function bindEvents() {
  el.brandBtn.addEventListener("click", showIngestScreen);
  el.newDocBtn.addEventListener("click", showIngestScreen);

  el.pdfDropzone.addEventListener("click", () => el.pdfFileInput.click());
  el.pdfFileInput.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) uploadPdfFile(f);
    e.target.value = "";
  });
  el.pdfDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.pdfDropzone.classList.add("drag-over");
  });
  el.pdfDropzone.addEventListener("dragleave", () => el.pdfDropzone.classList.remove("drag-over"));
  el.pdfDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    el.pdfDropzone.classList.remove("drag-over");
    const f = e.dataTransfer?.files?.[0];
    if (f) uploadPdfFile(f);
  });

  el.loadSampleBtn.addEventListener("click", loadSamplePaper);
  el.fillSampleTextBtn.addEventListener("click", () => {
    el.pasteTitleInput.value = "Scaling Laws for Sparse Multimodal Routing";
    el.pasteTextInput.value = SAMPLE_EXCERPT;
  });
  el.submitTextBtn.addEventListener("click", submitPastedText);

  el.explainFiguresToggle.addEventListener("change", (e) => {
    state.explainFigures = e.target.checked;
  });
  el.distinctFigureVoiceToggle.addEventListener("change", (e) => {
    state.distinctFigureVoice = e.target.checked;
  });
  el.readReferencesToggle.addEventListener("change", (e) => {
    state.readReferences = e.target.checked;
    renderSegmentsStream();
    highlightActiveSegment();
  });

  el.playPauseBtn.addEventListener("click", togglePlayPause);
  el.prevSegBtn.addEventListener("click", () => stepSegment(-1));
  el.nextSegBtn.addEventListener("click", () => stepSegment(1));

  el.voiceSelect.addEventListener("change", (e) => {
    state.selectedVoice = e.target.value;
    localStorage.setItem("readtome_voice", state.selectedVoice);
    if (state.isPlaying) {
      jumpToSegment(state.currentIndex, true);
    } else {
      prefetchUpcomingFrom(state.currentIndex);
    }
  });

  el.speedSelect.addEventListener("change", (e) => {
    state.speed = parseFloat(e.target.value) || 1.0;
    localStorage.setItem("readtome_speed", String(state.speed));
    if (state.currentAudio) {
      state.currentAudio.playbackRate = state.speed;
    }
  });

  el.closeLightboxBtn.addEventListener("click", () => el.figureLightbox.classList.add("hidden"));
  el.figureLightbox.addEventListener("click", (e) => {
    if (e.target === el.figureLightbox) el.figureLightbox.classList.add("hidden");
  });

  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.key === "Escape") {
      el.figureLightbox.classList.add("hidden");
      return;
    }
    if (!state.doc || el.readerView.classList.contains("hidden")) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      stepSegment(1);
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      stepSegment(-1);
    }
  });
}

initTheme();
bindEvents();
fetchStatusAndRecent();
