/**
 * app.js — Libras Wild: fila de segmentos pré-computados + anotação manual.
 *
 * Fluxo principal:
 *  1. Carrega config.json + segments.json
 *  2. Para cada segmento: carrega vídeo, salta para t_start/t_end automaticamente
 *  3. Usuário revisa o trecho (preview automático) e anota
 *  4. Ao salvar, avança para o próximo segmento não anotado
 *  5. Também suporta anotação manual via URL do YouTube
 */

// ── Estado ────────────────────────────────────────────────────
let annotator  = "";
let ytPlayer   = null;
let ytReady    = false;
let currentVideoId = null;
let tStart     = null;
let tEnd       = null;
let sessionAnns = [];

// Fila de segmentos pré-computados
let config      = null;
let segments    = [];
let currentSegIdx = 0;
let annotatedIds  = new Set();
let playerReady   = false;   // player pronto para seekTo

// ── YouTube IFrame API ────────────────────────────────────────
window.onYouTubeIframeAPIReady = function () {
  ytReady = true;
  ytPlayer = new YT.Player("yt-player", {
    height: "100%",
    width:  "100%",
    videoId: "",
    playerVars: { controls: 1, modestbranding: 1, rel: 0, iv_load_policy: 3 },
    events: {
      onReady: () => {
        playerReady = true;
        // Se segmentos já carregados, vai para o primeiro
        if (segments.length) loadSegment(currentSegIdx);
      },
      onStateChange: onPlayerStateChange,
    }
  });
};

// Para automaticamente ao atingir tEnd durante preview
function onPlayerStateChange(e) {
  if (e.data === YT.PlayerState.PLAYING && tEnd !== null) {
    const check = setInterval(() => {
      if (!ytPlayer) { clearInterval(check); return; }
      const t = ytPlayer.getCurrentTime();
      if (t >= tEnd) {
        ytPlayer.pauseVideo();
        clearInterval(check);
      }
    }, 200);
  }
}

// ── Fila de segmentos ─────────────────────────────────────────
async function initSegmentQueue() {
  try {
    [config, segments] = await Promise.all([
      fetch("data/config.json").then(r => r.json()),
      fetch("data/segments.json").then(r => r.json()),
    ]);
    // Marca já anotados
    const existing = await getAllAnnotations();
    existing.forEach(a => annotatedIds.add(a.seg_id));

    // Primeiro segmento não anotado
    let first = segments.findIndex(s => !annotatedIds.has(s.id));
    if (first < 0) first = 0;
    currentSegIdx = first;

    document.getElementById("seg-nav").style.display = "block";
    updateSegNavUI();

    if (playerReady) loadSegment(currentSegIdx);
  } catch (err) {
    console.warn("segments.json indisponível — modo manual.", err);
  }
}

function loadSegment(idx) {
  if (!segments.length || idx < 0 || idx >= segments.length) return;
  if (!playerReady) { currentSegIdx = idx; return; }

  const seg = segments[idx];
  currentSegIdx = idx;
  const vid = config.youtube_id;

  document.getElementById("video-placeholder").style.display = "none";
  document.getElementById("yt-url-input").value = `https://youtu.be/${vid}`;
  document.getElementById("time-controls").style.display = "block";

  if (currentVideoId !== vid) {
    currentVideoId = vid;
    ytPlayer.loadVideoById({ videoId: vid, startSeconds: seg.t_start });
    // onReady do vídeo dispara onPlayerStateChange; pausamos via seekTo após buffer
    setTimeout(() => { ytPlayer.pauseVideo(); setSegmentTimes(seg); }, 1500);
  } else {
    ytPlayer.seekTo(seg.t_start, true);
    ytPlayer.pauseVideo();
    setSegmentTimes(seg);
  }
  updateSegNavUI();
}

function setSegmentTimes(seg) {
  tStart = seg.t_start;
  tEnd   = seg.t_end;
  document.getElementById("t-start").value = formatTime(tStart);
  document.getElementById("t-end").value   = formatTime(tEnd);
  updateDuration();
}

function updateSegNavUI() {
  if (!segments.length) return;
  const seg = segments[currentSegIdx];
  document.getElementById("seg-counter").textContent =
    `Segmento ${currentSegIdx + 1} / ${segments.length}`;
  document.getElementById("seg-annotated").textContent =
    `${annotatedIds.size} anotados`;
  document.getElementById("seg-id-label").textContent = seg.id;
  document.getElementById("btn-seg-prev").disabled = currentSegIdx <= 0;
  document.getElementById("btn-seg-next").disabled = currentSegIdx >= segments.length - 1;
  // Destaca se já anotado
  const nav = document.getElementById("seg-nav");
  nav.classList.toggle("seg-done", annotatedIds.has(seg.id));
}

function segNext() {
  if (currentSegIdx < segments.length - 1) loadSegment(currentSegIdx + 1);
}
function segPrev() {
  if (currentSegIdx > 0) loadSegment(currentSegIdx - 1);
}

// ── Extrai ID do YouTube ──────────────────────────────────────
function extractYtId(url) {
  url = url.trim();
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([A-Za-z0-9_-]{11})/,
    /^([A-Za-z0-9_-]{11})$/,
  ];
  for (const re of patterns) {
    const m = url.match(re);
    if (m) return m[1];
  }
  return null;
}

// ── Carregar vídeo manualmente ────────────────────────────────
function loadVideo() {
  const raw = document.getElementById("yt-url-input").value;
  const vid = extractYtId(raw);
  if (!vid) { flashStatus("URL inválida.", "error"); return; }
  if (!playerReady) { flashStatus("Player carregando…", "error"); return; }
  currentVideoId = vid;
  document.getElementById("video-placeholder").style.display = "none";
  ytPlayer.loadVideoById(vid);
  clearTimes();
  document.getElementById("time-controls").style.display = "block";
  flashStatus("Vídeo carregado.", "ok");
}

// ── Marcação de tempos ────────────────────────────────────────
function markStart() {
  if (!ytPlayer) return;
  tStart = parseFloat(ytPlayer.getCurrentTime().toFixed(2));
  document.getElementById("t-start").value = formatTime(tStart);
  updateDuration();
}

function markEnd() {
  if (!ytPlayer) return;
  tEnd = parseFloat(ytPlayer.getCurrentTime().toFixed(2));
  document.getElementById("t-end").value = formatTime(tEnd);
  updateDuration();
}

function updateDuration() {
  const el = document.getElementById("time-dur");
  if (tStart !== null && tEnd !== null && tEnd > tStart) {
    const dur = (tEnd - tStart).toFixed(2);
    el.textContent = `${dur}s`;
    el.style.color = dur < 0.3 ? "var(--red)" : "var(--green)";
    checkFormReady();
  } else if (tEnd !== null && tStart !== null && tEnd <= tStart) {
    el.textContent = "⚠ fim antes do início";
    el.style.color = "var(--red)";
  } else {
    el.textContent = "—";
    el.style.color = "var(--dgray)";
  }
}

function clearTimes() {
  tStart = null; tEnd = null;
  document.getElementById("t-start").value = "";
  document.getElementById("t-end").value   = "";
  document.getElementById("time-dur").textContent = "—";
  document.getElementById("time-dur").style.color = "var(--dgray)";
  checkFormReady();
}

function previewSegment() {
  if (!ytPlayer || tStart === null || tEnd === null || tEnd <= tStart) {
    flashStatus("Marque início e fim primeiro.", "error"); return;
  }
  ytPlayer.seekTo(tStart, true);
  ytPlayer.playVideo();
}

// ── Validação ─────────────────────────────────────────────────
function checkFormReady() {
  const ok = currentVideoId && tStart !== null && tEnd !== null && tEnd > tStart;
  document.getElementById("btn-submit").disabled = !ok;
}

// ── Submissão ─────────────────────────────────────────────────
async function submitAnnotation() {
  if (!currentVideoId || tStart === null || tEnd === null || tEnd <= tStart) {
    flashStatus("Marque início e fim.", "error"); return;
  }

  const curSeg = segments[currentSegIdx];
  const segId  = curSeg ? curSeg.id : `${currentVideoId}_${Math.round(tStart * 10)}`;

  const ann = {
    seg_id:      segId,
    annotator,
    ts:          Date.now(),
    video_id:    currentVideoId,
    video_url:   `https://youtu.be/${currentVideoId}`,
    t_start:     tStart,
    t_end:       tEnd,
    duration:    parseFloat((tEnd - tStart).toFixed(2)),
    valid:       document.querySelector('[name="valid"]:checked')?.value,
    label:       document.getElementById("sign-label").value || null,
    outro_name:  document.getElementById("outro-name").value.trim() || null,
    confidence:  parseInt(document.querySelector('[name="confidence"]:checked')?.value),
    handshape:   document.getElementById("handshape").value.trim()   || null,
    location:    document.getElementById("location").value.trim()    || null,
    movement:    document.getElementById("movement").value.trim()    || null,
    orientation: document.getElementById("orientation").value.trim() || null,
    facial:      document.getElementById("facial").value.trim()      || null,
    notes:       document.getElementById("notes").value.trim()       || null,
  };

  await saveAnnotation(ann);
  const synced = await saveToSupabase(ann);
  annotatedIds.add(segId);
  sessionAnns.unshift({ ...ann, synced });
  updateSessionList();

  const total = await countAnnotations();
  document.getElementById("stat-done").textContent = `${total} anotações`;
  flashStatus(synced ? "✓ Salvo e sincronizado!" : "✓ Salvo localmente.");

  clearTimes();
  resetForm();
  updateSegNavUI();

  // Avança automaticamente para o próximo não anotado
  if (segments.length) {
    let next = currentSegIdx + 1;
    while (next < segments.length && annotatedIds.has(segments[next].id)) next++;
    if (next < segments.length) setTimeout(() => loadSegment(next), 400);
  }
}

// ── Lista da sessão ───────────────────────────────────────────
function updateSessionList() {
  const wrap = document.getElementById("session-segs");
  const list = document.getElementById("segs-list");
  if (!sessionAnns.length) { wrap.style.display = "none"; return; }
  wrap.style.display = "block";

  list.innerHTML = sessionAnns.slice(0, 10).map(a => `
    <div class="seg-item">
      <span class="seg-item-time">${formatTime(a.t_start)} – ${formatTime(a.t_end)}</span>
      <span class="seg-item-label ${a.label ? '' : 'no-label'}">${a.label || a.outro_name || '?'}</span>
      <span class="seg-item-dur">${a.duration}s</span>
      <span title="${a.synced ? 'Sincronizado' : 'Apenas local'}">${a.synced ? '☁' : '💾'}</span>
      <button class="btn-play-seg" title="Replay">▶</button>
    </div>
  `).join("");

  list.querySelectorAll(".btn-play-seg").forEach((btn, i) => {
    btn.addEventListener("click", () => {
      const a = sessionAnns[i];
      if (!ytPlayer) return;
      if (currentVideoId !== a.video_id) {
        currentVideoId = a.video_id;
        ytPlayer.loadVideoById({ videoId: a.video_id, startSeconds: a.t_start });
      } else {
        ytPlayer.seekTo(a.t_start, true);
      }
      ytPlayer.playVideo();
    });
  });
}

// ── Reset formulário ──────────────────────────────────────────
function resetForm() {
  document.querySelector('[name="valid"][value="yes"]').checked    = true;
  document.querySelector('[name="confidence"][value="2"]').checked = true;
  ["sign-label","outro-name","handshape","location","movement","orientation","facial","notes"]
    .forEach(id => { document.getElementById(id).value = ""; });
  document.getElementById("field-outro").style.display = "none";
}

// ── Export ────────────────────────────────────────────────────
async function exportData() {
  const annotations = await getAllAnnotations();
  if (!annotations.length) { alert("Nenhuma anotação para exportar."); return; }
  exportAnnotationsJSON(annotations);
}

// ── Helpers ───────────────────────────────────────────────────
function formatTime(s) {
  if (s === null || s === undefined) return "—";
  const m  = Math.floor(s / 60);
  const ss = (s % 60).toFixed(1).padStart(4, "0");
  return m > 0 ? `${m}:${ss}` : `${ss}s`;
}

function flashStatus(msg, type = "ok") {
  const el = document.getElementById("ann-status");
  el.textContent = msg;
  el.style.color = type === "error" ? "var(--red)" : "var(--green)";
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.textContent = ""; }, 3000);
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("modal-welcome").style.display = "flex";
  document.getElementById("btn-start").addEventListener("click", () => {
    annotator = document.getElementById("annotator-name").value.trim() || "anon";
    document.getElementById("modal-welcome").style.display = "none";
    initSegmentQueue();
  });

  document.getElementById("btn-load-video").addEventListener("click", loadVideo);
  document.getElementById("yt-url-input").addEventListener("keydown", e => {
    if (e.key === "Enter") loadVideo();
  });

  document.getElementById("btn-mark-start").addEventListener("click", markStart);
  document.getElementById("btn-mark-end").addEventListener("click",   markEnd);
  document.getElementById("btn-preview").addEventListener("click",    previewSegment);
  document.getElementById("btn-clear-times").addEventListener("click", clearTimes);

  document.getElementById("btn-seg-prev").addEventListener("click", segPrev);
  document.getElementById("btn-seg-next").addEventListener("click", segNext);

  document.getElementById("btn-submit").addEventListener("click", submitAnnotation);
  document.getElementById("sign-label").addEventListener("change", e => {
    document.getElementById("field-outro").style.display =
      e.target.value === "outro" ? "flex" : "none";
  });

  document.getElementById("btn-export").addEventListener("click", exportData);

  countAnnotations().then(n => {
    document.getElementById("stat-done").textContent = `${n} anotações`;
  });
});
