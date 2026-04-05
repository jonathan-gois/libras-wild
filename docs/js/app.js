/**
 * app.js — Libras Wild: anotação de sinais segmentados automaticamente.
 *
 * Fluxo:
 *  1. Carrega data/wild_index.json (686 clipes de 15 vídeos)
 *  2. Para cada clipe: carrega o vídeo YouTube certo, salta para t_start/t_end
 *  3. Preview automático + pausa no t_end
 *  4. Usuário anota → salva IndexedDB + Supabase → avança automaticamente
 *  5. Exporta JSON consolidado
 *
 *  Modo manual (fallback): usuário cola qualquer URL e marca tempos
 */

// ── Estado ────────────────────────────────────────────────────
let annotator     = "";
let ytPlayer      = null;
let playerReady   = false;
let currentVideoId = null;
let tStart = null, tEnd = null;
let sessionAnns   = [];

// Fila wild
let wildQueue    = [];     // array de {seg_id, video_id, t_start, t_end, ...}
let queueIdx     = 0;
let annotatedSet = new Set();

// ── YouTube IFrame API ────────────────────────────────────────
window.onYouTubeIframeAPIReady = function () {
  ytPlayer = new YT.Player("yt-player", {
    height: "100%", width: "100%", videoId: "",
    playerVars: { controls: 1, modestbranding: 1, rel: 0, iv_load_policy: 3 },
    events: {
      onReady: () => { playerReady = true; if (wildQueue.length) goToClip(queueIdx); },
      onStateChange: autoStopAtEnd,
    }
  });
};

function autoStopAtEnd(e) {
  if (e.data !== YT.PlayerState.PLAYING || tEnd === null) return;
  const iv = setInterval(() => {
    if (!ytPlayer) { clearInterval(iv); return; }
    if (ytPlayer.getCurrentTime() >= tEnd) { ytPlayer.pauseVideo(); clearInterval(iv); }
  }, 200);
}

// ── Fila wild ─────────────────────────────────────────────────
async function initQueue() {
  try {
    wildQueue = await fetch("data/wild_index.json").then(r => r.json());
    const saved = await getAllAnnotations();
    saved.forEach(a => annotatedSet.add(a.seg_id));

    // Primeiro não anotado
    let first = wildQueue.findIndex(c => !annotatedSet.has(c.seg_id));
    if (first < 0) first = 0;
    queueIdx = first;

    document.getElementById("queue-nav").style.display = "block";
    renderQueueNav();
    if (playerReady) goToClip(queueIdx);
  } catch (e) {
    console.warn("wild_index.json não disponível — modo manual.", e);
  }
}

function goToClip(idx) {
  if (!wildQueue.length || idx < 0 || idx >= wildQueue.length) return;
  if (!playerReady) { queueIdx = idx; return; }
  queueIdx = idx;
  const clip = wildQueue[idx];

  // Oculta placeholder
  document.getElementById("video-placeholder").style.display = "none";
  document.getElementById("time-controls").style.display = "block";
  document.getElementById("yt-url-input").value = clip.video_url;

  if (currentVideoId !== clip.video_id) {
    currentVideoId = clip.video_id;
    ytPlayer.loadVideoById({ videoId: clip.video_id, startSeconds: clip.t_start });
    setTimeout(() => { ytPlayer.pauseVideo(); fillTimes(clip); }, 1800);
  } else {
    ytPlayer.seekTo(clip.t_start, true);
    ytPlayer.pauseVideo();
    fillTimes(clip);
  }
  renderQueueNav();
}

function fillTimes(clip) {
  tStart = clip.t_start;
  tEnd   = clip.t_end;
  document.getElementById("t-start").value = formatTime(tStart);
  document.getElementById("t-end").value   = formatTime(tEnd);
  updateDuration();
}

function renderQueueNav() {
  if (!wildQueue.length) return;
  const clip = wildQueue[queueIdx];
  const done = annotatedSet.size;
  document.getElementById("q-counter").textContent =
    `${queueIdx + 1} / ${wildQueue.length}`;
  document.getElementById("q-done").textContent = `${done} anotados`;
  document.getElementById("q-vid").textContent =
    `${clip.video_id}  ${formatTime(clip.t_start)}–${formatTime(clip.t_end)}` +
    `  score=${clip.spotter_score}`;
  document.getElementById("btn-q-prev").disabled = queueIdx <= 0;
  document.getElementById("btn-q-next").disabled = queueIdx >= wildQueue.length - 1;
  document.getElementById("queue-nav").classList.toggle(
    "done", annotatedSet.has(clip.seg_id));
}

function qNext() { if (queueIdx < wildQueue.length - 1) goToClip(queueIdx + 1); }
function qPrev() { if (queueIdx > 0) goToClip(queueIdx - 1); }

// ── Carrega vídeo manualmente ─────────────────────────────────
function extractYtId(url) {
  const m = url.trim().match(
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([A-Za-z0-9_-]{11})/
  ) || url.trim().match(/^([A-Za-z0-9_-]{11})$/);
  return m ? m[1] : null;
}

function loadVideo() {
  const vid = extractYtId(document.getElementById("yt-url-input").value);
  if (!vid) { flashStatus("URL inválida.", "error"); return; }
  if (!playerReady) { flashStatus("Player carregando…", "error"); return; }
  currentVideoId = vid;
  document.getElementById("video-placeholder").style.display = "none";
  ytPlayer.loadVideoById(vid);
  clearTimes();
  document.getElementById("time-controls").style.display = "block";
  flashStatus("Vídeo carregado.", "ok");
}

// ── Marcação ──────────────────────────────────────────────────
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
  } else {
    el.textContent = tEnd !== null && tStart !== null && tEnd <= tStart
      ? "⚠ fim antes do início" : "—";
    el.style.color = "var(--dgray)";
  }
}
function clearTimes() {
  tStart = tEnd = null;
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
function checkFormReady() {
  const ok = currentVideoId && tStart !== null && tEnd !== null && tEnd > tStart;
  document.getElementById("btn-submit").disabled = !ok;
}

// ── Submissão ─────────────────────────────────────────────────
async function submitAnnotation() {
  if (!currentVideoId || tStart === null || tEnd === null || tEnd <= tStart) {
    flashStatus("Marque início e fim.", "error"); return;
  }
  const curClip = wildQueue[queueIdx];
  const ann = {
    seg_id:      curClip ? curClip.seg_id : `${currentVideoId}_${Math.round(tStart*10)}`,
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
  annotatedSet.add(ann.seg_id);
  sessionAnns.unshift({ ...ann, synced });
  updateSessionList();

  const total = await countAnnotations();
  document.getElementById("stat-done").textContent = `${total} anotações`;
  flashStatus(synced ? "✓ Salvo e sincronizado!" : "✓ Salvo localmente.");

  clearTimes(); resetForm(); renderQueueNav();

  // Avança para próximo não anotado
  if (wildQueue.length) {
    let next = queueIdx + 1;
    while (next < wildQueue.length && annotatedSet.has(wildQueue[next].seg_id)) next++;
    if (next < wildQueue.length) setTimeout(() => goToClip(next), 400);
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
      <span class="seg-item-time">${formatTime(a.t_start)}–${formatTime(a.t_end)}</span>
      <span class="seg-item-label ${a.label ? '' : 'no-label'}">${a.label || a.outro_name || '?'}</span>
      <span class="seg-item-dur">${a.duration}s</span>
      <span title="${a.synced ? 'Sincronizado' : 'Local'}">${a.synced ? '☁' : '💾'}</span>
      <button class="btn-play-seg">▶</button>
    </div>`).join("");
  list.querySelectorAll(".btn-play-seg").forEach((btn, i) => {
    btn.addEventListener("click", () => {
      const a = sessionAnns[i];
      if (!ytPlayer) return;
      if (currentVideoId !== a.video_id) {
        currentVideoId = a.video_id;
        ytPlayer.loadVideoById({ videoId: a.video_id, startSeconds: a.t_start });
      } else { ytPlayer.seekTo(a.t_start, true); }
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

async function exportData() {
  const annotations = await getAllAnnotations();
  if (!annotations.length) { alert("Nenhuma anotação para exportar."); return; }
  exportAnnotationsJSON(annotations);
}

// ── Helpers ───────────────────────────────────────────────────
function formatTime(s) {
  if (s == null) return "—";
  const m = Math.floor(s / 60), ss = (s % 60).toFixed(1).padStart(4, "0");
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
    initQueue();
  });

  document.getElementById("btn-load-video").addEventListener("click", loadVideo);
  document.getElementById("yt-url-input").addEventListener("keydown",
    e => { if (e.key === "Enter") loadVideo(); });
  document.getElementById("btn-mark-start").addEventListener("click", markStart);
  document.getElementById("btn-mark-end").addEventListener("click",   markEnd);
  document.getElementById("btn-preview").addEventListener("click",    previewSegment);
  document.getElementById("btn-clear-times").addEventListener("click", clearTimes);
  document.getElementById("btn-q-prev").addEventListener("click", qPrev);
  document.getElementById("btn-q-next").addEventListener("click", qNext);
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
