/**
 * app.js — Libras Wild: anota sinais em qualquer vídeo do YouTube.
 *
 * Fluxo:
 *  1. Usuário cola URL do YouTube
 *  2. Vídeo carrega no player
 *  3. Usuário assiste e clica "Marcar Início" / "Marcar Fim" no momento certo
 *  4. Preenche o formulário de anotação
 *  5. Salva em IndexedDB + lista os segmentos da sessão
 *  6. Exporta JSON para contribuição ao dataset
 */

// ── Estado ────────────────────────────────────────────────────
let annotator  = "";
let ytPlayer   = null;
let ytReady    = false;
let currentVideoId = null;
let tStart     = null;
let tEnd       = null;
let sessionAnns = [];   // anotações desta sessão (para exibir na lista)

// ── YouTube IFrame API ────────────────────────────────────────
window.onYouTubeIframeAPIReady = function () {
  ytReady = true;
  ytPlayer = new YT.Player("yt-player", {
    height: "100%",
    width:  "100%",
    videoId: "",
    playerVars: {
      controls:        1,
      modestbranding:  1,
      rel:             0,
      iv_load_policy:  3,
    },
    events: { onReady: () => {} }
  });
};

// ── Extrai ID do YouTube de qualquer formato de URL ───────────
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

// ── Carrega vídeo ─────────────────────────────────────────────
function loadVideo() {
  const raw = document.getElementById("yt-url-input").value;
  const vid = extractYtId(raw);
  if (!vid) {
    flashStatus("URL inválida. Use um link do YouTube.", "error");
    return;
  }
  if (!ytPlayer || !ytPlayer.loadVideoById) {
    flashStatus("Player ainda carregando, tente em 2 segundos.", "error");
    return;
  }

  currentVideoId = vid;
  document.getElementById("video-placeholder").style.display = "none";
  ytPlayer.loadVideoById(vid);
  clearTimes();

  document.getElementById("time-controls").style.display = "block";
  document.getElementById("time-hint").style.display = "block";
  flashStatus("Vídeo carregado. Assista e marque o trecho.", "ok");
}

// ── Marcação de tempos ────────────────────────────────────────
function markStart() {
  if (!ytPlayer) return;
  tStart = parseFloat(ytPlayer.getCurrentTime().toFixed(2));
  document.getElementById("t-start").value = formatTime(tStart);
  updateDuration();
  flashStatus(`Início marcado: ${formatTime(tStart)}`, "ok");
}

function markEnd() {
  if (!ytPlayer) return;
  tEnd = parseFloat(ytPlayer.getCurrentTime().toFixed(2));
  document.getElementById("t-end").value = formatTime(tEnd);
  updateDuration();
  flashStatus(`Fim marcado: ${formatTime(tEnd)}`, "ok");
}

function updateDuration() {
  const el = document.getElementById("time-dur");
  if (tStart !== null && tEnd !== null && tEnd > tStart) {
    const dur = (tEnd - tStart).toFixed(2);
    el.textContent = `${dur}s`;
    el.style.color = dur < 0.3 ? "var(--red)" : "var(--green)";
    // Habilita botão salvar se formulário ok
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
  tStart = null;
  tEnd   = null;
  document.getElementById("t-start").value = "";
  document.getElementById("t-end").value   = "";
  document.getElementById("time-dur").textContent = "—";
  document.getElementById("time-dur").style.color = "var(--dgray)";
  checkFormReady();
}

function previewSegment() {
  if (!ytPlayer || tStart === null || tEnd === null || tEnd <= tStart) {
    flashStatus("Marque início e fim primeiro.", "error");
    return;
  }
  ytPlayer.seekTo(tStart, true);
  ytPlayer.playVideo();
  // Para automaticamente no fim
  const duration = (tEnd - tStart) * 1000;
  setTimeout(() => {
    if (ytPlayer.getPlayerState() === YT.PlayerState.PLAYING)
      ytPlayer.pauseVideo();
  }, duration + 200);
}

// ── Validação do formulário ───────────────────────────────────
function checkFormReady() {
  const ok = currentVideoId && tStart !== null && tEnd !== null && tEnd > tStart;
  document.getElementById("btn-submit").disabled = !ok;
}

// ── Submissão ─────────────────────────────────────────────────
async function submitAnnotation() {
  if (!currentVideoId || tStart === null || tEnd === null || tEnd <= tStart) {
    flashStatus("Marque o início e fim do sinal primeiro.", "error");
    return;
  }

  const valid      = document.querySelector('[name="valid"]:checked')?.value;
  const label      = document.getElementById("sign-label").value;
  const outroName  = document.getElementById("outro-name").value.trim();
  const confidence = document.querySelector('[name="confidence"]:checked')?.value;

  const ann = {
    seg_id:      `${currentVideoId}_${Math.round(tStart*10)}`,
    annotator,
    ts:          Date.now(),
    video_id:    currentVideoId,
    video_url:   `https://youtu.be/${currentVideoId}`,
    t_start:     tStart,
    t_end:       tEnd,
    duration:    parseFloat((tEnd - tStart).toFixed(2)),
    valid,
    label:       label || null,
    outro_name:  outroName || null,
    confidence:  parseInt(confidence),
    handshape:   document.getElementById("handshape").value.trim()    || null,
    location:    document.getElementById("location").value.trim()     || null,
    movement:    document.getElementById("movement").value.trim()     || null,
    orientation: document.getElementById("orientation").value.trim()  || null,
    facial:      document.getElementById("facial").value.trim()       || null,
    notes:       document.getElementById("notes").value.trim()        || null,
  };

  // Salva localmente (IndexedDB) — funciona offline
  await saveAnnotation(ann);

  // Salva remotamente (Supabase) — para agregar dados de todos os voluntários
  const synced = await saveToSupabase(ann);
  sessionAnns.unshift({ ...ann, synced });
  updateSessionList();

  const total = await countAnnotations();
  document.getElementById("stat-done").textContent = `${total} anotações`;

  flashStatus(synced ? "✓ Salvo e sincronizado!" : "✓ Salvo localmente (offline).");

  // Limpa tempos e form para próxima anotação
  clearTimes();
  resetForm();
}

// ── Lista de segmentos da sessão ──────────────────────────────
function updateSessionList() {
  const wrap = document.getElementById("session-segs");
  const list = document.getElementById("segs-list");
  if (!sessionAnns.length) { wrap.style.display = "none"; return; }
  wrap.style.display = "block";

  list.innerHTML = sessionAnns.slice(0, 10).map(a => `
    <div class="seg-item" data-start="${a.t_start}" data-end="${a.t_end}">
      <span class="seg-item-time">${formatTime(a.t_start)} – ${formatTime(a.t_end)}</span>
      <span class="seg-item-label ${a.label ? '' : 'no-label'}">${a.label || a.outro_name || '?'}</span>
      <span class="seg-item-dur">${a.duration}s</span>
      <span title="${a.synced ? 'Sincronizado' : 'Apenas local'}">${a.synced ? '☁' : '💾'}</span>
      <button class="btn-play-seg" title="Replay">▶</button>
    </div>
  `).join("");

  // Replay de segmento salvo
  list.querySelectorAll(".btn-play-seg").forEach((btn, i) => {
    btn.addEventListener("click", () => {
      const a = sessionAnns[i];
      if (!ytPlayer) return;
      const vid = document.getElementById("yt-url-input").value;
      const id  = extractYtId(vid);
      if (id !== a.video_id)
        ytPlayer.loadVideoById({ videoId: a.video_id, startSeconds: a.t_start });
      else
        ytPlayer.seekTo(a.t_start, true);
      ytPlayer.playVideo();
    });
  });
}

// ── Reset formulário ──────────────────────────────────────────
function resetForm() {
  document.querySelector('[name="valid"][value="yes"]').checked    = true;
  document.querySelector('[name="confidence"][value="2"]').checked = true;
  document.getElementById("sign-label").value   = "";
  document.getElementById("outro-name").value   = "";
  document.getElementById("field-outro").style.display = "none";
  document.getElementById("handshape").value    = "";
  document.getElementById("location").value     = "";
  document.getElementById("movement").value     = "";
  document.getElementById("orientation").value  = "";
  document.getElementById("facial").value       = "";
  document.getElementById("notes").value        = "";
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
  // Modal de boas-vindas
  document.getElementById("modal-welcome").style.display = "flex";
  document.getElementById("btn-start").addEventListener("click", () => {
    annotator = document.getElementById("annotator-name").value.trim() || "anon";
    document.getElementById("modal-welcome").style.display = "none";
  });

  // Carregar vídeo
  document.getElementById("btn-load-video").addEventListener("click", loadVideo);
  document.getElementById("yt-url-input").addEventListener("keydown", e => {
    if (e.key === "Enter") loadVideo();
  });

  // Marcação
  document.getElementById("btn-mark-start").addEventListener("click", markStart);
  document.getElementById("btn-mark-end").addEventListener("click",   markEnd);
  document.getElementById("btn-preview").addEventListener("click",    previewSegment);
  document.getElementById("btn-clear-times").addEventListener("click", clearTimes);

  // Formulário
  document.getElementById("btn-submit").addEventListener("click", submitAnnotation);
  document.getElementById("sign-label").addEventListener("change", e => {
    document.getElementById("field-outro").style.display =
      e.target.value === "outro" ? "flex" : "none";
  });

  // Export
  document.getElementById("btn-export").addEventListener("click", exportData);

  // Contador inicial
  countAnnotations().then(n => {
    document.getElementById("stat-done").textContent = `${n} anotações`;
  });
});
