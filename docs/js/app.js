/**
 * app.js — Lógica principal da ferramenta de anotação Libras Wild.
 *
 * Fluxo:
 *  1. Carrega segments.json
 *  2. Mostra modal de boas-vindas
 *  3. Para cada segmento: toca YouTube embed, mostra skeleton, coleta anotação
 *  4. Salva em IndexedDB
 *  5. Exporta JSON para submissão ao projeto
 */

// ── Constantes ────────────────────────────────────────────────
const SEGMENTS_URL = "data/segments.json";
const CONFIG_URL   = "data/config.json";
const PRED_COLORS  = ["#2ecc71","#f39c12","#e74c3c","#9b59b6","#3498db"];
const LOOP_DELAY   = 800;   // ms entre repetições do clipe

// ── Estado ────────────────────────────────────────────────────
let segments    = [];
let config      = {};
let queue       = [];   // índices a anotar (não anotados ainda)
let queuePos    = 0;
let annotator   = "";
let ytPlayer    = null;
let ytReady     = false;
let skeleton    = null;
let loopTimer   = null;
let playCount   = 0;
let currentSeg  = null;

// ── Init ──────────────────────────────────────────────────────
async function init() {
  try {
    [config, segments] = await Promise.all([
      fetch(CONFIG_URL).then(r => r.json()),
      fetch(SEGMENTS_URL).then(r => r.json()),
    ]);
  } catch(e) {
    alert("Erro ao carregar dados: " + e.message);
    return;
  }

  skeleton = new SkeletonRenderer(document.getElementById("skeleton-canvas"));

  // Mostra modal de boas-vindas
  document.getElementById("modal-welcome").style.display = "flex";
  document.getElementById("btn-start").addEventListener("click", startSession);

  // Botões
  document.getElementById("btn-submit").addEventListener("click", submitAnnotation);
  document.getElementById("btn-prev").addEventListener("click", () => navigate(-1));
  document.getElementById("btn-skip").addEventListener("click", skip);
  document.getElementById("btn-replay").addEventListener("click", replayClip);
  document.getElementById("btn-export").addEventListener("click", exportData);
  document.getElementById("btn-export-done").addEventListener("click", exportData);
  document.getElementById("btn-restart").addEventListener("click", restartSkipped);

  document.getElementById("show-skeleton").addEventListener("change", e => {
    skeleton.setVisible(e.target.checked);
  });
  document.getElementById("show-hands").addEventListener("change", e => {
    skeleton.setShowHands(e.target.checked);
  });

  // sign-label: mostra/esconde campo "outro"
  document.getElementById("sign-label").addEventListener("change", e => {
    document.getElementById("field-outro").style.display =
      e.target.value === "outro" ? "flex" : "none";
  });

  // Atualiza contadores
  const done = await countAnnotations();
  updateCounters(done, segments.length);
}

async function startSession() {
  annotator = document.getElementById("annotator-name").value.trim() || "anon";
  document.getElementById("modal-welcome").style.display = "none";
  await buildQueue();
  if (queue.length === 0) {
    showDone();
    return;
  }
  queuePos = 0;
  loadSegment(queue[0]);
}

async function buildQueue() {
  // Coloca na fila segmentos ainda não anotados
  const annotated = new Set((await getAllAnnotations()).map(a => a.seg_id));
  queue = segments
    .map((s, i) => i)
    .filter(i => !annotated.has(segments[i].id));
}

// ── YouTube IFrame API callback ───────────────────────────────
window.onYouTubeIframeAPIReady = function() {
  ytReady = true;
  ytPlayer = new YT.Player("yt-player", {
    height: "100%",
    width: "100%",
    videoId: "",
    playerVars: {
      autoplay: 0,
      controls: 1,
      modestbranding: 1,
      rel: 0,
      iv_load_policy: 3,
      fs: 0,
    },
    events: {
      onReady:       onPlayerReady,
      onStateChange: onPlayerStateChange,
    }
  });
};

function onPlayerReady() {}

function onPlayerStateChange(e) {
  if (e.data === YT.PlayerState.ENDED || e.data === YT.PlayerState.PAUSED) {
    const seg = currentSeg;
    if (!seg) return;
    const cur = ytPlayer.getCurrentTime();
    if (cur >= seg.t_end - 0.1) {
      playCount++;
      document.getElementById("play-count").textContent = `× ${playCount}`;
      // Loop automático após delay
      clearTimeout(loopTimer);
      loopTimer = setTimeout(() => replayClip(), LOOP_DELAY);
    }
  }
}

// ── Carregamento de segmento ──────────────────────────────────
function loadSegment(segIdx) {
  currentSeg = segments[segIdx];
  const seg  = currentSeg;
  playCount  = 0;
  clearTimeout(loopTimer);

  // Header info
  document.getElementById("seg-id").textContent  = seg.id;
  document.getElementById("seg-time").textContent = `${seg.t_start.toFixed(1)}s – ${seg.t_end.toFixed(1)}s`;
  document.getElementById("seg-dur").textContent  = `${seg.duration.toFixed(2)}s`;
  document.getElementById("nav-counter").textContent =
    `${queuePos + 1} / ${queue.length}`;
  document.getElementById("play-count").textContent = "";

  // Predições
  renderPredictions(seg.top5 || []);

  // Pré-seleciona rótulo
  document.getElementById("sign-label").value = seg.pred_class || "";
  document.getElementById("field-outro").style.display = "none";

  // Reset formulário
  document.querySelector('[name="valid"][value="yes"]').checked = true;
  document.querySelector('[name="confidence"][value="2"]').checked = true;
  document.getElementById("handshape").value    = "";
  document.getElementById("location").value     = "";
  document.getElementById("movement").value     = "";
  document.getElementById("orientation").value  = "";
  document.getElementById("facial").value       = "";
  document.getElementById("notes").value        = "";
  document.getElementById("ann-status").textContent = "";

  // Toca vídeo
  playClip(seg);

  // Skeleton
  if (seg.keyframes && seg.keyframes.length > 0) {
    skeleton.load(seg.keyframes);
    skeleton.animate(8);
    document.getElementById("frame-display").textContent =
      `frame 0/${seg.keyframes.length}`;
  } else {
    skeleton.load([]);
    document.getElementById("frame-display").textContent = "sem dados de esqueleto";
  }

  // Verifica se já anotado (reedição)
  getAnnotation(seg.id).then(existing => {
    if (!existing) return;
    document.querySelector(`[name="valid"][value="${existing.valid}"]`).checked = true;
    document.getElementById("sign-label").value    = existing.label || "";
    document.getElementById("handshape").value     = existing.handshape || "";
    document.getElementById("location").value      = existing.location || "";
    document.getElementById("movement").value      = existing.movement || "";
    document.getElementById("orientation").value   = existing.orientation || "";
    document.getElementById("facial").value        = existing.facial || "";
    document.getElementById("notes").value         = existing.notes || "";
    if (existing.label === "outro")
      document.getElementById("field-outro").style.display = "flex";
    document.getElementById("ann-status").textContent = "↺ Recarregado (já anotado)";
  });
}

function playClip(seg) {
  if (!ytPlayer || !ytPlayer.loadVideoById) return;
  ytPlayer.loadVideoById({
    videoId:   config.youtube_id || "-ZDkdbPqUZg",
    startSeconds: seg.t_start,
    endSeconds:   seg.t_end,
  });
}

function replayClip() {
  if (!currentSeg) return;
  playClip(currentSeg);
}

// ── Predições ─────────────────────────────────────────────────
function renderPredictions(top5) {
  const list = document.getElementById("pred-list");
  list.innerHTML = "";
  const max = top5.length ? top5[0][1] : 1;
  top5.forEach(([label, prob], i) => {
    const pct  = (prob * 100).toFixed(1);
    const fill = (prob / max * 100).toFixed(1);
    const col  = PRED_COLORS[i] || PRED_COLORS[4];
    list.insertAdjacentHTML("beforeend", `
      <div class="pred-item">
        <span class="pred-label">${label}</span>
        <div class="pred-bar-wrap">
          <div class="pred-bar" style="width:${fill}%;background:${col}"></div>
        </div>
        <span class="pred-pct">${pct}%</span>
      </div>
    `);
  });
}

// ── Submissão ─────────────────────────────────────────────────
async function submitAnnotation() {
  const seg = currentSeg;
  if (!seg) return;

  const valid      = document.querySelector('[name="valid"]:checked')?.value;
  const label      = document.getElementById("sign-label").value;
  const outroName  = document.getElementById("outro-name").value.trim();
  const confidence = document.querySelector('[name="confidence"]:checked')?.value;

  if (!valid) { flashStatus("Selecione se é sinal válido.", "error"); return; }

  const ann = {
    seg_id:      seg.id,
    annotator,
    ts:          Date.now(),
    video_id:    config.youtube_id,
    t_start:     seg.t_start,
    t_end:       seg.t_end,
    duration:    seg.duration,
    valid,
    label:       label || null,
    outro_name:  outroName || null,
    confidence:  parseInt(confidence),
    pred_class:  seg.pred_class,
    pred_conf:   seg.pred_conf,
    handshape:   document.getElementById("handshape").value.trim() || null,
    location:    document.getElementById("location").value.trim()  || null,
    movement:    document.getElementById("movement").value.trim()  || null,
    orientation: document.getElementById("orientation").value.trim() || null,
    facial:      document.getElementById("facial").value.trim()    || null,
    notes:       document.getElementById("notes").value.trim()     || null,
  };

  await saveAnnotation(ann);

  skeleton.stop();
  clearTimeout(loopTimer);

  const done = await countAnnotations();
  updateCounters(done, segments.length);
  flashStatus("✓ Salvo!");

  // Avança
  setTimeout(() => advance(), 400);
}

function advance() {
  queuePos++;
  if (queuePos >= queue.length) {
    showDone();
    return;
  }
  loadSegment(queue[queuePos]);
}

function skip() {
  skeleton.stop();
  clearTimeout(loopTimer);
  advance();
}

function navigate(delta) {
  const next = queuePos + delta;
  if (next < 0 || next >= queue.length) return;
  queuePos = next;
  loadSegment(queue[queuePos]);
}

// ── Export ────────────────────────────────────────────────────
async function exportData() {
  const annotations = await getAllAnnotations();
  if (!annotations.length) { alert("Nenhuma anotação para exportar."); return; }
  exportAnnotationsJSON(annotations);
}

// ── Done modal ────────────────────────────────────────────────
async function showDone() {
  skeleton.stop();
  const done = await countAnnotations();
  document.getElementById("done-msg").innerHTML =
    `Você anotou <strong>${done}</strong> segmentos de um total de <strong>${segments.length}</strong>.<br>
     Exporte o JSON e envie para o repositório do projeto.`;
  document.getElementById("modal-done").style.display = "flex";
}

async function restartSkipped() {
  document.getElementById("modal-done").style.display = "none";
  await buildQueue();
  if (!queue.length) { showDone(); return; }
  queuePos = 0;
  loadSegment(queue[0]);
}

// ── UI helpers ────────────────────────────────────────────────
function updateCounters(done, total) {
  document.getElementById("stat-done").textContent  = `${done} anotados`;
  document.getElementById("stat-total").textContent = `${done} / ${total}`;
  document.getElementById("progress-bar").style.width =
    `${total ? (done / total * 100) : 0}%`;
}

function flashStatus(msg, type = "ok") {
  const el = document.getElementById("ann-status");
  el.textContent  = msg;
  el.style.color  = type === "error" ? "var(--red)" : "var(--green)";
  setTimeout(() => { el.textContent = ""; }, 2000);
}

// ── Start ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
