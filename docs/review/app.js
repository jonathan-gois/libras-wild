// LiBERT — site de revisão manual da pré-anotação do wild.
// Sem backend: progresso fica em localStorage, exportação é um download de CSV.

const STORAGE_PREFIX = "libert_review_v1";

let clips = [];
let reviewer = "";
let idx = 0;
let answers = {}; // key = video_id__seg_id -> {real_gloss, sem_sinal, nao_sei}

const el = (id) => document.getElementById(id);

function answerKey(c) {
  return `${c.video_id}__${c.seg_id}`;
}

function storageKey() {
  return `${STORAGE_PREFIX}_${reviewer}`;
}

function loadAnswers() {
  const raw = localStorage.getItem(storageKey());
  answers = raw ? JSON.parse(raw) : {};
}

function saveAnswers() {
  localStorage.setItem(storageKey(), JSON.stringify(answers));
}

function isReviewed(c) {
  const a = answers[answerKey(c)];
  if (!a) return false;
  return !!(a.real_gloss && a.real_gloss.trim()) || a.sem_sinal || a.nao_sei;
}

function currentAnswer(c) {
  return answers[answerKey(c)] || { real_gloss: "", sem_sinal: false, nao_sei: false };
}

function setAnswer(c, partial) {
  const key = answerKey(c);
  answers[key] = Object.assign(currentAnswer(c), partial);
  saveAnswers();
  renderProgress();
}

function renderProgress() {
  const reviewedCount = clips.filter(isReviewed).length;
  const pct = Math.round((reviewedCount / clips.length) * 100);
  el("progress-fill").style.width = pct + "%";
  el("progress-text").textContent =
    `Clipe ${idx + 1}/${clips.length} — ${reviewedCount}/${clips.length} revisados`;
}

function renderClip() {
  const c = clips[idx];
  const video = el("clip-video");
  video.src = "clips/" + c.clip_file;
  video.currentTime = 0;
  video.play().catch(() => {});

  const pct = Math.round(c.similarity * 100);
  if (c.status === "confiante") {
    el("predicted-info").innerHTML =
      `O modelo propôs: <b>"${c.predicted_class}"</b> (similaridade ${pct}%) ` +
      `<span class="tag-confiante">[confiante]</span>`;
  } else {
    el("predicted-info").innerHTML =
      `O modelo <b>não conseguiu identificar</b> este sinal ` +
      `<span class="tag-desconhecido">[desconhecido]</span>`;
  }

  el("question-text").textContent =
    "Qual é a glosa correta deste clipe (na sua opinião)?";

  const a = currentAnswer(c);
  el("gloss-input").value = a.real_gloss || "";
  el("no-sign-checkbox").checked = !!a.sem_sinal;
  el("no-id-checkbox").checked = !!a.nao_sei;

  renderProgress();
}

function wireClipEvents() {
  el("gloss-input").addEventListener("input", (e) => {
    const c = clips[idx];
    setAnswer(c, { real_gloss: e.target.value, sem_sinal: false, nao_sei: false });
    el("no-sign-checkbox").checked = false;
    el("no-id-checkbox").checked = false;
  });

  el("no-sign-checkbox").addEventListener("change", (e) => {
    const c = clips[idx];
    if (e.target.checked) {
      el("no-id-checkbox").checked = false;
      el("gloss-input").value = "";
      setAnswer(c, { sem_sinal: true, nao_sei: false, real_gloss: "" });
    } else {
      setAnswer(c, { sem_sinal: false });
    }
  });

  el("no-id-checkbox").addEventListener("change", (e) => {
    const c = clips[idx];
    if (e.target.checked) {
      el("no-sign-checkbox").checked = false;
      el("gloss-input").value = "";
      setAnswer(c, { nao_sei: true, sem_sinal: false, real_gloss: "" });
    } else {
      setAnswer(c, { nao_sei: false });
    }
  });

  el("btn-next").addEventListener("click", () => {
    if (idx < clips.length - 1) {
      idx++;
      localStorage.setItem(storageKey() + "_idx", idx);
      renderClip();
    } else {
      showDone();
    }
  });

  el("btn-prev").addEventListener("click", () => {
    if (idx > 0) {
      idx--;
      localStorage.setItem(storageKey() + "_idx", idx);
      renderClip();
    }
  });
}

function csvEscape(v) {
  if (v === null || v === undefined) return "";
  const s = String(v);
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function buildCsv() {
  const header = [
    "video_id", "seg_id", "clip_file", "t_start", "t_end",
    "predicted_class", "similarity", "status",
    "reviewer", "real_gloss", "sem_sinal", "nao_sei_identificar", "match",
  ];
  const rows = [header.join(",")];

  for (const c of clips) {
    const a = currentAnswer(c);
    let match = "";
    if (c.status === "confiante" && a.real_gloss && a.real_gloss.trim()) {
      match = a.real_gloss.trim().toLowerCase() === c.predicted_class.trim().toLowerCase() ? "sim" : "nao";
    }
    const row = [
      c.video_id, c.seg_id, c.clip_file, c.t_start, c.t_end,
      c.predicted_class, c.similarity, c.status,
      reviewer, a.real_gloss || "", a.sem_sinal ? "1" : "0", a.nao_sei ? "1" : "0", match,
    ].map(csvEscape);
    rows.push(row.join(","));
  }
  return rows.join("\n");
}

function exportCsv() {
  const csv = buildCsv();
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const ts = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  a.href = url;
  a.download = `wild_review_${reviewer}_${ts}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function showDone() {
  el("review-screen").classList.add("hidden");
  el("done-screen").classList.remove("hidden");
}

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
  el(id).classList.remove("hidden");
}

async function start() {
  const nameInput = el("reviewer-name").value.trim();
  if (!nameInput) {
    alert("Digite seu nome antes de começar.");
    return;
  }
  reviewer = nameInput.toLowerCase().replace(/\s+/g, "_");
  localStorage.setItem(STORAGE_PREFIX + "_last_reviewer", reviewer);

  const resp = await fetch("clips.json");
  clips = await resp.json();

  loadAnswers();
  const savedIdx = parseInt(localStorage.getItem(storageKey() + "_idx") || "0", 10);
  idx = isNaN(savedIdx) ? 0 : Math.min(savedIdx, clips.length - 1);

  showScreen("review-screen");
  wireClipEvents();
  renderClip();
}

window.addEventListener("DOMContentLoaded", () => {
  const lastReviewer = localStorage.getItem(STORAGE_PREFIX + "_last_reviewer");
  if (lastReviewer) el("reviewer-name").value = lastReviewer;

  el("start-btn").addEventListener("click", start);
  el("btn-export").addEventListener("click", exportCsv);
  el("btn-export-final").addEventListener("click", exportCsv);
});
