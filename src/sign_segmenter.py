"""
sign_segmenter.py — Segmentação de sinais em diálogo contínuo de Libras.

Estratégia: detecção de picos e vales na curva de energia cinemática.

Base fonológica:
  - Durante um sinal: alta velocidade (fase de movimento/stroke)
  - Entre sinais: pausa/hold → vale local na curva de velocidade
  - Fronteira do sinal = vale entre dois picos consecutivos

Pipeline:
  1. Energia = velocidade dos pulsos + centros das mãos (MediaPipe)
  2. Suavização gaussiana dupla (curta para picos, longa para envoltória)
  3. find_peaks() encontra centros de movimento
  4. Vale entre picos consecutivos = fronteira do sinal
  5. Segmento = [vale_anterior, vale_posterior] ao redor de cada pico
  6. Filtros: duração mínima/máxima, energia mínima

Referências:
  - Koller et al. (2016): "Continuous sign language recognition: Towards large vocabulary statistical recognition systems"
  - Lim et al. (2019): "Isolated sign language recognition with multi-scale spatial-temporal graph CNN"
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from dataclasses import dataclass, field


@dataclass
class SignSegment:
    seg_id:       str
    t_start:      float
    t_end:        float
    duration:     float
    frame_start:  int
    frame_end:    int
    peak_frame:   int          # frame de maior energia (centro do sinal)
    energy_peak:  float        # energia máxima no segmento
    energy_mean:  float        # energia média (qualidade)
    top5:         list = field(default_factory=list)


# ── Índices MediaPipe Holistic (landmarks de 75 pontos × 3 coords) ────────────
# Pose: 0-32  |  Mão esq: 33-53  |  Mão dir: 54-74
RWRIST = 16   # pulso direito (pose)
LWRIST = 15   # pulso esquerdo (pose)
LHAND_BASE = 33   # raiz mão esquerda
RHAND_BASE = 54   # raiz mão direita


def compute_energy(landmarks: np.ndarray, fps: float,
                   smooth_sigma_s: float = 0.15) -> np.ndarray:
    """
    Energia por frame: velocidade dos pulsos + centros das mãos.

    Args:
        landmarks: (T, 225) float32
        fps: frames por segundo
        smooth_sigma_s: suavização gaussiana em segundos (padrão: 0.15s)

    Returns:
        energy: (T,) float32 — energia normalizada [0, 1]
    """
    T = len(landmarks)
    xyz = landmarks.reshape(T, 75, 3)

    # Centros das mãos (média dos 21 landmarks de cada mão)
    lhand_center = xyz[:, 33:54, :2].mean(axis=1)   # (T, 2)
    rhand_center = xyz[:, 54:75, :2].mean(axis=1)   # (T, 2)

    # Pulsos
    lwrist = xyz[:, LWRIST, :2]   # (T, 2)
    rwrist = xyz[:, RWRIST, :2]   # (T, 2)

    # Velocidade frame a frame (diferença finita)
    def vel(seq):
        v = np.diff(seq, axis=0)            # (T-1, 2)
        v = np.concatenate([[v[0]], v])     # (T, 2)  — replica primeiro frame
        return np.linalg.norm(v, axis=1)   # (T,)

    # Energia combinada: max dos pulsos + contribuição dos centros das mãos
    e = (0.4 * np.maximum(vel(lwrist), vel(rwrist)) +
         0.3 * vel(lhand_center) +
         0.3 * vel(rhand_center))

    # Suavização gaussiana (sigma em frames)
    sigma = smooth_sigma_s * fps
    if sigma > 0.5:
        e = gaussian_filter1d(e, sigma=sigma)

    # Normaliza para [0, 1]
    emax = e.max()
    if emax > 1e-9:
        e = e / emax

    return e.astype(np.float32)


def segment_signs(landmarks: np.ndarray,
                  fps: float,
                  # Suavização
                  smooth_sigma_s: float      = 0.08,  # sigma fino para picos
                  # Critérios de pico
                  min_peak_height: float     = 0.25,  # fração do máximo global
                  min_peak_dist_s: float     = 0.25,  # distância mínima entre picos (s)
                  min_peak_prominence: float = 0.15,  # proeminência mínima
                  # Janela centrada no pico
                  half_win_s: float          = 0.55,  # meio-janela máx ao redor do pico (s)
                  energy_drop: float         = 0.20,  # para de expandir quando energia < drop * pico
                  # Critérios de segmento
                  min_dur_s: float           = 0.25,  # duração mínima (s)
                  max_dur_s: float           = 1.8,   # duração máxima — 1 sinal tem ~0.5-1.5s
                  min_energy_mean: float     = 0.06,
                  ) -> list[SignSegment]:
    """
    Detecta sinais individuais em Libras contínuo.

    Estratégia: janela adaptativa CENTRADA NO PICO de energia.
    Expande a partir do pico enquanto energia > (energy_drop * e_pico),
    limitando a half_win_s em cada direção. Garante clips de 0.3-1.8s
    contendo um único sinal.
    """
    T = len(landmarks)
    energy = compute_energy(landmarks, fps, smooth_sigma_s)

    # ── 1. Encontra picos ──────────────────────────────────────────────────────
    min_dist_f = max(1, int(min_peak_dist_s * fps))
    peaks, _ = find_peaks(
        energy,
        height     = min_peak_height,
        distance   = min_dist_f,
        prominence = min_peak_prominence,
    )
    if len(peaks) == 0:
        return []

    # ── 2. Janela adaptativa centrada em cada pico ─────────────────────────────
    half_f = int(half_win_s * fps)
    min_f  = int(min_dur_s  * fps)
    max_f  = int(max_dur_s  * fps)
    segments = []

    for seg_idx, peak in enumerate(peaks):
        e_peak = float(energy[peak])
        thresh = energy_drop * e_peak

        # Expande para a esquerda enquanto energia > thresh
        fs = peak
        while fs > 0 and energy[fs - 1] > thresh and (peak - fs) < half_f:
            fs -= 1

        # Expande para a direita
        fe = peak
        while fe < T - 1 and energy[fe + 1] > thresh and (fe - peak) < half_f:
            fe += 1

        # Limita ao tamanho máximo recentrando no pico
        dur_f = fe - fs
        if dur_f < min_f:
            # Estende simetricamente
            ext = (min_f - dur_f) // 2
            fs = max(0, fs - ext)
            fe = min(T - 1, fe + ext)
            dur_f = fe - fs
        if dur_f > max_f:
            half = max_f // 2
            fs = max(0,     peak - half)
            fe = min(T - 1, peak + half)
            dur_f = fe - fs

        seg_energy = energy[fs:fe + 1]
        mean_e = float(seg_energy.mean())
        if mean_e < min_energy_mean:
            continue

        segments.append(SignSegment(
            seg_id      = f"seg_{seg_idx:04d}_{fs}",
            t_start     = fs / fps,
            t_end       = fe / fps,
            duration    = dur_f / fps,
            frame_start = int(fs),
            frame_end   = int(fe),
            peak_frame  = int(peak),
            energy_peak = e_peak,
            energy_mean = mean_e,
        ))

    # ── 3. Remove sobreposições >50% (mantém o de maior energia) ──────────────
    filtered = []
    for seg in sorted(segments, key=lambda s: s.energy_peak, reverse=True):
        overlap = False
        for kept in filtered:
            lo = max(seg.frame_start, kept.frame_start)
            hi = min(seg.frame_end,   kept.frame_end)
            if hi > lo:
                shorter = min(seg.duration, kept.duration)
                if (hi - lo) / (shorter * fps + 1e-9) > 0.5:
                    overlap = True
                    break
        if not overlap:
            filtered.append(seg)

    return sorted(filtered, key=lambda s: s.t_start)


def plot_segmentation(energy: np.ndarray, segments: list[SignSegment],
                      fps: float, out_path: str = None):
    """Visualiza curva de energia + segmentos detectados (requer matplotlib)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    t = np.arange(len(energy)) / fps
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, energy, color="#2ecc71", linewidth=0.8, label="energia")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2a3a5a")

    for seg in segments:
        ax.axvspan(seg.t_start, seg.t_end, alpha=0.25, color="#f39c12")
        ax.axvline(seg.peak_frame / fps, color="#e74c3c", linewidth=0.6, alpha=0.8)

    ax.set_xlabel("tempo (s)", color="white")
    ax.set_ylabel("energia normalizada", color="white")
    ax.set_title(f"{len(segments)} sinais detectados", color="white")
    ax.legend(facecolor="#16213e", labelcolor="white")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"Gráfico salvo: {out_path}")
    else:
        plt.show()
    plt.close()


# ── CLI rápido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pickle, json
    from pathlib import Path

    lm_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/youtube/-ZDkdbPqUZg/landmarks.pkl")
    fps      = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    plot     = "--plot" in sys.argv

    print(f"Carregando {lm_path}...")
    with open(lm_path, "rb") as f:
        lm = pickle.load(f)
    print(f"  {len(lm)} frames @ {fps}fps")

    segs = segment_signs(lm, fps)
    print(f"\n{len(segs)} sinais detectados")
    print(f"  Duração média: {np.mean([s.duration for s in segs]):.2f}s")
    print(f"  Min: {min(s.duration for s in segs):.2f}s  "
          f"Max: {max(s.duration for s in segs):.2f}s")

    for s in segs[:10]:
        print(f"  [{s.t_start:.1f}s – {s.t_end:.1f}s] {s.duration:.2f}s  "
              f"e_peak={s.energy_peak:.3f}")

    out = lm_path.parent / "segments_v2.json"
    data = [{
        "seg_id": s.seg_id, "t_start": round(s.t_start, 3),
        "t_end": round(s.t_end, 3), "duration": round(s.duration, 3),
        "frame_start": s.frame_start, "frame_end": s.frame_end,
        "peak_frame": s.peak_frame, "energy_peak": round(s.energy_peak, 4),
        "energy_mean": round(s.energy_mean, 4), "top5": [],
    } for s in segs]
    out.write_text(json.dumps(data, indent=2))
    print(f"\nSalvo: {out}")

    if plot:
        energy = compute_energy(lm, fps)
        plot_segmentation(energy, segs, fps,
                          out_path=str(lm_path.parent / "segmentation.png"))
