"""
Gait Energy Image (GEI) from MediaPipe landmarks.

Renderiza imagens do esqueleto 2D a partir dos landmarks e calcula
a média temporal (GEI) por sequência.

MediaPipe Holistic layout:
  [0:99]   = 33 pose landmarks  (x, y, z) * 33
  [99:162] = 21 left-hand landmarks
  [162:225]= 21 right-hand landmarks
"""

import numpy as np
import cv2

# ---------- índices MediaPipe Holistic no vetor flat (x,y,z) ----------
POSE_N = 33
LHAND_N = 21
RHAND_N = 21

POSE_OFFSET  = 0
LHAND_OFFSET = POSE_N
RHAND_OFFSET = POSE_N + LHAND_N

# Conexões do esqueleto (pares de índices LOCAIS dentro de cada grupo)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def _get_group_xy(flat_frame: np.ndarray, offset: int, n: int):
    """Retorna array (n, 2) com x,y normalizados [0,1] do grupo."""
    seg = flat_frame[offset*3 : (offset + n)*3].reshape(n, 3)
    return seg[:, :2]  # x, y


def render_skeleton(flat_frame: np.ndarray,
                    H: int = 64, W: int = 48) -> np.ndarray:
    """
    Dado um frame (225,), renderiza o esqueleto em imagem binária (H x W).
    Retorna imagem float32 em [0,1].
    """
    canvas = np.zeros((H, W), dtype=np.float32)

    def draw_group(xy, connections, color=1.0):
        for (i, j) in connections:
            xi, yi = int(xy[i, 0] * W), int(xy[i, 1] * H)
            xj, yj = int(xy[j, 0] * W), int(xy[j, 1] * H)
            # clip
            xi, xj = np.clip([xi, xj], 0, W-1)
            yi, yj = np.clip([yi, yj], 0, H-1)
            cv2.line(canvas, (xi, yi), (xj, yj), color, 1)

    pose_xy  = _get_group_xy(flat_frame, POSE_OFFSET,  POSE_N)
    lhand_xy = _get_group_xy(flat_frame, LHAND_OFFSET, LHAND_N)
    rhand_xy = _get_group_xy(flat_frame, RHAND_OFFSET, RHAND_N)

    draw_group(pose_xy,  POSE_CONNECTIONS)
    draw_group(lhand_xy, HAND_CONNECTIONS)
    draw_group(rhand_xy, HAND_CONNECTIONS)

    return canvas


def compute_gei(sequence: np.ndarray, H: int = 64, W: int = 48) -> np.ndarray:
    """
    sequence: (T, 225)
    Retorna GEI flatten: (H*W,)
    """
    T = sequence.shape[0]
    acc = np.zeros((H, W), dtype=np.float32)
    for t in range(T):
        acc += render_skeleton(sequence[t], H, W)
    gei = acc / T  # média temporal

    # Crop região com movimento (bounding box de pixels não-zero)
    ys, xs = np.where(gei > 0)
    if len(ys) > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        crop = gei[y0:y1+1, x0:x1+1]
        # Redimensiona mantendo aspect ratio dentro de H x W
        crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
        gei = crop

    return gei.flatten()


def batch_gei(sequences: list, H: int = 64, W: int = 48) -> np.ndarray:
    """
    sequences: lista de arrays (T_i, 225)
    Retorna (N, H*W)
    """
    return np.stack([compute_gei(s, H, W) for s in sequences])
