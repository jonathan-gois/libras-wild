from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WILD_DIR = PROJECT_ROOT / "dataset" / "wild"
LIBERT_DIR = Path(__file__).resolve().parent
CACHE_DIR = LIBERT_DIR / "cache"
CKPT_DIR = LIBERT_DIR / "checkpoints"

FEAT_DIM = 225          # 33 pose + 21 lhand + 21 rhand, 3 coords
POSE_N, LHAND_N, RHAND_N = 33, 21, 21

# Pré-processamento (mesmos parâmetros do pipeline MINDS-Libras v3)
KALMAN_Q = 1e-4
KALMAN_R = 5e-3
USE_Z_NORM = True

# Janela de pré-treino
WINDOW = 200
STRIDE = 100             # 50% overlap

# Masking (spans contíguos)
MASK_RATIO = 0.4
MASK_SPAN_MIN = 5
MASK_SPAN_MAX = 25

# Arquitetura
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 8
FFN_DIM = 1536
DROPOUT = 0.1

# Treino
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
EPOCHS = 100
VAL_FRACTION = 0.1
GRAD_CLIP = 1.0
SEED = 42
