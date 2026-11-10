"""Central configuration for SynPhen-CL.

Override data paths with environment variables or edit this file. Keep model/training settings here
so training and inference share the same definitions.
"""
from pathlib import Path
import os

# Set before importing PyTorch/CUDA libraries for reproducibility.
DEFAULT_SEED = int(os.environ.get("SYNPHEN_SEED", os.environ.get("V2P_SEED", "42")))
os.environ["PYTHONHASHSEED"] = str(DEFAULT_SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EMBEDDING_DIR = DATA_DIR / "embeddings"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODEL_ASSET_DIR = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

TRAIN_CSV = os.environ.get("SYNPHEN_TRAIN_CSV", str(RAW_DATA_DIR / "train_selected.csv"))
TEST_CSV = os.environ.get("SYNPHEN_TEST_CSV", str(RAW_DATA_DIR / "test_selected.csv"))
PREDICT_CSV = os.environ.get("SYNPHEN_PREDICT_CSV", TEST_CSV)
CODER_NPY = os.environ.get("SYNPHEN_CODER_NPY", str(EMBEDDING_DIR / "coder_phenotype_embeddings.npy"))
CUI2VEC_NPY = os.environ.get("SYNPHEN_CUI2VEC_NPY", str(EMBEDDING_DIR / "cui2vec_phenotype_embeddings_avg.npy"))
CODER_MODEL_NAME = os.environ.get("SYNPHEN_CODER_MODEL_DIR", str(MODEL_ASSET_DIR / "coder"))
CUI2VEC_SOURCE_CSV = os.environ.get("SYNPHEN_CUI2VEC_SOURCE", str(EXTERNAL_DATA_DIR / "cui2vec_pretrained.csv"))
OUTPUT_DIR = os.environ.get("SYNPHEN_TRAIN_OUTPUT", str(OUTPUT_ROOT / "training"))
PREDICTION_OUTPUT_DIR = os.environ.get("SYNPHEN_PREDICT_OUTPUT", str(OUTPUT_ROOT / "predictions"))

ID_COL = "sequence_id"
LABEL_COL = "Label"
LABEL_SEP = "|"
N_TOTAL_LABELS = 24
N_LABELS = N_TOTAL_LABELS

# FT-Transformer hyperparameters
FT_D_TOKEN = 128
FT_N_LAYERS = 2
FT_N_HEADS = 8
FT_D_FFN_FACTOR = 2.0
FT_ATTN_DROPOUT = 0.20
FT_FFN_DROPOUT = 0.15
FT_RESIDUAL_DROPOUT = 0.00
FT_TOKEN_BIAS = True

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1.5e-4
LABEL_PRIOR_LR = 1e-5
WEIGHT_DECAY = 3e-4
PATIENCE = 10
MIN_EPOCHS_BEFORE_EARLY_STOP = 25
LR_SCHEDULER_PATIENCE = 8
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 5.0
EARLY_STOP_MIN_DELTA = 1e-5
INFERENCE_BATCH_SIZE = 256
FUSION_SEARCH_STEPS = 21
DEFAULT_FUSION_WEIGHT = 0.5
QUANTILE_MAX_QUANTILES = 1000
POS_WEIGHT_MIN = 1.0
POS_WEIGHT_MAX = 10.0
VAL_FRAC = 0.1
SEED = DEFAULT_SEED

D_MODEL = FT_D_TOKEN
N_HEADS = FT_N_HEADS
N_LABEL_FUSION_LAYERS = 1
DIM_FF = 4 * D_MODEL
TRANS_DROPOUT = 0.15

# Feature preprocessing
USE_FEATURE_SCALER = True
SCALER_TYPE = "quantile"  # quantile, standard, or none

# Loss weights and contrastive-learning schedule
ALPHA_CE = 0.3
BETA_FOCAL = 0.7
FOCAL_P_POS = 2
FOCAL_P_NEG = 2
FOCAL_CLIP_POS = 0.7
FOCAL_CLIP_NEG = 0.5
FOCAL_POS_WEIGHT = 0.3
FOCAL_REDUCTION = "mean"
ETA_CL = 0.003
CL_TEMPERATURE = 1.0
CL_WARMUP_START_EPOCH = 2
CL_WARMUP_EPOCHS = 6
CONTRAST_INCLUDE_BENIGN = True

# Training strategy
USE_PHASE2_RETRAIN = False
USE_WEIGHTED_FUSION = True
FUSION_METRIC = "micro_ap"  # micro_ap or macro_ap_pheno
EARLY_STOP_METRIC = "micro_ap"

# Runtime device; override with SYNPHEN_DEVICE=cpu or SYNPHEN_DEVICE=cuda.
try:
    import torch
    _auto_device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _auto_device = "cpu"
DEVICE = os.environ.get("SYNPHEN_DEVICE", _auto_device)
