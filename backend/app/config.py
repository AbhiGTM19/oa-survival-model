"""
Configuration settings for the OA Survival Model backend.
Values can be overridden via environment variables.
"""
import os
from pathlib import Path

# --- BASE PATHS ---
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
SRC_DIR = BASE_DIR / "src"

# --- RISK SCORE THRESHOLDS ---
# Risk classification thresholds (0-100 scale)
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "70"))
RISK_THRESHOLD_MODERATE = float(os.getenv("RISK_THRESHOLD_MODERATE", "30"))

# --- MODEL FILES ---
MODEL_PATHS = {
    "random_forest": os.path.join(MODELS_DIR, os.getenv("RF_MODEL_FILE", "random_forest_survival.joblib")),
    "stats": os.path.join(MODELS_DIR, os.getenv("STATS_FILE", "imputer_stats.joblib")),
    "semantic_encoder": os.path.join(MODELS_DIR, os.getenv("ENCODER_FILE", "semantic_encoder_final.pth")),
    "unet": os.path.join(MODELS_DIR, os.getenv("UNET_FILE", "diffusion_unet_final.pth")),
    "trimodal": os.path.join(MODELS_DIR, os.getenv("TRIMODAL_FILE", "tri_modal_survival_model.pth")),
}

# --- MODEL HYPERPARAMETERS ---
SEMANTIC_ENCODER_LATENT_DIM = int(os.getenv("ENCODER_LATENT_DIM", "256"))
DDPM_TIMESTEPS = int(os.getenv("DDPM_TIMESTEPS", "1000"))
DIFFUSION_INFERENCE_STEPS = int(os.getenv("DIFFUSION_INFERENCE_STEPS", "20"))
DIFFUSION_NOISE_SCALE = float(os.getenv("DIFFUSION_NOISE_SCALE", "1.5"))

# --- TRI-MODAL MODEL DIMENSIONS ---
TRIMODAL_WIDE_INPUT_DIM = int(os.getenv("TRIMODAL_WIDE_DIM", "11"))
TRIMODAL_BIO_INPUT_DIM = int(os.getenv("TRIMODAL_BIO_DIM", "5"))

# --- BIOMARKER REFERENCE RANGES ---
# Used for classifying biomarker levels as normal/elevated
BIOMARKER_THRESHOLDS = {
    "COMP": float(os.getenv("COMP_THRESHOLD", "1200")),  # ng/mL
    "CTX": float(os.getenv("CTX_THRESHOLD", "400")),     # ng/mmol
}

# --- DEFAULT STATS (when imputer_stats.joblib is not found) ---
DEFAULT_STATS = {
    "risk_score_min": float(os.getenv("DEFAULT_RISK_MIN", "0")),
    "risk_score_max": float(os.getenv("DEFAULT_RISK_MAX", "100")),
}

# --- SERVER SETTINGS ---
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
