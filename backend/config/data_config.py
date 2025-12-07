"""
Configuration for OAI Tri-Modal Survival Model

Separates local development settings from production (Kaggle) settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Data-related configuration."""
    # Paths
    data_dir: Path = Path("data/OAICompleteData_ASCII")
    processed_dir: Path = Path("data/processed")
    image_dir: Path = Path("data/images")  # For production with real images
    sandbox_dir: Path = Path("data/sandbox")  # For local testing
    
    # Cohort settings
    sample_size: Optional[int] = None  # None = use all data
    min_followup_days: int = 365  # Minimum follow-up time
    
    # Feature settings
    use_biomarkers: bool = True  # Only ~600 patients have biomarkers
    use_mri_features: bool = True  # Only ~3700 patients have detailed MRI
    imputation_method: str = "median"  # median, knn, mice


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Image backbone
    image_backbone: str = "densenet121"  # densenet121, swin_v2, efficientnet
    image_pretrained: bool = True
    image_size: int = 224
    
    # Feature dimensions
    clinical_dim: int = 15
    biomarker_dim: int = 10  # Expanded from 5
    image_feature_dim: int = 1024  # DenseNet-121 output
    
    # Fusion settings
    fusion_method: str = "concat"  # concat, attention, gated
    fusion_hidden_dim: int = 64
    dropout: float = 0.3
    
    # Multi-task learning
    use_kl_auxiliary: bool = False  # Auxiliary KL grade classification


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Regularization
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    
    # Evaluation
    val_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: Path = Path("models/checkpoints")


@dataclass 
class LocalConfig:
    """
    Configuration for local development/testing.
    Fast iteration with reduced data and simpler settings.
    """
    data: DataConfig = field(default_factory=lambda: DataConfig(
        sample_size=500,  # Quick testing
        use_biomarkers=True,
        use_mri_features=False,  # Skip MRI for speed
    ))
    
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        image_backbone="densenet121",
        image_size=224,
        fusion_method="concat",
    ))
    
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        epochs=5,  # Fast validation
        batch_size=8,  # Smaller for local memory
        early_stopping_patience=3,
    ))
    
    # Local-specific settings
    image_mode: str = "sandbox"  # Use random images
    device: str = "auto"  # auto-detect mps/cuda/cpu
    num_workers: int = 0  # Avoid multiprocessing issues on Mac
    verbose: bool = True


@dataclass
class ProductionConfig:
    """
    Configuration for production training (Kaggle).
    Full dataset with optimized settings.
    """
    data: DataConfig = field(default_factory=lambda: DataConfig(
        sample_size=None,  # Full dataset
        use_biomarkers=True,
        use_mri_features=True,
        imputation_method="median",
    ))
    
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        image_backbone="densenet121",
        image_size=224,
        fusion_method="concat",
        use_kl_auxiliary=False,
    ))
    
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        early_stopping_patience=15,
    ))
    
    # Production-specific settings
    image_mode: str = "production"  # Use real mapped images
    device: str = "auto"  # TPU on Kaggle
    num_workers: int = 4
    verbose: bool = True
    
    # Experiment tracking (optional)
    use_wandb: bool = False
    experiment_name: str = "oai_trimodal_v2"


def get_config(mode: str = "local"):
    """
    Get configuration based on mode.
    
    Args:
        mode: 'local' for development, 'production' for Kaggle training
        
    Returns:
        Configuration dataclass
    """
    if mode == "local":
        return LocalConfig()
    elif mode == "production":
        return ProductionConfig()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'local' or 'production'.")


def detect_environment() -> str:
    """
    Auto-detect whether we're running locally or on Kaggle.
    
    Returns:
        'local' or 'production'
    """
    # Check for Kaggle environment
    if os.path.exists("/kaggle/input"):
        return "production"
    
    # Check for TPU
    try:
        import torch_xla
        return "production"
    except ImportError:
        pass
    
    return "local"


# Feature sets for different model configurations
FEATURE_SETS = {
    "minimal": {
        "clinical": ["V00AGE", "P02SEX", "P01BMI", "V00WOMTSR", "V00WOMTSL"],
        "imaging": ["V00XRKL"],
        "biomarker": [],
    },
    "basic": {
        "clinical": [
            "V00AGE", "P02SEX", "P01BMI", 
            "V00WOMTSR", "V00WOMTSL",
            "V00KOOSQOL", "V00PASE"
        ],
        "imaging": ["V00XRKL"],
        "biomarker": [
            "V00Serum_Comp_lc", "V00Serum_CTXI_lc", 
            "V00Serum_HA_lc", "V00Serum_C2C_lc", "V00Serum_CPII_lc"
        ],
    },
    "full": {
        "clinical": [
            "V00AGE", "P02SEX", "P01BMI",
            "V00WOMTSR", "V00WOMTSL", "V00WOMPNR", "V00WOMPNL",
            "V00WOMSTFR", "V00WOMSTFL", "V00WOMFNR", "V00WOMFNL",
            "V00KOOSYMR", "V00KOOSYML", "V00KOOSQOL",
            "V00PASE", "V00EDCV", "V00INCOME",
        ],
        "imaging": [
            "V00XRKL", "V00XROSTL", "V00XROSTM",
            "V00XRJSL", "V00XRJSM",
            "V00MACLBML", "V00MACLCYS", "V00MACLCAR",
            "V00WMTMTH", "V00WLTMTH",
        ],
        "biomarker": [
            "V00Serum_C1_2C_lc", "V00Serum_C2C_lc", "V00Serum_CPII_lc",
            "V00Serum_Comp_lc", "V00Serum_CS846_lc",
            "V00Serum_CTXI_lc", "V00Serum_NTXI_lc",
            "V00Serum_HA_lc", "V00Serum_MMP_3_lc", "V00Serum_PIIANP_lc",
            "V00Urine_CTXII_lc", "V00Urine_NTXI_lc",
        ],
    },
}


if __name__ == "__main__":
    # Test configurations
    env = detect_environment()
    print(f"Detected environment: {env}")
    
    config = get_config(env)
    print(f"\nUsing configuration: {type(config).__name__}")
    print(f"  Sample size: {config.data.sample_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Image mode: {config.image_mode}")
