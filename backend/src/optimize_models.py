import torch
import torch.nn as nn
import os
import sys
import logging
from diffusers import UNet2DModel

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- PATH SETUP ---
# Get the directory where this script is located (src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to src/
MODEL_DIR = os.path.join(current_dir, '..', 'models')
OUTPUT_DIR = os.path.join(current_dir, '..', 'models', 'optimized')

# Add src to python path to import model.py
sys.path.append(current_dir)
from model import WideAndDeepSurvivalModel, SemanticEncoder

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger.info("🚀 Starting Model Optimization Pipeline...")
logger.info(f"   Looking for models in: {os.path.abspath(MODEL_DIR)}")

def optimize_survival_model():
    logger.info("1. Optimizing Survival Model (Dynamic Quantization)...")
    fname = "tri_modal_survival_model.pth"
    path = os.path.join(MODEL_DIR, fname)
    
    if not os.path.exists(path):
        logger.warning(f"   ⚠️ Skipping: {fname} not found (Download from Kaggle first)")
        return

    try:
        # 1. Load Original (CPU)
        model = WideAndDeepSurvivalModel(wide_input_dim=8, bio_input_dim=5)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        
        # 2. Apply Dynamic Quantization (Linear Layers -> INT8)
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        # 3. Save
        out_path = os.path.join(OUTPUT_DIR, "tri_modal_survival_quantized.pth")
        torch.save(quantized_model.state_dict(), out_path)
        
        # Compare Sizes
        orig_size = os.path.getsize(path) / 1024
        new_size = os.path.getsize(out_path) / 1024
        logger.info(f"   ✅ Success! Size reduced: {orig_size:.1f} KB -> {new_size:.1f} KB ({new_size/orig_size:.1%})")
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")

def optimize_generative_models():
    logger.info("2. Optimizing Generative Models (FP16 Conversion)...")
    
    files = {
        "semantic_encoder.pth": SemanticEncoder(latent_dim=256),
        "diffusion_unet.pth": UNet2DModel(
            sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            class_embed_type="identity"
        )
    }
    
    for fname, model_arch in files.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"   ⚠️ Skipping: {fname} not found")
            continue

        try:
            # Load
            model_arch.load_state_dict(torch.load(path, map_location='cpu'))
            
            # Convert to Half Precision
            model_arch.half() 
            
            # Save
            out_name = fname.replace(".pth", "_fp16.pth")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            torch.save(model_arch.state_dict(), out_path)
            
            # Compare
            orig_size = os.path.getsize(path) / (1024 * 1024)
            new_size = os.path.getsize(out_path) / (1024 * 1024)
            logger.info(f"   ✅ {fname}: {orig_size:.1f} MB -> {new_size:.1f} MB (50.0%)")
            
        except Exception as e:
            logger.error(f"   ❌ Failed {fname}: {e}")

if __name__ == "__main__":
    optimize_survival_model()
    optimize_generative_models()
    logger.info("✨ Optimization Pipeline Finished.")