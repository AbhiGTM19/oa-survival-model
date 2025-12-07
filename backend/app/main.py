import sys
import os
import io
import base64
import logging
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any
from fastapi import FastAPI, Form, File, UploadFile, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from diffusers import UNet2DModel, DDIMScheduler
from pydantic import BaseModel

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
from .config import (
    MODEL_PATHS, DEFAULT_STATS, SEMANTIC_ENCODER_LATENT_DIM,
    DDPM_TIMESTEPS, DIFFUSION_INFERENCE_STEPS,
    TRIMODAL_WIDE_INPUT_DIM, TRIMODAL_BIO_INPUT_DIM,
    RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MODERATE, SRC_DIR
)

# --- PATH SETUP ---
# Add src to python path to import model definitions
sys.path.append(str(SRC_DIR))

from model import WideAndDeepSurvivalModel, SemanticEncoder
from .utils import preprocess_inputs, generate_heatmap, analyze_biomarker_regions
from .report import create_pdf_report

# --- GLOBAL STATE ---
MODELS = {}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading AI Models...")
    
    try:
        # 1. Random Forest
        rf_path = MODEL_PATHS['random_forest']
        if os.path.exists(rf_path):
            MODELS['rf'] = joblib.load(rf_path)
            logger.info("   ✅ Random Forest loaded")
        else:
            logger.warning(f"   ⚠️ Random Forest not found at {rf_path}")

        # 2. Stats
        stats_path = MODEL_PATHS['stats']
        if os.path.exists(stats_path):
             MODELS['stats'] = joblib.load(stats_path)
             logger.info("   ✅ Stats loaded")
        else:
             MODELS['stats'] = DEFAULT_STATS.copy()
             logger.warning("   ⚠️ Stats not found, using defaults")
             
        # 3. Encoder
        encoder_path = MODEL_PATHS['semantic_encoder']
        if os.path.exists(encoder_path):
            encoder = SemanticEncoder(latent_dim=SEMANTIC_ENCODER_LATENT_DIM).to(DEVICE)
            encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
            encoder.eval()
            MODELS['encoder'] = encoder
            logger.info("   ✅ Semantic Encoder loaded")
        else:
            logger.warning(f"   ⚠️ Semantic Encoder not found at {encoder_path}")
        
        # 4. UNet
        unet_path = MODEL_PATHS['unet']
        if os.path.exists(unet_path):
            unet = UNet2DModel(
                sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
                block_out_channels=(64, 128, 256, 512),
                down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                class_embed_type="identity"
            ).to(DEVICE)
            unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
            unet.eval()
            MODELS['unet'] = unet
            logger.info("   ✅ UNet loaded")
        else:
            logger.warning(f"   ⚠️ UNet not found at {unet_path}")
        
        # 5. Scheduler (DDIM for deterministic sampling)
        MODELS['scheduler'] = DDIMScheduler(
            num_train_timesteps=DDPM_TIMESTEPS,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
            set_alpha_to_one=False,
            prediction_type="epsilon"
        )
        logger.info("   ✅ DDIM Scheduler initialized")

        # 6. Tri-Modal Survival Model
        trimodal_path = MODEL_PATHS['trimodal']
        if os.path.exists(trimodal_path):
            trimodal = WideAndDeepSurvivalModel(
                wide_input_dim=TRIMODAL_WIDE_INPUT_DIM, 
                bio_input_dim=TRIMODAL_BIO_INPUT_DIM
            ).to(DEVICE)
            trimodal.load_state_dict(torch.load(trimodal_path, map_location=DEVICE))
            trimodal.eval()
            MODELS['trimodal'] = trimodal
            logger.info("   ✅ Tri-Modal Survival Model loaded")
        else:
            logger.warning(f"   ⚠️ Tri-Modal Model not found at {trimodal_path}")
        
    except Exception as e:
        logger.error(f"   ❌ Error loading models: {e}")
        # Don't crash, just log. Endpoints might fail though.

    yield
    MODELS.clear()

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput:
    def __init__(
        self,
        age: int = Form(...),
        sex: str = Form(...),
        bmi: float = Form(...),
        kl_grade: int = Form(...),
        womac: int = Form(...),
        pase: int = Form(...),
        koos: int = Form(...),
        stiffness: int = Form(...),
        nsaid: str = Form(...),
        bio_comp: float = Form(...),
        bio_ctx: float = Form(...),
        bio_ha: float = Form(...),
        bio_c2c: float = Form(...),
        bio_cpii: float = Form(...),
        mri_bml: float = Form(...),
        mri_cyst: float = Form(...),
        file: UploadFile = File(...)
    ):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.kl_grade = kl_grade
        self.womac = womac
        self.pase = pase
        self.koos = koos
        self.stiffness = stiffness
        self.nsaid = nsaid
        self.bio_comp = bio_comp
        self.bio_ctx = bio_ctx
        self.bio_ha = bio_ha
        self.bio_c2c = bio_c2c
        self.bio_cpii = bio_cpii
        self.mri_bml = mri_bml
        self.mri_cyst = mri_cyst
        self.file = file

@app.post("/api/predict")
async def predict(input_data: PatientInput = Depends()):
    # Check if models are loaded
    if 'rf' not in MODELS or 'encoder' not in MODELS:
        return JSONResponse(status_code=503, content={"message": "Models not loaded"})

    # 1. Process Image
    contents = await input_data.file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Process Data
    sex_val = 1 if input_data.sex == "Female" else 0
    nsaid_val = 1 if input_data.nsaid == "Yes" else 0
    
    patient_data = pd.DataFrame([{
        'Age': input_data.age, 'BMI': input_data.bmi, 'Sex': sex_val, 'KL_Grade': input_data.kl_grade, 
        'WOMAC_Score': input_data.womac, 'KOOS_Score': input_data.koos, 'Stiffness': input_data.stiffness, 
        'V00PASE': input_data.pase, 'V00NSAIDRX': nsaid_val, 
        'Bio_COMP': input_data.bio_comp, 'Bio_CTXI': input_data.bio_ctx, 'Bio_HA': input_data.bio_ha, 
        'Bio_C2C': input_data.bio_c2c, 'Bio_CPII': input_data.bio_cpii,
        'MRI_BML_Score': input_data.mri_bml,
        'Medial_Tibial_Thickness': 0.0, 'Lateral_Tibial_Thickness': 0.0,
        'Education': 3.0, 'Income': 3.0
    }])
    
    # 3. Risk Prediction
    rf = MODELS['rf']
    stats = MODELS['stats']
    
    raw_risk_score = rf.predict(patient_data)[0]
    min_score = stats.get('risk_score_min', 0.0)
    max_score = stats.get('risk_score_max', 1000.0)
    
    if max_score > min_score:
        risk_score = (raw_risk_score - min_score) / (max_score - min_score) * 100.0
    else:
        risk_score = 50.0
    risk_score = np.clip(risk_score, 0.0, 100.0)
    
    risk_class = "High" if risk_score > RISK_THRESHOLD_HIGH else "Moderate" if risk_score > RISK_THRESHOLD_MODERATE else "Low"
    
    # Survival Curve
    surv_funcs = rf.predict_survival_function(patient_data)
    fn = surv_funcs[0]
    years = (fn.x / 365.0).tolist()
    prob_surgery = (1.0 - fn(fn.x)).tolist()
    
    survival_data = [{"x": t, "y": p} for t, p in zip(years, prob_surgery)]
    
    # 4. Generative Analysis
    img_surv, img_gen = preprocess_inputs(image_pil)
    encoder = MODELS['encoder']
    unet = MODELS['unet']
    scheduler = MODELS['scheduler']
    
    with torch.no_grad():
        # Encode input image to semantic latent
        z = encoder(img_gen)
        
        # DETERMINISTIC SAMPLING: Use fixed seed for reproducibility
        # Same input -> same z -> same output (no random noise injection)
        torch.manual_seed(42)
        
        # Start from fixed noise (seeded)
        cf_image = torch.randn_like(img_gen)
        
        # DDIM deterministic reverse process
        scheduler.set_timesteps(DIFFUSION_INFERENCE_STEPS)
        for t in scheduler.timesteps:
            noise_pred = unet(cf_image, t, class_labels=z).sample
            cf_image = scheduler.step(noise_pred, t, cf_image, return_dict=False)[0]
            
    recon_np = img_gen.cpu().squeeze().numpy() * 0.5 + 0.5
    cf_np = cf_image.cpu().squeeze().numpy() * 0.5 + 0.5
    
    recon_pil = Image.fromarray((np.clip(recon_np, 0, 1) * 255).astype(np.uint8))
    cf_pil = Image.fromarray((np.clip(cf_np, 0, 1) * 255).astype(np.uint8))
    
    # Heatmap
    diff = generate_heatmap(recon_np, cf_np)
    # Normalize diff for visualization
    # Using matplotlib to get the "hot" colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('hot')
    colored_diff = cmap(diff)
    heatmap_pil = Image.fromarray((colored_diff[:, :, :3] * 255).astype(np.uint8))

    findings = analyze_biomarker_regions(diff)

    # 5. Tri-Modal Prediction (Optional/Logging for now)
    if 'trimodal' in MODELS:
        try:
            # Prepare Clinical Tensor (15 Features - matching dataset.py)
            # 1. Age, 2. BMI, 3. WOMAC
            # 4-7. KL Grade One-Hot (1, 2, 3, 4)
            # 8. Sex_2 (Female)
            # 9. V00KOOSQOL (KOOS), 10. V00PASE, 11. MRI_BML_Score
            # 12. Medial_Tibial_Thickness, 13. Lateral_Tibial_Thickness
            # 14. Education, 15. Income
            kl_1 = 1.0 if input_data.kl_grade == 1 else 0.0
            kl_2 = 1.0 if input_data.kl_grade == 2 else 0.0
            kl_3 = 1.0 if input_data.kl_grade == 3 else 0.0
            kl_4 = 1.0 if input_data.kl_grade == 4 else 0.0
            
            sex_2 = 1.0 if input_data.sex == "Female" else 0.0
            
            clin_features = [
                float(input_data.age), float(input_data.bmi), float(input_data.womac),
                kl_1, kl_2, kl_3, kl_4, sex_2,
                float(input_data.koos), float(input_data.pase), float(input_data.mri_bml),
                0.0, 0.0,  # Medial/Lateral Tibial Thickness (not collected from user)
                0.0, 0.0   # Education, Income (not collected from user)
            ]
            clin_tensor = torch.tensor([clin_features], dtype=torch.float32).to(DEVICE)
            
            # Prepare Biomarker Tensor (5 Features)
            bio_features = [
                float(input_data.bio_comp), float(input_data.bio_ctx), float(input_data.bio_ha),
                float(input_data.bio_c2c), float(input_data.bio_cpii)
            ]
            bio_tensor = torch.tensor([bio_features], dtype=torch.float32).to(DEVICE)
            
            # Prepare Image Tensor
            # img_surv is already [1, 3, 224, 224] on DEVICE from preprocess_inputs
            
            with torch.no_grad():
                trimodal_out = MODELS['trimodal'](img_surv, clin_tensor, bio_tensor)
                # Output is likely raw logit or risk score. 
                # For now, let's just print it to verify it works.
                logger.debug(f"Tri-Modal Output: {trimodal_out.item()}")
                
        except Exception as e:
            logger.error(f"Tri-Modal Inference Failed: {e}")
    
    return JSONResponse({
        "risk_score": float(risk_score),
        "risk_class": risk_class,
        "survival_curve": survival_data,
        "images": {
            "original": image_to_base64(recon_pil),
            "counterfactual": image_to_base64(cf_pil),
            "heatmap": image_to_base64(heatmap_pil)
        },
        "findings": findings,
        "patient_data": {
            "ID": "Anonymous",
            "Age": input_data.age,
            "Sex": input_data.sex,
            "BMI": input_data.bmi,
            "KL Grade": input_data.kl_grade,
            "WOMAC": input_data.womac,
            "COMP": input_data.bio_comp,
            "CTX": input_data.bio_ctx
        }
    })

class ReportRequest(BaseModel):
    patient_data: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    findings: List[str]
    images: Dict[str, str] # base64 strings

@app.post("/api/report")
async def generate_report(req: ReportRequest):
    try:
        # Decode images
        images = {}
        for key, b64_str in req.images.items():
            try:
                img_data = base64.b64decode(b64_str)
                images[key] = Image.open(io.BytesIO(img_data))
            except Exception as e:
                print(f"Error decoding image {key}: {e}")
                
        # Generate PDF
        pdf_buffer = create_pdf_report(
            req.patient_data,
            req.risk_analysis,
            req.findings,
            images
        )
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=OA_Prognosis_Report.pdf"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
