import sys
import os
import io
import base64
import json
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from diffusers import UNet2DModel, DDPMScheduler
from typing import Dict, Any, List

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)
sys.path.append(current_dir)

from model import SemanticEncoder
from utils import preprocess_inputs, DEVICE, generate_heatmap, analyze_biomarker_regions


# --- APP SETUP ---
app = FastAPI(title="Knee OA Prognosis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# --- MODEL LOADING ---
MODEL_DIR = os.path.join(current_dir, '..', 'models')
MODELS = {}

def load_models():
    if MODELS: return
    print("Loading models...")
    try:
        # 1. Random Forest
        MODELS['rf'] = joblib.load(os.path.join(MODEL_DIR, "random_forest_survival.joblib"))
        
        # 2. Imputer Stats
        MODELS['stats'] = joblib.load(os.path.join(MODEL_DIR, "imputer_stats.joblib"))
        
        # 3. Encoders & Diffusion
        enc = SemanticEncoder(latent_dim=256).to(DEVICE)
        enc.load_state_dict(torch.load(os.path.join(MODEL_DIR, "semantic_encoder.pth"), map_location=DEVICE))
        enc.eval()
        MODELS['encoder'] = enc
        
        unet = UNet2DModel(
            sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            class_embed_type="identity"
        ).to(DEVICE)
        unet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "diffusion_unet.pth"), map_location=DEVICE))
        unet.eval()
        MODELS['unet'] = unet
        
        MODELS['scheduler'] = DDPMScheduler(num_train_timesteps=1000)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load on startup
load_models()

# --- UTILS ---
def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(current_dir, "static", "index.html")) as f:
        return f.read()

@app.post("/api/predict")
async def predict(
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
    # 1. Process Image
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Process Data
    sex_val = 1 if sex == "Female" else 0
    nsaid_val = 1 if nsaid == "Yes" else 0
    
    patient_data = pd.DataFrame([{
        'Age': age, 'BMI': bmi, 'Sex': sex_val, 'KL_Grade': kl_grade, 
        'WOMAC_Score': womac, 'KOOS_Score': koos, 'Stiffness': stiffness, 
        'V00PASE': pase, 'V00NSAIDRX': nsaid_val, 
        'Bio_COMP': bio_comp, 'Bio_CTXI': bio_ctx, 'Bio_HA': bio_ha, 
        'Bio_C2C': bio_c2c, 'Bio_CPII': bio_cpii,
        'MRI_BML_Score': mri_bml,
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
    
    risk_class = "High" if risk_score > 70 else "Moderate" if risk_score > 30 else "Low"
    
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
        z = encoder(img_gen)
        z_mod = z + (torch.randn_like(z) * 1.5)
        scheduler.set_timesteps(20)
        cf_image = torch.randn_like(img_gen)
        for t in scheduler.timesteps:
            out = unet(cf_image, t, class_labels=z_mod).sample
            cf_image = scheduler.step(out, t, cf_image).prev_sample
            
    recon_np = img_gen.cpu().squeeze().numpy() * 0.5 + 0.5
    cf_np = cf_image.cpu().squeeze().numpy() * 0.5 + 0.5
    
    recon_pil = Image.fromarray((np.clip(recon_np, 0, 1) * 255).astype(np.uint8))
    cf_pil = Image.fromarray((np.clip(cf_np, 0, 1) * 255).astype(np.uint8))
    
    # Heatmap
    diff = generate_heatmap(recon_np, cf_np)
    # Normalize diff for visualization
    diff_vis = (diff * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(diff_vis).convert("L")
    # Apply colormap manually or just send grayscale? 
    # Let's send grayscale and handle coloring in frontend or use matplotlib here.
    # Using matplotlib to get the "hot" colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('hot')
    colored_diff = cmap(diff)
    heatmap_pil = Image.fromarray((colored_diff[:, :, :3] * 255).astype(np.uint8))

    findings = analyze_biomarker_regions(diff)
    
    return JSONResponse({
        "risk_score": float(risk_score),
        "risk_class": risk_class,
        "survival_curve": survival_data,
        "images": {
            "original": image_to_base64(recon_pil),
            "counterfactual": image_to_base64(cf_pil),
            "heatmap": image_to_base64(heatmap_pil)
        },
        "findings": findings
    })

# --- REPORT GENERATION ---
from pydantic import BaseModel
from report import create_pdf_report
from fastapi.responses import Response

class ReportRequest(BaseModel):
    patient_data: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    findings: List[str]
    images: Dict[str, str] # base64 strings

@app.post("/api/report")
async def generate_report(req: ReportRequest):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
