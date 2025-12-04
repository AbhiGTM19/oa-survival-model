import sys
import os
import io
import base64
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import Depends

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
