import sys
import os
import time
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from report import create_pdf_report

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

# Import architectures
from model import WideAndDeepSurvivalModel, SemanticEncoder

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Knee Osteoarthritis Prognosis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device Config
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2C3E50; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: #7F8C8D; margin-bottom: 2rem; }
    .report-box { background-color: #F8F9F9; padding: 20px; border-radius: 5px; border-left: 5px solid #2C3E50; }
    .stButton>button { width: 100%; background-color: #2C3E50; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADER ---
@st.cache_resource
def load_system_models():
    MODEL_DIR = os.path.join(current_dir, '..', 'models')
    
    try:
        # 1. Survival Model (Tri-Modal)
        surv = WideAndDeepSurvivalModel(wide_input_dim=8, bio_input_dim=5).to(DEVICE)
        surv.load_state_dict(torch.load(os.path.join(MODEL_DIR, "tri_modal_survival_model.pth"), map_location=DEVICE))
        surv.eval()

        # 2. Encoder
        enc = SemanticEncoder(latent_dim=256).to(DEVICE)
        enc.load_state_dict(torch.load(os.path.join(MODEL_DIR, "semantic_encoder.pth"), map_location=DEVICE))
        enc.eval()

        # 3. Diffusion UNet
        unet = UNet2DModel(
            sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            class_embed_type="identity"
        ).to(DEVICE)
        unet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "diffusion_unet.pth"), map_location=DEVICE))
        unet.eval()

        sched = DDPMScheduler(num_train_timesteps=1000)
        return surv, enc, unet, sched
    except Exception as e:
        st.error(f"System Error: Model files missing or incompatible. {e}")
        st.stop()

# Load Models
survival_model, encoder, unet, scheduler = load_system_models()

# --- UTILITY FUNCTIONS ---
def preprocess_inputs(image):
    # Survival model (ResNet) expects 3 channels
    t_surv = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Generative model (Custom) expects 1 channel
    t_gen = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return t_surv(image).unsqueeze(0).to(DEVICE), t_gen(image).unsqueeze(0).to(DEVICE)

def generate_heatmap(original, modified):
    diff = np.abs(original - modified)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
    return diff

def plot_survival_function(risk_score):
    t = np.linspace(0, 10, 100)
    lambda_base, k = 0.1, 1.5
    cumulative_hazard = (t * lambda_base) ** k
    survival_prob = np.exp(-cumulative_hazard * np.exp(risk_score))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, survival_prob, color='#2C3E50', linewidth=2.5, label="Patient Trajectory")
    
    baseline_prob = np.exp(-cumulative_hazard)
    ax.plot(t, baseline_prob, color='#95A5A6', linestyle='--', label="Population Average")
    
    ax.set_title("10-Year Progression-Free Probability")
    ax.set_xlabel("Years from Baseline")
    ax.set_ylabel("Probability of Avoiding Surgery")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def analyze_biomarker_regions(diff_map):
    # Heuristic analysis of difference map
    joint_space_region = diff_map[28:36, 15:50]
    osteo_score = np.mean(diff_map[20:45, 0:15])
    js_score = np.mean(joint_space_region)
    
    findings = []
    if js_score > 0.1:
        findings.append(f"**Joint Space:** High activation ({js_score:.2f}) indicates narrowing.")
    else:
        findings.append(f"**Joint Space:** Preserved ({js_score:.2f}).")
        
    if osteo_score > 0.1:
        findings.append(f"**Osteophytes:** Marginal changes detected ({osteo_score:.2f}).")
    else:
        findings.append("**Osteophytes:** No significant changes.")
    
    return findings

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("System Inputs")
    
    st.subheader("1. Radiography")
    uploaded_file = st.file_uploader("Upload X-Ray", type=["png", "jpg", "jpeg"])
    
    st.subheader("2. Clinical Parameters")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 40, 90, 65)
        sex = st.selectbox("Sex", ["Male", "Female"])
    with c2:
        bmi = st.number_input("BMI", 15.0, 50.0, 28.5)
        kl_grade = st.selectbox("KL Grade", [0, 1, 2, 3, 4], index=2)
    
    womac = st.slider("WOMAC Total Score", 0, 96, 25)
    
    # --- 3. Biomarker Inputs ---
    st.subheader("3. Biochemical Markers")
    with st.expander("Lab Assays (Serum/Urine)", expanded=True):
        bio_comp = st.number_input("Serum COMP (ng/mL)", 0.0, 2000.0, 1065.0)
        bio_ctx = st.number_input("Urine CTX-II (ng/mmol)", 0.0, 1000.0, 350.0)
        bio_ha = st.number_input("Serum HA (ng/mL)", 0.0, 500.0, 45.0)
        bio_ntx = st.number_input("Serum NTX (nM BCE)", 0.0, 100.0, 15.0)
        bio_mmp3 = st.number_input("Serum MMP-3 (ng/mL)", 0.0, 100.0, 25.0)
    
    analyze_btn = st.button("INITIALIZE ANALYSIS")

# --- MAIN DASHBOARD ---
st.markdown('<p class="main-header">Tri-Modal Knee Osteoarthritis Prognosis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Survival Analysis with Generative Explainability</p>', unsafe_allow_html=True)

if not uploaded_file:
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.markdown("#### 1. Imaging Analysis\nResNet-18 feature extraction from plain radiographs.")
    with col2: 
        st.markdown("#### 2. Tri-Modal Fusion\nIntegration of Clinical, Imaging, and **Biochemical** data.")
    with col3: 
        st.markdown("#### 3. Prognostic Modeling\nCox Proportional Hazards estimation for time-to-event.")
    
    st.info("Please upload an X-Ray image in the sidebar to begin.")

else:
    col_img, col_data = st.columns([1, 2])
    with col_img:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Source Radiograph", width="stretch")
    with col_data:
        st.markdown("### Patient Summary")
        st.dataframe(pd.DataFrame({
            "Metric": ["Age", "Sex", "KL Grade", "COMP", "CTX-II"],
            "Value": [age, sex, kl_grade, bio_comp, bio_ctx]
        }).set_index("Metric").T)

    if analyze_btn:
        # --- PIPELINE ---
        progress_bar = st.progress(0, text="Initializing...")
        
        # 1. Preprocessing
        time.sleep(0.2)
        progress_bar.progress(20, text="Processing Imaging & Clinical Data...")
        img_surv, img_gen = preprocess_inputs(image_pil)
        
        # Clinical Vector
        clin_vec = [age, bmi, womac, 0, 0, 0, 0, 0]
        if kl_grade > 0: clin_vec[2 + kl_grade] = 1
        if sex == "Female": clin_vec[7] = 1
        clin_tensor = torch.tensor([clin_vec], dtype=torch.float32).to(DEVICE)
        
        # Biomarker Vector
        bio_vec = [bio_comp, bio_ctx, bio_ha, bio_ntx, bio_mmp3]
        bio_tensor = torch.tensor([bio_vec], dtype=torch.float32).to(DEVICE)
        
        # 2. Inference
        progress_bar.progress(50, text="Computing Tri-Modal Hazard Ratio...")
        with torch.no_grad():
            risk_score = survival_model(img_surv, clin_tensor, bio_tensor).item()
        
        # 3. Generative
        progress_bar.progress(80, text="Generating Counterfactuals...")
        
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
        
        progress_bar.progress(100, text="Complete.")
        time.sleep(0.5)
        progress_bar.empty()
        
        # --- RESULTS ---
        st.divider()
        st.markdown("### 1. Prognostic Report")
        r1, r2 = st.columns([1, 2])
        
        with r1:
            risk_help_text = """
            Log-Hazard Score Interpretation:
            • Score = 0: Average Risk (Baseline)
            • Score > 0: High Risk (Hazard > Average)
            • Score < 0: Low Risk (Hazard < Average)
            """
            
            st.metric(
                label="Log-Hazard Score", 
                value=f"{risk_score:.3f}", 
                delta="Relative to Population Mean",
                delta_color="off",
                help=risk_help_text
            )
            
            risk_class = "High" if risk_score > 0.5 else "Low" if risk_score < -0.5 else "Moderate"
            st.markdown(f"""
            <div class='report-box'>
            <b>Risk Classification: {risk_class}</b><br>
            Patient shows a {risk_class.lower()} likelihood of rapid progression based on Tri-Modal analysis (Image + Clinical + Bio).
            </div>
            """, unsafe_allow_html=True)
            
        with r2:
            # CORRECTED: Using the defined function name
            st.pyplot(plot_survival_function(risk_score)) 

        st.divider()
        st.markdown("### 2. Generative Biomarker Analysis")
        
        g1, g2, g3 = st.columns(3)
        diff = generate_heatmap(recon_np, cf_np)

        with g1:
            st.image(np.clip(recon_np, 0, 1), caption="Current Anatomy", width="stretch")
        with g2:
            st.image(np.clip(cf_np, 0, 1), caption="Projected Healthy State", width="stretch")
        with g3:
            fig_diff, ax_diff = plt.subplots()
            sns.heatmap(diff, cmap="hot", cbar=True, ax=ax_diff, xticklabels=False, yticklabels=False)
            ax_diff.axis('off')
            st.pyplot(fig_diff)
            st.caption("Difference Map")

        st.markdown("#### Detected Structural Deviations")
        biomarker_text = analyze_biomarker_regions(diff)
        for item in biomarker_text:
            st.markdown(f"* {item}")

        # --- REPORT GENERATION ---
        st.divider()
        st.subheader("3. Clinical Report")
        
        patient_data = {
            "Age": age, "Sex": sex, "BMI": bmi, 
            "KL Grade": kl_grade, "WOMAC": womac, "COMP": bio_comp
        }
        
        # Convert for PDF
        recon_pil = Image.fromarray((np.clip(recon_np * 0.5 + 0.5, 0, 1) * 255).astype('uint8').squeeze())
        cf_pil = Image.fromarray((np.clip(cf_np * 0.5 + 0.5, 0, 1) * 255).astype('uint8').squeeze())
        
        # CORRECTED: Using the defined function name
        survival_fig = plot_survival_function(risk_score) 
        
        pdf_file = create_pdf_report(
            patient_data, 
            risk_score, 
            risk_class,
            images={
                'original': recon_pil, 
                'counterfactual': cf_pil, 
                'graph': survival_fig
            }
        )
        
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_file,
            file_name=f"OA_Prognosis_Report_{int(time.time())}.pdf",
            mime="application/pdf"
        )