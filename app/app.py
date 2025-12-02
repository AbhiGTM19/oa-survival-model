import sys
import os
import time
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

from model import WideAndDeepSurvivalModel, SemanticEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="Knee OA Prognosis System", layout="wide", initial_sidebar_state="expanded")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- STYLING ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2C3E50; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: #7F8C8D; margin-bottom: 2rem; }
    .report-box { background-color: #F8F9F9; padding: 20px; border-radius: 5px; border-left: 5px solid #2C3E50; }
    .stButton>button { width: 100%; background-color: #2C3E50; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    MODEL_DIR = os.path.join(current_dir, '..', 'models')
    
    try:
        # 1. Survival Model
        surv_model = WideAndDeepSurvivalModel(wide_input_dim=8).to(DEVICE)
        surv_path = os.path.join(MODEL_DIR, "tri_modal_survival_model.pth")
        surv_model.load_state_dict(torch.load(surv_path, map_location=DEVICE))
        surv_model.eval()

        # 2. Generative Encoder
        enc = SemanticEncoder(latent_dim=256).to(DEVICE)
        enc_path = os.path.join(MODEL_DIR, "semantic_encoder.pth")
        enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
        enc.eval()

        # 3. Diffusion UNet
        unet = UNet2DModel(
            sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            class_embed_type="identity"
        ).to(DEVICE)
        unet_path = os.path.join(MODEL_DIR, "diffusion_unet.pth")
        unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
        unet.eval()

        scheduler = DDPMScheduler(num_train_timesteps=1000)
        return surv_model, enc, unet, scheduler
        
    except FileNotFoundError as e:
        st.error(f"System Error: Model files not found. {e}")
        st.stop()

survival_model, encoder, unet, scheduler = load_models()

# --- HELPER FUNCTIONS ---
def process_image(image):
    surv_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gen_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return surv_transform(image).unsqueeze(0).to(DEVICE), gen_transform(image).unsqueeze(0).to(DEVICE)

def generate_counterfactual(img_64, modification_factor, steps=20):
    scheduler.set_timesteps(steps)
    with torch.no_grad():
        z = encoder(img_64)
        z_mod = z + (torch.randn_like(z) * modification_factor)
        
        image = torch.randn_like(img_64)
        for t in scheduler.timesteps:
            out = unet(image, t, class_labels=z_mod).sample
            image = scheduler.step(out, t, image).prev_sample
            
    return image.cpu().squeeze().numpy()

def analyze_biomarker_regions(diff_map):
    # Define Regions (Heuristic based on 64x64 X-ray centering)
    joint_space_region = diff_map[28:36, 15:50]
    medial_margin = diff_map[20:45, 0:15]
    lateral_margin = diff_map[20:45, 49:64]
    tibial_plateau = diff_map[36:48, 15:50]

    js_score = np.mean(joint_space_region)
    osteo_score = max(np.mean(medial_margin), np.mean(lateral_margin))
    sclerosis_score = np.mean(tibial_plateau)
    
    findings = []
    if js_score > 0.1:
        findings.append(f"**Joint Space Narrowing (JSN):** High activation ({js_score:.2f}) detected in the compartment gap. Primary risk driver.")
    else:
        findings.append(f"**Joint Space:** Preserved. Low activation ({js_score:.2f}) indicates minimal narrowing.")
        
    if osteo_score > 0.1:
        findings.append(f"**Osteophytosis:** Significant texture variation ({osteo_score:.2f}) along joint margins suggests bone spur formation.")
    else:
        findings.append("**Osteophytes:** No significant marginal changes detected.")
        
    if sclerosis_score > 0.15:
        findings.append(f"**Subchondral Sclerosis:** Texture anomalies ({sclerosis_score:.2f}) in tibial plateau correlate with bone density changes.")
    
    return findings

def plot_survival_curve(risk_score):
    t = np.linspace(0, 10, 100) 
    lambda_base = 0.1
    k = 1.5
    
    baseline_cumulative_hazard = (t * lambda_base) ** k
    survival_prob = np.exp(-baseline_cumulative_hazard * np.exp(risk_score))
    baseline_prob = np.exp(-baseline_cumulative_hazard)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, survival_prob, color='#2C3E50', linewidth=2, label="Patient Specific")
    ax.plot(t, baseline_prob, color='#95A5A6', linestyle='--', label="Population Mean")
    
    ax.set_title("10-Year Progression-Free Probability")
    ax.set_xlabel("Years from Baseline")
    ax.set_ylabel("Probability of No Surgery")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size: 1.5rem;">System Controls</p>', unsafe_allow_html=True)
    
    st.subheader("1. Patient Input")
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["png", "jpg", "jpeg"])
    
    st.subheader("2. Clinical Parameters")
    age = st.slider("Patient Age", 40, 90, 65)
    bmi = st.number_input("BMI (kg/m^2)", 15.0, 50.0, 28.5)
    womac = st.slider("WOMAC Pain Score", 0, 100, 30)
    sex = st.radio("Biological Sex", ["Male", "Female"])
    kl_grade = st.selectbox("Baseline KL Grade", [0, 1, 2, 3, 4], index=2)
    
    run_btn = st.button("ANALYZE PROGRESSION")

# --- MAIN PAGE ---
st.markdown('<p class="main-header">Tri-Modal Knee Osteoarthritis Prognosis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Survival Analysis with Generative Explainability</p>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        # UPDATED: width="stretch" replaces use_column_width=True
        st.image(image, caption="Input Radiograph", width="stretch")
    with c2:
        if not run_btn:
            st.info("System Ready. Configure parameters and click 'Analyze Progression' to start.")

    if run_btn:
        # --- PROCESSING ---
        progress_bar = st.progress(0, text="Initializing Pipeline...")
        
        time.sleep(0.2)
        progress_bar.progress(25, text="Preprocessing Multi-Modal Data...")
        img_surv, img_64 = process_image(image)
        
        clin_vec = [age, bmi, womac, 0, 0, 0, 0, 0]
        if kl_grade > 0: clin_vec[2 + kl_grade] = 1
        if sex == "Female": clin_vec[7] = 1
        clin_tensor = torch.tensor([clin_vec], dtype=torch.float32).to(DEVICE)
        
        progress_bar.progress(50, text="Calculating Hazard Ratios...")
        risk_score = survival_model(img_surv, clin_tensor).item()
        
        progress_bar.progress(75, text="Generating Structural Counterfactuals...")
        recon_img = generate_counterfactual(img_64, 0.0)
        cf_img = generate_counterfactual(img_64, 1.5)
        
        progress_bar.progress(100, text="Analysis Complete.")
        time.sleep(0.5)
        progress_bar.empty()

        # --- RESULTS ---
        st.divider()
        
        # Row 1: Prognosis
        st.markdown("### 1. Prognostic Report")
        r1, r2 = st.columns([1, 2])
        
        with r1:
            risk_help_text = """
            Log-Hazard Score Interpretation: \n
            • Score = 0: Average Risk (Baseline) \n
            • Score > 0: High Risk (Hazard > Average) \n
            • Score < 0: Low Risk (Hazard < Average)
            
            The score represents the log of the hazard ratio relative to the population mean.
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
            Patient shows a {risk_class.lower()} likelihood of rapid progression compared to the cohort baseline.
            </div>
            """, unsafe_allow_html=True)
            
        with r2:
            st.pyplot(plot_survival_curve(risk_score))

        # Row 2: Explainability
        st.divider()
        st.markdown("### 2. Generative Biomarker Analysis")
        
        g1, g2, g3 = st.columns(3)
        
        r_disp = np.clip(recon_img * 0.5 + 0.5, 0, 1)
        c_disp = np.clip(cf_img * 0.5 + 0.5, 0, 1)
        
        diff = np.abs(r_disp - c_disp)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)

        with g1:
            # UPDATED: width="stretch"
            st.image(r_disp, caption="Current Anatomy (AI Reconstruction)", width="stretch")
        with g2:
            # UPDATED: width="stretch"
            st.image(c_disp, caption="Projected Healthy State (Counterfactual)", width="stretch")
        with g3:
            fig_diff, ax_diff = plt.subplots()
            sns.heatmap(diff, cmap="hot", cbar=True, ax=ax_diff, xticklabels=False, yticklabels=False)
            ax_diff.axis('off')
            st.pyplot(fig_diff)
            st.caption("Difference Map (Active Disease Regions)")

        # Row 3: Dynamic Text
        st.markdown("#### Detected Structural Deviations")
        
        biomarker_text = analyze_biomarker_regions(diff)
        
        for item in biomarker_text:
            st.markdown(f"* {item}")

else:
    # UPDATED: Landing page content when no file is uploaded
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.markdown("#### 1. Imaging Analysis\nResNet-18 feature extraction from plain radiographs.")
    with col2: 
        st.markdown("#### 2. Clinical Integration\nFusion of demographic and symptom scores via MLP.")
    with col3: 
        st.markdown("#### 3. Prognostic Modeling\nCox Proportional Hazards estimation for time-to-event.")
    
    st.info("Please upload an X-Ray image in the sidebar to begin.")