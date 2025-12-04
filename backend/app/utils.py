import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Any
import pandas as pd

# --- CONSTANTS & STATS ---
BIO_STATS = {
    'Bio_COMP': {'mean': 1065.0, 'std': 330.0},
    'Bio_CTXI': {'mean': 350.0, 'std': 180.0},
    'Bio_HA':   {'mean': 45.0, 'std': 35.0},
    'Bio_C2C':  {'mean': 180.0, 'std': 60.0},
    'Bio_CPII': {'mean': 400.0, 'std': 150.0},
    'V00PASE':  {'mean': 130.0, 'std': 65.0},
    'KOOS':     {'mean': 75.0, 'std': 18.0},
    'Stiff':    {'mean': 70.0, 'std': 20.0}
}

# Define transforms once
TRANSFORMS_SURVIVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORMS_GEN = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def normalize(val: float, feature_key: str) -> float:
    stats = BIO_STATS.get(feature_key, {'mean': 0, 'std': 1})
    return (val - stats['mean']) / stats['std']

def preprocess_inputs(image) -> Any:
    # image is PIL Image
    return TRANSFORMS_SURVIVAL(image).unsqueeze(0).to(DEVICE), TRANSFORMS_GEN(image).unsqueeze(0).to(DEVICE)

def generate_heatmap(original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    diff = np.abs(original - modified)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
    return diff

def plot_survival_function(rf_model: Any, patient_df: pd.DataFrame) -> plt.Figure:
    surv_funcs = rf_model.predict_survival_function(patient_df)
    
    # Removed style context to prevent rendering issues on first load
    # plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True, alpha=0.2, linestyle='--') 
        
    for fn in surv_funcs:
        # Convert days to years (x / 365) and Survival to Probability of Surgery (1 - y)
        years = fn.x / 365.0
        prob_surgery = 1.0 - fn(fn.x)
        ax.step(years, prob_surgery, where="post", color='#005EB8', linewidth=3, label="Patient Trajectory")
    
    t = np.linspace(0, 4000, 100)
    t_years = t / 365.0
    base = 1.0 - np.exp(-0.0001 * t) # Inverse of survival for base
    ax.plot(t_years, base, color='#95A5A6', linestyle='--', linewidth=1.5, label="Population Average")
    
    ax.set_title("Projected Probability of Surgery", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Years from Baseline", fontsize=10)
    ax.set_ylabel("Probability of Surgery", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, framealpha=0.9)
        
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def analyze_biomarker_regions(diff_map: np.ndarray) -> List[str]:
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
