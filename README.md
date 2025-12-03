# Tri-Modal Knee Osteoarthritis Prognosis System

A comprehensive system for Knee Osteoarthritis (OA) prognosis using Deep Survival Analysis and Generative Explainability. This project integrates radiographic imaging, clinical parameters, and biochemical markers to predict disease progression and provide visual explanations.

## Features

*   **Multi-Modal Analysis**: Combines X-Ray imaging (ResNet-18 features), clinical data (demographics, WOMAC, KOOS), and biochemical markers (COMP, CTX-II, etc.).
*   **Deep Survival Analysis**: Utilizes Random Survival Forests to predict progression-free survival probabilities over time.
*   **Generative Explainability**: Uses Diffusion Models (UNet) to generate counterfactual "healthy" versions of patient X-Rays, highlighting structural deviations via heatmaps.
*   **Interactive Dashboard**: User-friendly Streamlit interface for data input and result visualization.
*   **Automated Reporting**: Generates detailed PDF reports with prognostic insights and visual analysis.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd oa-survival-model
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Dashboard

To start the web application:

```bash
streamlit run app/app.py
```

The application will open in your default web browser.

### Model Optimization

To optimize the models for deployment (quantization and FP16 conversion):

```bash
python src/optimize_models.py
```

## Project Structure

```
oa-survival-model/
├── app/                    # Streamlit application
│   └── app.py              # Main application entry point
├── src/                    # Source code
│   ├── model.py            # Model definitions (WideAndDeep, SemanticEncoder)
│   ├── dataset.py          # Dataset handling
│   ├── optimize_models.py  # Model optimization script
│   └── ...                 # Data processing scripts
├── models/                 # Trained model files
│   ├── random_forest_survival.joblib
│   ├── semantic_encoder.pth
│   └── diffusion_unet.pth
├── notebooks/              # Jupyter notebooks for research & training
├── data/                   # Data directory
└── requirements.txt        # Python dependencies
```

## Models

*   **Survival Model**: Random Survival Forest (scikit-survival) integrating clinical and biomarker data.
*   **Feature Extractor**: ResNet-18 (or similar) for radiographic feature extraction.
*   **Generative Model**: Latent Diffusion Model (UNet + Semantic Encoder) for generating counterfactual images.

## License