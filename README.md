# Tri-Modal Knee Osteoarthritis Prognosis System

A comprehensive system for Knee Osteoarthritis (OA) prognosis using Deep Survival Analysis and Generative Explainability. This project integrates radiographic imaging, clinical parameters, and biochemical markers to predict disease progression and provide visual explanations.

## Features

*   **Multi-Modal Analysis**: Combines X-Ray imaging (DenseNet-121 features), clinical data (demographics, WOMAC, KOOS), and biochemical markers (COMP, CTX-II, etc.).
*   **Deep Survival Analysis**: Utilizes Random Survival Forests to predict progression-free survival probabilities over time.
*   **Generative Explainability**: Uses Diffusion Models (UNet) to generate counterfactual "healthy" versions of patient X-Rays, highlighting structural deviations via heatmaps.
*   **Interactive Dashboard**: User-friendly Web Interface (FastAPI) for data input and result visualization.
*   **Automated Reporting**: Generates detailed PDF reports with prognostic insights and visual analysis.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd oa-survival-model
    ```

2.  **Install uv** (Recommended for faster setup):
    ```bash
    pip install uv
    ```

3.  **Create a virtual environment**:
    ```bash
    uv venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4.  **Install dependencies**:
    ```bash
    uv pip install -r backend/requirements.txt
    ```

## Usage

### Data Processing

To build the mega cohort from raw OAI data:

```bash
python src/build_mega_cohort.py
```

### Running the Web Interface

The system is organized into a `frontend` (Angular) and `backend` (FastAPI) monorepo structure.

1.  **Start Both Services (Recommended)**:
    From the root directory, run:
    ```bash
    npm start
    ```
    This will start both the backend (port 8000) and frontend (port 4200).

2.  **Manual Start (Alternative)**:

    **Backend**:
    ```bash
    cd backend
    ../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

    **Frontend**:
    ```bash
    cd frontend
    npm install # Only first time
    npx ng serve --proxy-config proxy.conf.json
    ```

3.  **Access the App**:
    Open your browser and navigate to `http://localhost:4200`.

### Model Optimization

To optimize the PyTorch models using OpenVINO:

```bash
cd backend
python src/optimize_model.py
```

### Verification

To verify the dataset loading and model upgrade:

```bash
python verify_upgrade.py
```

## Project Structure

```
oa-survival-model/
├── backend/                # Backend Code (FastAPI + Python)
│   ├── app/                # FastAPI application
│   │   └── main.py         # Main application entry point
│   ├── src/                # Source code
│   │   ├── model.py        # Model definitions
│   │   └── ...
│   ├── models/             # Trained model weights
│   ├── data/               # Dataset directory
│   ├── notebooks/          # Jupyter notebooks
│   └── requirements.txt    # Python dependencies
├── frontend/               # Frontend Code (Angular)
│   ├── src/
│   │   ├── app/            # Components & Services
│   │   └── ...
│   └── ...
└── README.md               # Project documentation
```

## Models

*   **Survival Model**: Random Survival Forest (scikit-survival) integrating clinical and biomarker data.
*   **Feature Extractor**: DenseNet-121 (or similar) for radiographic feature extraction.
*   **Generative Model**: Latent Diffusion Model (UNet + Semantic Encoder) for generating counterfactual images.

## License