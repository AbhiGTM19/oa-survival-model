/**
 * TypeScript interfaces for the OA Survival Model API
 */

// ===== Survival Curve =====
export interface SurvivalPoint {
    x: number;  // Years from baseline
    y: number;  // Probability of surgery
}

// ===== Generated Images =====
export interface PredictionImages {
    original: string;       // Base64 encoded PNG
    counterfactual: string; // Base64 encoded PNG
    heatmap: string;        // Base64 encoded PNG
}

// ===== Patient Data =====
export interface PatientData {
    ID: string;
    Age: number;
    Sex: string;
    BMI: number;
    'KL Grade': number;
    WOMAC: number;
    COMP: number;
    CTX: number;
}

// ===== Risk Classification =====
export type RiskClass = 'Low' | 'Moderate' | 'High';

// ===== Main Prediction Result =====
export interface PredictionResult {
    risk_score: number;
    risk_class: RiskClass;
    survival_curve: SurvivalPoint[];
    images: PredictionImages;
    findings: string[];
    patient_data: PatientData;
}

// ===== Report Generation =====
export interface RiskAnalysis {
    risk_score: number;
    risk_class: RiskClass;
    prob_5yr: string;
}

export interface ReportRequest {
    patient_data: PatientData;
    risk_analysis: RiskAnalysis;
    findings: string[];
    images: PredictionImages;
}

// ===== Form Input =====
export interface PatientFormInput {
    patientId: string;
    age: number;
    sex: 'Male' | 'Female';
    bmi: number;
    kl_grade: number;
    womac: number;
    pase: number;
    koos: number;
    stiffness: number;
    nsaid: 'Yes' | 'No';
    bio_comp: number;
    bio_ctx: number;
    bio_ha: number;
    bio_c2c: number;
    bio_cpii: number;
    mri_bml: number;
    mri_cyst: number;
}
