"""
Feature Registry for OAI Tri-Modal Survival Model
Centralized definition of all features by modality

Based on comprehensive analysis of OAICompleteData_ASCII (157 files, 36.5M rows)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class FeatureLevel(Enum):
    """Whether the feature is patient-level or knee-level"""
    PATIENT = "patient"
    KNEE = "knee"


class FeatureSource(Enum):
    """Source file category for the feature"""
    CLINICAL = "AllClinical00.txt"
    SUBJECT_CHAR = "SubjectChar00.txt"
    OUTCOMES = "OUTCOMES99.txt"
    XRAY = "KXR_SQ_BU00.txt"
    MRI_MOAKS = "kMRI_FNIH_SQ_MOAKS_BICL00.txt"
    MRI_QCART = "kMRI_FNIH_QCart_Chondrometrics00.txt"
    BIOSPEC = "Biospec_FNIH_Labcorp00.txt"
    BIOMARKERS = "Biomarkers00.txt"
    ENROLLEES = "Enrollees.txt"


@dataclass
class FeatureDefinition:
    """Definition of a single feature"""
    name: str                          # Column name in data
    display_name: str                  # Human-readable name
    modality: str                      # clinical, imaging, biomarker
    source: FeatureSource              # Source file
    level: FeatureLevel                # Patient or knee level
    data_type: str                     # numeric, categorical, date
    description: str = ""
    requires_cleaning: bool = True     # Whether OAI value needs parsing (e.g., "1: Male" -> 1)
    imputation_strategy: str = "median" # median, mode, zero, drop


class FeatureRegistry:
    """
    Central registry of all features for the OAI Tri-Modal Survival Model.
    
    Features are organized by modality:
    - Clinical: Demographics, symptoms, function scores
    - Imaging: X-ray (KL Grade), MRI (MOAKS, cartilage)
    - Biomarker: Serum and urine markers from FNIH sub-cohort
    """
    
    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._register_all_features()
    
    def _register_all_features(self):
        """Register all known features from OAI dataset."""
        
        # =============================================
        # CLINICAL MODALITY - Demographics & Baseline
        # =============================================
        self._register_clinical_features()
        
        # =============================================
        # IMAGING MODALITY - X-Ray and MRI
        # =============================================
        self._register_imaging_features()
        
        # =============================================
        # BIOMARKER MODALITY - Serum & Urine
        # =============================================
        self._register_biomarker_features()
        
        # =============================================
        # TARGET VARIABLES - Outcomes
        # =============================================
        self._register_target_features()
    
    def _register_clinical_features(self):
        """Clinical features from AllClinical00.txt and SubjectChar00.txt"""
        
        # Demographics
        demographics = [
            FeatureDefinition("V00AGE", "Age", "clinical", FeatureSource.SUBJECT_CHAR, 
                            FeatureLevel.PATIENT, "numeric", "Age at baseline visit"),
            FeatureDefinition("P02SEX", "Sex", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "categorical", "1=Male, 2=Female"),
            FeatureDefinition("P01BMI", "BMI", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "numeric", "Body Mass Index at baseline"),
            FeatureDefinition("P01RACE", "Race", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "categorical", "Race/ethnicity"),
            FeatureDefinition("V00EDCV", "Education", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "categorical", "Education level"),
            FeatureDefinition("V00INCOME", "Income", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "categorical", "Household income"),
        ]
        
        # Symptom Scores
        symptoms = [
            # WOMAC - Western Ontario and McMaster Universities Osteoarthritis Index
            FeatureDefinition("V00WOMTSR", "WOMAC_Total_Right", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Total Score - Right Knee"),
            FeatureDefinition("V00WOMTSL", "WOMAC_Total_Left", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Total Score - Left Knee"),
            FeatureDefinition("V00WOMPNR", "WOMAC_Pain_Right", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Pain Score - Right"),
            FeatureDefinition("V00WOMPNL", "WOMAC_Pain_Left", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Pain Score - Left"),
            FeatureDefinition("V00WOMSTFR", "WOMAC_Stiffness_Right", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Stiffness Score - Right"),
            FeatureDefinition("V00WOMSTFL", "WOMAC_Stiffness_Left", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Stiffness Score - Left"),
            FeatureDefinition("V00WOMFNR", "WOMAC_Function_Right", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Function Score - Right"),
            FeatureDefinition("V00WOMFNL", "WOMAC_Function_Left", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "WOMAC Function Score - Left"),
            
            # KOOS - Knee Injury and Osteoarthritis Outcome Score
            FeatureDefinition("V00KOOSYMR", "KOOS_Symptoms_Right", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "KOOS Symptoms - Right"),
            FeatureDefinition("V00KOOSYML", "KOOS_Symptoms_Left", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "numeric", "KOOS Symptoms - Left"),
            FeatureDefinition("V00KOOSQOL", "KOOS_QOL", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.PATIENT, "numeric", "KOOS Quality of Life - Strong TKR predictor"),
        ]
        
        # Physical Function
        function = [
            FeatureDefinition("V00PASE", "PASE", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "numeric", "Physical Activity Scale for Elderly"),
            FeatureDefinition("V00CHMWKPCE", "Chair_Stand_Pace", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.PATIENT, "numeric", "Chair stand pace"),
            FeatureDefinition("V0020MWLKP", "Walk_Pace_20m", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.PATIENT, "numeric", "20m walk pace"),
        ]
        
        # Medical History
        medical = [
            FeatureDefinition("V00NSAIDRX", "NSAID_Use", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.PATIENT, "categorical", "NSAID medication use"),
            FeatureDefinition("P02FAMHXKR", "Family_Hx_KR", "clinical", FeatureSource.SUBJECT_CHAR,
                            FeatureLevel.PATIENT, "categorical", "Family history of knee replacement"),
            FeatureDefinition("V00INJL", "Previous_Injury_Left", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "categorical", "Previous injury left knee"),
            FeatureDefinition("V00INJR", "Previous_Injury_Right", "clinical", FeatureSource.CLINICAL,
                            FeatureLevel.KNEE, "categorical", "Previous injury right knee"),
        ]
        
        for f in demographics + symptoms + function + medical:
            self._features[f.name] = f
    
    def _register_imaging_features(self):
        """X-ray and MRI features"""
        
        # X-Ray Features (from KXR_SQ_BU00.txt)
        xray = [
            FeatureDefinition("V00XRKL", "KL_Grade", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "categorical", 
                            "Kellgren-Lawrence Grade: 0=Normal, 1=Doubtful, 2=Minimal, 3=Moderate, 4=Severe"),
            FeatureDefinition("V00XROSTL", "Osteophyte_Lat", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "numeric", "Lateral osteophyte grade"),
            FeatureDefinition("V00XROSTM", "Osteophyte_Med", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "numeric", "Medial osteophyte grade"),
            FeatureDefinition("V00XRJSL", "JSN_Lateral", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "numeric", "Lateral joint space narrowing"),
            FeatureDefinition("V00XRJSM", "JSN_Medial", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "numeric", "Medial joint space narrowing"),
            FeatureDefinition("V00XRSCL", "Sclerosis_Lat", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "numeric", "Lateral sclerosis"),
            FeatureDefinition("V00XRSCM", "Sclerosis_Med", "imaging", FeatureSource.XRAY,
                            FeatureLevel.KNEE, "numeric", "Medial sclerosis"),
        ]
        
        # MRI Features - MOAKS Scoring (from kMRI_FNIH_SQ_MOAKS_BICL00.txt)
        mri_moaks = [
            FeatureDefinition("V00MACLBML", "MRI_BML_Score", "imaging", FeatureSource.MRI_MOAKS,
                            FeatureLevel.KNEE, "numeric", "Bone Marrow Lesion Score - Strong TKR predictor"),
            FeatureDefinition("V00MACLCYS", "MRI_Cyst_Score", "imaging", FeatureSource.MRI_MOAKS,
                            FeatureLevel.KNEE, "numeric", "Subchondral Cyst Score"),
            FeatureDefinition("V00MACLCAR", "MRI_Cartilage_Score", "imaging", FeatureSource.MRI_MOAKS,
                            FeatureLevel.KNEE, "numeric", "Cartilage lesion score"),
            FeatureDefinition("V00MACLMEN", "MRI_Meniscus_Score", "imaging", FeatureSource.MRI_MOAKS,
                            FeatureLevel.KNEE, "numeric", "Meniscal pathology score"),
            FeatureDefinition("V00MACLOSP", "MRI_Osteophyte_Score", "imaging", FeatureSource.MRI_MOAKS,
                            FeatureLevel.KNEE, "numeric", "MRI Osteophyte score"),
            FeatureDefinition("V00MACLEFF", "MRI_Effusion_Score", "imaging", FeatureSource.MRI_MOAKS,
                            FeatureLevel.KNEE, "numeric", "Joint effusion/synovitis"),
        ]
        
        # MRI Features - Quantitative Cartilage (from kMRI_FNIH_QCart_Chondrometrics00.txt)
        mri_qcart = [
            FeatureDefinition("V00WMTMTH", "Medial_Tibial_Thickness", "imaging", FeatureSource.MRI_QCART,
                            FeatureLevel.KNEE, "numeric", "Medial tibial cartilage thickness (mm)"),
            FeatureDefinition("V00WLTMTH", "Lateral_Tibial_Thickness", "imaging", FeatureSource.MRI_QCART,
                            FeatureLevel.KNEE, "numeric", "Lateral tibial cartilage thickness (mm)"),
            FeatureDefinition("V00WMFMTH", "Medial_Femoral_Thickness", "imaging", FeatureSource.MRI_QCART,
                            FeatureLevel.KNEE, "numeric", "Medial femoral cartilage thickness (mm)"),
            FeatureDefinition("V00WLFMTH", "Lateral_Femoral_Thickness", "imaging", FeatureSource.MRI_QCART,
                            FeatureLevel.KNEE, "numeric", "Lateral femoral cartilage thickness (mm)"),
            FeatureDefinition("V00WMTMVC", "Medial_Tibial_Volume", "imaging", FeatureSource.MRI_QCART,
                            FeatureLevel.KNEE, "numeric", "Medial tibial cartilage volume"),
            FeatureDefinition("V00WLTMVC", "Lateral_Tibial_Volume", "imaging", FeatureSource.MRI_QCART,
                            FeatureLevel.KNEE, "numeric", "Lateral tibial cartilage volume"),
        ]
        
        for f in xray + mri_moaks + mri_qcart:
            self._features[f.name] = f
    
    def _register_biomarker_features(self):
        """
        Biomarker features from Biospec_FNIH_Labcorp00.txt
        Note: Only available for ~600 patients in FNIH sub-cohort
        """
        
        # Serum Biomarkers - Cartilage Degradation
        serum_cartilage = [
            FeatureDefinition("V00Serum_C1_2C_lc", "Bio_C1_2C", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Collagen type I/II degradation marker"),
            FeatureDefinition("V00Serum_C2C_lc", "Bio_C2C", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Type II Collagen Cleavage - cartilage degradation"),
            FeatureDefinition("V00Serum_CPII_lc", "Bio_CPII", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Procollagen II C-Propeptide - cartilage synthesis"),
            FeatureDefinition("V00Serum_Comp_lc", "Bio_COMP", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Cartilage Oligomeric Matrix Protein - cartilage breakdown"),
            FeatureDefinition("V00Serum_CS846_lc", "Bio_CS846", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Chondroitin sulfate epitope - aggrecan synthesis"),
            FeatureDefinition("V00Serum_COLL2_1_NO2_lc", "Bio_COLL2_1_NO2", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Nitrated type II collagen"),
        ]
        
        # Serum Biomarkers - Bone Turnover
        serum_bone = [
            FeatureDefinition("V00Serum_CTXI_lc", "Bio_CTXI", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "C-Telopeptide of Type I Collagen - bone resorption"),
            FeatureDefinition("V00Serum_NTXI_lc", "Bio_NTXI", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "N-Telopeptide of Type I Collagen - bone turnover"),
            FeatureDefinition("V00Serum_PIIANP_lc", "Bio_PIIANP", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Type IIA Procollagen N-Propeptide"),
        ]
        
        # Serum Biomarkers - Inflammation
        serum_inflammation = [
            FeatureDefinition("V00Serum_HA_lc", "Bio_HA", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Hyaluronic Acid - inflammation/synovitis marker"),
            FeatureDefinition("V00Serum_MMP_3_lc", "Bio_MMP3", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Matrix Metalloproteinase-3 - cartilage destruction"),
        ]
        
        # Urine Biomarkers
        urine = [
            FeatureDefinition("V00Urine_CTXII_lc", "Bio_uCTXII", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Urinary CTX-II - cartilage degradation"),
            FeatureDefinition("V00Urine_C1_2C_lc", "Bio_uC1_2C", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Urinary C1,2C"),
            FeatureDefinition("V00Urine_C2C_lc", "Bio_uC2C", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Urinary C2C"),
            FeatureDefinition("V00Urine_NTXI_lc", "Bio_uNTXI", "biomarker", FeatureSource.BIOSPEC,
                            FeatureLevel.PATIENT, "numeric", "Urinary NTX-I - bone resorption"),
        ]
        
        for f in serum_cartilage + serum_bone + serum_inflammation + urine:
            self._features[f.name] = f
    
    def _register_target_features(self):
        """Target variables from OUTCOMES99.txt"""
        
        targets = [
            FeatureDefinition("V99ERKDATE", "TKR_Date_Right", "target", FeatureSource.OUTCOMES,
                            FeatureLevel.KNEE, "date", "Total Knee Replacement date - Right"),
            FeatureDefinition("V99ELKDATE", "TKR_Date_Left", "target", FeatureSource.OUTCOMES,
                            FeatureLevel.KNEE, "date", "Total Knee Replacement date - Left"),
            FeatureDefinition("V99EDDDATE", "Death_Date", "target", FeatureSource.OUTCOMES,
                            FeatureLevel.PATIENT, "date", "Date of death"),
            FeatureDefinition("V99RNTCNT", "Last_Contact", "target", FeatureSource.OUTCOMES,
                            FeatureLevel.PATIENT, "categorical", "Last contact visit number"),
        ]
        
        for f in targets:
            self._features[f.name] = f
    
    # =============================================
    # Public API
    # =============================================
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature definition by column name."""
        return self._features.get(name)
    
    def get_features_by_modality(self, modality: str) -> List[FeatureDefinition]:
        """Get all features for a given modality."""
        return [f for f in self._features.values() if f.modality == modality]
    
    def get_features_by_source(self, source: FeatureSource) -> List[FeatureDefinition]:
        """Get all features from a given source file."""
        return [f for f in self._features.values() if f.source == source]
    
    def get_knee_level_features(self) -> List[FeatureDefinition]:
        """Get all knee-level features."""
        return [f for f in self._features.values() if f.level == FeatureLevel.KNEE]
    
    def get_patient_level_features(self) -> List[FeatureDefinition]:
        """Get all patient-level features."""
        return [f for f in self._features.values() if f.level == FeatureLevel.PATIENT]
    
    @property
    def clinical(self) -> List[str]:
        """Clinical feature column names."""
        return [f.name for f in self.get_features_by_modality("clinical")]
    
    @property
    def imaging(self) -> List[str]:
        """Imaging feature column names."""
        return [f.name for f in self.get_features_by_modality("imaging")]
    
    @property
    def biomarker(self) -> List[str]:
        """Biomarker feature column names."""
        return [f.name for f in self.get_features_by_modality("biomarker")]
    
    @property
    def all_features(self) -> List[str]:
        """All feature column names."""
        return list(self._features.keys())
    
    def summary(self) -> Dict:
        """Get summary statistics of registered features."""
        return {
            "total": len(self._features),
            "clinical": len(self.clinical),
            "imaging": len(self.imaging),
            "biomarker": len(self.biomarker),
            "knee_level": len(self.get_knee_level_features()),
            "patient_level": len(self.get_patient_level_features()),
        }


# Singleton instance
_registry = None

def get_registry() -> FeatureRegistry:
    """Get the singleton feature registry."""
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry


if __name__ == "__main__":
    # Test the registry
    reg = get_registry()
    summary = reg.summary()
    print("Feature Registry Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    print("\nClinical Features:")
    for f in reg.get_features_by_modality("clinical")[:5]:
        print(f"  {f.name}: {f.display_name} ({f.level.value})")
    
    print("\nBiomarker Features:")
    for f in reg.get_features_by_modality("biomarker")[:5]:
        print(f"  {f.name}: {f.display_name}")
