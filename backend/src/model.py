import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. Tri-Modal Survival Model (Upgraded to DenseNet-121) ---
class WideAndDeepSurvivalModel(nn.Module):
    def __init__(self, wide_input_dim, bio_input_dim=5): 
        super(WideAndDeepSurvivalModel, self).__init__()
        
        # --- DEEP PATHWAY (Image) ---
        # UPGRADE: Using DenseNet-121 instead of ResNet-18
        # DenseNet is often superior for medical texture analysis (X-rays)
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # DenseNet Features Output: [Batch, 1024, 7, 7]
        self.image_encoder = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Compress to [Batch, 1024, 1, 1]
        self.image_out_dim = 1024 # DenseNet-121 outputs 1024 channels
        
        # --- WIDE PATHWAY (Clinical) ---
        self.wide_encoder = nn.Sequential(
            nn.Linear(wide_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.wide_out_dim = 16

        # --- BIOMARKER PATHWAY ---
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.bio_out_dim = 8
        
        # --- FUSION HEAD ---
        fusion_input_dim = self.image_out_dim + self.wide_out_dim + self.bio_out_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, image, clinical_data, bio_data):
        # Image Feature Extraction
        x = self.image_encoder(image)
        x = self.pool(x)
        img_features = x.view(x.size(0), -1) # Flatten to [Batch, 1024]
        
        # Tabular Feature Extraction
        clinical_features = self.wide_encoder(clinical_data)
        bio_features = self.bio_encoder(bio_data)
        
        # Fusion
        combined = torch.cat((img_features, clinical_features, bio_features), dim=1)
        return self.fusion_layer(combined)

# --- 2. Generative Encoder (Keep ResNet-18 for Efficiency) ---
class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim=256): 
        super(SemanticEncoder, self).__init__()
        
        # UPGRADE: Using DenseNet-121 (Matches Survival Model)
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # --- GRAYSCALE FIX FOR DENSENET ---
        # DenseNet's first layer is named 'features.conv0' (unlike ResNet's 'conv1')
        original_first_layer = densenet.features.conv0
        
        # Create new 1-channel layer
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Average weights to preserve pre-training
        with torch.no_grad():
            densenet.features.conv0.weight[:] = original_first_layer.weight.sum(dim=1, keepdim=True) / 3.0
            
        # Remove classifier (DenseNet output is 'classifier')
        # We keep 'features' (the CNN part)
        self.features = densenet.features
        
        # DenseNet-121 outputs 1024 channels at the end
        # DenseNet-121 outputs 1024 channels at the end
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Match checkpoint: proj.0 (Linear), proj.1 (BatchNorm/LayerNorm)
        self.proj = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, x):
        x = self.features(x)            # [Batch, 1024, 7, 7]
        x = self.global_pool(x)         # [Batch, 1024, 1, 1]
        x = x.view(x.size(0), -1)       # [Batch, 1024]
        z = self.proj(x)                # [Batch, 256]
        return z