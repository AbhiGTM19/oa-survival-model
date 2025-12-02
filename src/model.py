import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. Tri-Modal Survival Model Architecture (Updated for Track 3) ---
class WideAndDeepSurvivalModel(nn.Module):
    def __init__(self, wide_input_dim, bio_input_dim=5): # <--- Note the new argument
        super(WideAndDeepSurvivalModel, self).__init__()
        
        # 1. Image Pathway (Deep)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_out_dim = 512
        
        # 2. Clinical Pathway (Wide)
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

        # 3. Biomarker Pathway (New Track 3)
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.bio_out_dim = 8
        
        # 4. Fusion Head
        fusion_input_dim = self.image_out_dim + self.wide_out_dim + self.bio_out_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, image, clinical_data, bio_data):
        img_features = self.image_encoder(image).view(image.size(0), -1)
        clinical_features = self.wide_encoder(clinical_data)
        bio_features = self.bio_encoder(bio_data)
        
        # Concatenate all 3
        combined = torch.cat((img_features, clinical_features, bio_features), dim=1)
        return self.fusion_layer(combined)

# --- 2. Generative Encoder (Track 2) ---
class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim=256): 
        super(SemanticEncoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Grayscale Fix
        original_first_layer = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:] = original_first_layer.weight.sum(dim=1, keepdim=True) / 3.0
            
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        z = self.projection(x)
        return z

# --- 2. Generative Encoder Architecture (Track 2) ---
class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim=256): 
        super(SemanticEncoder, self).__init__()
        # Standard ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Grayscale Fix (3 channels -> 1 channel)
        original_first_layer = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Average weights to preserve pre-training
        with torch.no_grad():
            resnet.conv1.weight[:] = original_first_layer.weight.sum(dim=1, keepdim=True) / 3.0
            
        # Bottleneck
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        z = self.projection(x)
        return z