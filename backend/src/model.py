import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for feature fusion.
    Learns to weight features from each modality based on their relevance.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        # query: [B, 1, D], key_value: [B, N, D]
        attn_out, weights = self.attention(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out)), weights


# --- 1. Tri-Modal Survival Model (Upgraded with Attention-Based Fusion) ---
class WideAndDeepSurvivalModel(nn.Module):
    """
    Tri-modal survival model combining:
    - Image features (DenseNet-121)
    - Clinical features (MLP)
    - Biomarker features (MLP)
    
    Supports two fusion modes:
    - 'concat': Simple concatenation (default, faster)
    - 'attention': Cross-modal attention (better performance, requires 64-dim outputs)
    """
    def __init__(self, wide_input_dim, bio_input_dim=5, fusion_mode='concat'): 
        super(WideAndDeepSurvivalModel, self).__init__()
        self.fusion_mode = fusion_mode
        
        # --- DEEP PATHWAY (Image) ---
        # DenseNet-121: Superior for medical texture analysis (X-rays)
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.image_encoder = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_out_dim = 1024  # DenseNet-121 outputs 1024 channels
        
        # --- WIDE PATHWAY (Clinical) ---
        self.wide_encoder = nn.Sequential(
            nn.Linear(wide_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.wide_out_dim = 32

        # --- BIOMARKER PATHWAY ---
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.bio_out_dim = 16
        
        # --- FUSION MECHANISM ---
        if fusion_mode == 'attention':
            # Project all modalities to same dimension for attention
            self.embed_dim = 64
            self.img_proj = nn.Linear(self.image_out_dim, self.embed_dim)
            self.wide_proj = nn.Linear(self.wide_out_dim, self.embed_dim)
            self.bio_proj = nn.Linear(self.bio_out_dim, self.embed_dim)
            
            # Cross-modal attention: query clinical+bio, attend to image
            self.cross_attn = CrossModalAttention(self.embed_dim, num_heads=4)
            
            # Self-attention for final fusion
            self.self_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(self.embed_dim)
            
            fusion_input_dim = self.embed_dim * 3  # 3 modalities
        else:
            # Simple concatenation
            fusion_input_dim = self.image_out_dim + self.wide_out_dim + self.bio_out_dim
        
        # --- FUSION HEAD ---
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
        img_features = x.view(x.size(0), -1)  # [Batch, 1024]
        
        # Tabular Feature Extraction
        clinical_features = self.wide_encoder(clinical_data)  # [Batch, 32]
        bio_features = self.bio_encoder(bio_data)  # [Batch, 16]
        
        if self.fusion_mode == 'attention':
            # Project to common embedding space
            img_embed = self.img_proj(img_features)  # [B, 64]
            wide_embed = self.wide_proj(clinical_features)  # [B, 64]
            bio_embed = self.bio_proj(bio_features)  # [B, 64]
            
            # Stack as sequence: [B, 3, 64]
            modality_tokens = torch.stack([img_embed, wide_embed, bio_embed], dim=1)
            
            # Self-attention across modalities
            attn_out, _ = self.self_attn(modality_tokens, modality_tokens, modality_tokens)
            attn_out = self.attn_norm(modality_tokens + attn_out)
            
            # Flatten all modality features
            combined = attn_out.view(attn_out.size(0), -1)  # [B, 192]
        else:
            # Simple concatenation
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
        z = torch.clamp(z, -5.0, 5.0)   # Safety clamp (matches training)
        return z