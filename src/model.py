import torch
import torch.nn as nn
import torchvision.models as models

class WideAndDeepSurvivalModel(nn.Module):
    def __init__(self, wide_input_dim):
        """
        Args:
            wide_input_dim (int): Number of clinical features (e.g., 8).
        """
        super(WideAndDeepSurvivalModel, self).__init__()
        
        # --- 1. The Deep Component (Image Encoder) ---
        # We use ResNet-18 for a good balance of speed and performance.
        # We load pre-trained weights to leverage knowledge from ImageNet.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer (fc)
        # ResNet-18 outputs 512 features before the final layer.
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_out_dim = 512
        
        # --- 2. The Wide Component (Clinical Encoder) ---
        # A simple MLP to process the tabular data
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
        
        # --- 3. The Fusion Head (Combination) ---
        # We concatenate the image and clinical features
        fusion_input_dim = self.image_out_dim + self.wide_out_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Final output: Single continuous risk score
            nn.Linear(64, 1)
        )

    def forward(self, image, clinical_data):
        """
        Args:
            image: Tensor of shape (Batch, 3, 224, 224)
            clinical_data: Tensor of shape (Batch, wide_input_dim)
        """
        # 1. Image Pathway
        img_features = self.image_encoder(image)
        # Flatten: (Batch, 512, 1, 1) -> (Batch, 512)
        img_features = img_features.view(img_features.size(0), -1)
        
        # 2. Clinical Pathway
        clinical_features = self.wide_encoder(clinical_data)
        
        # 3. Fusion
        combined = torch.cat((img_features, clinical_features), dim=1)
        risk_score = self.fusion_layer(combined)
        
        return risk_score

if __name__ == "__main__":
    # Sanity Check: Test the model with random data
    print("Testing Model Architecture...")
    
    # Random Inputs
    dummy_img = torch.randn(2, 3, 224, 224) # Batch of 2 images
    dummy_data = torch.randn(2, 8)          # Batch of 2 patients with 8 features
    
    # Initialize Model
    model = WideAndDeepSurvivalModel(wide_input_dim=8)
    
    # Forward Pass
    output = model(dummy_img, dummy_data)
    
    print(f"Image Input Shape: {dummy_img.shape}")
    print(f"Clinical Input Shape: {dummy_data.shape}")
    print(f"Model Output Shape: {output.shape}") # Should be [2, 1]
    print("Success! Model is ready.")