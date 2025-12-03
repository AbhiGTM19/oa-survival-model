import torch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from dataset import TriModalDataset
from model import WideAndDeepSurvivalModel

def verify_dataset_and_model():
    print("🚀 Starting Verification...")

    # 1. Mock Dataframe
    print("   Creating Mock DataFrame...")
    data = {
        'ID': [101, 102],
        'Knee_Side': ['R', 'L'],
        'event': [1, 0],
        'time_to_event': [100, 200],
        # Clinical
        'Age': [60, 70], 'BMI': [25.0, 30.0], 'WOMAC_Score': [10, 20],
        'KL_Grade_1.0': [1, 0], 'KL_Grade_2.0': [0, 1], 'KL_Grade_3.0': [0, 0], 'KL_Grade_4.0': [0, 0],
        'Sex_2': [0, 1],
        'V00KOOSQOL': [80, 70], 'V00PASE': [150, 100], 'MRI_BML_Score': [1, 2],
        'Medial_Tibial_Thickness': [2.5, 2.0], 'Lateral_Tibial_Thickness': [2.5, 2.0],
        'Education': [1, 2], 'Income': [1, 2],
        # Biomarkers
        'Bio_COMP': [100, 110], 'Bio_CTXI': [0.5, 0.6], 'Bio_HA': [20, 30], 'Bio_C2C': [50, 60], 'Bio_CPII': [300, 400]
    }
    df = pd.DataFrame(data)

    # 2. Test Dataset
    print("   Testing TriModalDataset...")
    # Create a dummy image directory
    os.makedirs('dummy_images', exist_ok=True)
    # Create dummy images if not exist (sandbox mode uses glob)
    if not os.path.exists('dummy_images/test.png'):
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        img.save('dummy_images/test.png')

    dataset = TriModalDataset(df, 'dummy_images', mode='sandbox')
    
    try:
        image, clin, bio, event, time = dataset[0]
        print(f"   ✅ Dataset Item 0 Loaded.")
        print(f"      Image Shape: {image.shape}")
        print(f"      Clinical Shape: {clin.shape}")
        print(f"      Bio Shape: {bio.shape}")
        
        expected_clin_dim = 17 # 8 original + 3 platinum + 2 cartilage + 2 demo + 2 extra? 
        # Let's count: 
        # Age, BMI, WOMAC, KL1, KL2, KL3, KL4, Sex2 = 8
        # KOOS, PASE, MRI = 3
        # Medial, Lateral = 2
        # Edu, Inc = 2
        # Total = 8 + 3 + 2 + 2 = 15?
        # Wait, let's check dataset.py source again.
        # clin_cols = [Age, BMI, WOMAC, KL1, KL2, KL3, KL4, Sex2, KOOS, PASE, MRI, Medial, Lateral, Edu, Inc]
        # 3 + 4 + 1 + 3 + 2 + 2 = 15.
        # Why did I say 17 in the plan?
        # Ah, I might have miscounted or included something else.
        # Let's verify the actual length returned.
        
        if clin.shape[0] != 15:
             print(f"   ⚠️ WARNING: Clinical dimension is {clin.shape[0]}, expected 15 based on list.")
             # If I put 17 in the notebook, I need to correct it to 15 or find the missing 2.
             # Let's check the previous dataset.py content I wrote.
             # 'V00KOOSQOL', 'V00PASE', 'MRI_BML_Score', 'Medial_Tibial_Thickness', 'Lateral_Tibial_Thickness', 'Education', 'Income'
             # + 'Age', 'BMI', 'WOMAC_Score', 'KL_Grade_1.0', 'KL_Grade_2.0', 'KL_Grade_3.0', 'KL_Grade_4.0', 'Sex_2'
             # 7 new + 8 old = 15.
             # So 17 was wrong. I need to fix the notebook to use 15.
        
    except Exception as e:
        print(f"   ❌ Dataset Failed: {e}")
        return

    # 3. Test Model
    print("   Testing WideAndDeepSurvivalModel...")
    # Use the actual dimension from dataset
    real_wide_dim = clin.shape[0]
    model = WideAndDeepSurvivalModel(wide_input_dim=real_wide_dim, bio_input_dim=5)
    
    # Create batch
    images = image.unsqueeze(0) # [1, 3, 224, 224] (Dataset returns PIL? No, transforms usually convert to tensor. But here no transform passed, so PIL)
    # Wait, dataset returns PIL if no transform. Model expects Tensor.
    # Let's apply a basic transform in the test
    from torchvision import transforms
    t = transforms.Compose([transforms.ToTensor()])
    dataset.transform = t
    image, clin, bio, event, time = dataset[0]
    images = image.unsqueeze(0)
    
    clins = clin.unsqueeze(0)
    bios = bio.unsqueeze(0)
    
    try:
        output = model(images, clins, bios)
        print(f"   ✅ Model Forward Pass Successful. Output Shape: {output.shape}")
    except Exception as e:
        print(f"   ❌ Model Failed: {e}")

    # Cleanup
    import shutil
    shutil.rmtree('dummy_images')
    print("✅ Verification Complete.")

if __name__ == "__main__":
    verify_dataset_and_model()
