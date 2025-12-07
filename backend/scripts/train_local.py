"""
Local Training Script for Tri-Modal Survival Model

Fast iteration with:
- Sampled data (configurable, default 500)
- Sandbox images (random assignment)
- Reduced epochs (5)
- Full metrics logging

Usage:
    python train_local.py
    python train_local.py --epochs 10 --sample-size 1000
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import WideAndDeepSurvivalModel
from src.dataset import TriModalDataset
from config import get_config, FEATURE_SETS

# Try to import survival metrics
try:
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("⚠️ sksurv not installed - C-index will use basic implementation")

# Try to import torchsurv loss
try:
    from torchsurv.loss.cox import neg_partial_log_likelihood as base_cox_loss
    HAS_TORCHSURV = True
except ImportError:
    HAS_TORCHSURV = False
    base_cox_loss = None


def get_device():
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cox_loss(risk, events, times, device):
    """
    Cox proportional hazards negative partial log-likelihood.
    Handles MPS compatibility issues.
    """
    events_bool = events.bool()
    
    if HAS_TORCHSURV and base_cox_loss is not None:
        # MPS Fix: Move to CPU for logcumsumexp
        if risk.device.type == 'mps':
            return base_cox_loss(risk.cpu(), events_bool.cpu(), times.cpu()).to(device)
        return base_cox_loss(risk, events_bool, times)
    else:
        # Fallback implementation
        order = torch.argsort(times, descending=True)
        risk = risk[order]
        events_sorted = events_bool[order]
        
        risk_cpu = risk.cpu()
        log_cumsum = torch.logcumsumexp(risk_cpu, dim=0).to(risk.device)
        
        if events_sorted.sum() > 0:
            return -torch.sum(events_sorted.float() * (risk - log_cumsum)) / events_sorted.sum()
        return torch.tensor(0.0, requires_grad=True, device=device)


def compute_c_index(risk_scores, events, times):
    """Compute concordance index."""
    if HAS_SKSURV:
        events_bool = events.astype(bool)
        c_index = concordance_index_censored(events_bool, times, risk_scores)[0]
    else:
        # Basic implementation
        n = len(risk_scores)
        concordant = 0
        comparable = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] and times[i] < times[j]:
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif events[j] and times[j] < times[i]:
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
        
        c_index = concordant / comparable if comparable > 0 else 0.5
    
    return c_index


def prepare_data(config, sample_size=None):
    """Load and prepare data for training."""
    parquet_path = Path(__file__).parent.parent / 'data' / 'processed' / 'OAI_mega_cohort_imputed.parquet'
    
    if not parquet_path.exists():
        # Fall back to original path
        parquet_path = Path(__file__).parent.parent / 'data' / 'processed' / 'OAI_mega_cohort.parquet'
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Data file not found: {parquet_path}\nRun build_cohort_v2.py first.")
    
    df = pd.read_parquet(parquet_path)
    print(f"📂 Loaded data: {df.shape}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"📉 Sampled to: {df.shape}")
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['KL_Grade'], drop_first=True, dtype=float)
    
    # Ensure expected columns exist
    expected_kl_cols = ['KL_Grade_1.0', 'KL_Grade_2.0', 'KL_Grade_3.0', 'KL_Grade_4.0']
    for col in expected_kl_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Check for Sex column and encode if needed
    if 'P02SEX' in df.columns:
        df['Sex_2'] = (df['P02SEX'] == 2).astype(float)
    elif 'Sex' in df.columns:
        df['Sex_2'] = df['Sex'].apply(lambda x: 1 if x == 2 or x == 'Female' else 0).astype(float)
    else:
        df['Sex_2'] = 0
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Local Training for Tri-Modal Survival Model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--sample-size', type=int, default=500, help='Number of samples (0 for all)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save', action='store_true', help='Save best model')
    args = parser.parse_args()
    
    # Get config
    config = get_config('local')
    
    # Device
    device = get_device()
    print(f"🚀 Local Training on {device}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch_size}, Samples: {args.sample_size or 'all'}")
    
    # Prepare data
    sample_size = args.sample_size if args.sample_size > 0 else None
    df = prepare_data(config, sample_size)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['event'])
    print(f"📊 Train: {len(train_df)} ({train_df['event'].sum()} events)")
    print(f"   Val: {len(val_df)} ({val_df['event'].sum()} events)")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    image_root = Path(__file__).parent.parent / 'data' / 'sandbox'
    train_dataset = TriModalDataset(train_df, str(image_root), transform=transform, mode='sandbox')
    val_dataset = TriModalDataset(val_df, str(image_root), transform=transform, mode='sandbox')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    # Count clinical features in dataset
    clinical_cols = [c for c in df.columns if c not in ['ID', 'Knee_Side', 'event', 'time_to_event'] 
                    and not c.startswith('Bio_')]
    print(f"   Clinical features: {len(clinical_cols)}")
    
    model = WideAndDeepSurvivalModel(
        wide_input_dim=15,  # Default from dataset.py
        bio_input_dim=5     # Default from dataset.py
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print(f"\n🏋️ Training...")
    best_val_c_index = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            img, clin, bio, event, time = batch
            img = img.to(device)
            clin = clin.to(device)
            bio = bio.to(device)
            event = event.to(device)
            time = time.to(device)
            
            optimizer.zero_grad()
            risk = model(img, clin, bio).squeeze()
            
            loss = cox_loss(risk, event, time, device)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_risks = []
        val_events = []
        val_times = []
        
        with torch.no_grad():
            for batch in val_loader:
                img, clin, bio, event, time = batch
                img = img.to(device)
                clin = clin.to(device)
                bio = bio.to(device)
                event = event.to(device)
                time = time.to(device)
                
                risk = model(img, clin, bio).squeeze()
                loss = cox_loss(risk, event, time, device)
                
                val_losses.append(loss.item())
                val_risks.extend(risk.cpu().numpy().flatten())
                val_events.extend(event.cpu().numpy().flatten())
                val_times.extend(time.cpu().numpy().flatten())
        
        avg_val_loss = np.mean(val_losses)
        
        # C-index
        val_c_index = compute_c_index(
            np.array(val_risks),
            np.array(val_events),
            np.array(val_times)
        )
        
        # LR
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Print
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"C-index: {val_c_index:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_epoch = epoch + 1
            
            if args.save:
                model_dir = Path(__file__).parent.parent / 'models' / 'local'
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / 'tri_modal_local_best.pth')
                print(f"   ✅ Best model saved (C-index: {best_val_c_index:.4f})")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🏆 Training Complete!")
    print(f"   Best C-index: {best_val_c_index:.4f} (Epoch {best_epoch})")
    print(f"   Target: 0.75 | Gap: {0.75 - best_val_c_index:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
