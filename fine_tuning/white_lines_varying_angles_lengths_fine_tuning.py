import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTConfig
import loralib as lora
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import time
import json

# Configuration
print("Initializing configuration...")
output_dir = r"Path_to_Model\white_lines_varying_angles_lengths_fine_tuned_model"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory set to: {output_dir}")
excel_path = os.path.join(r"Path_to_Labels\white_lines_varying_angles_lengths", "line_labels.xlsx")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
max_samples = 50000

# Dataset
print("Defining LineDataset class...")
class LineDataset(Dataset):
    def __init__(self, image_dir, excel_path, max_samples=50000, transform=None):
        print(f"Loading dataset from {excel_path} with max {max_samples} samples...")
        self.df = pd.read_excel(excel_path).iloc[:max_samples]
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure consistent size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Parse coordinates and normalize them to [0,1] range
        self.df['start_point'] = self.df['start_point'].apply(lambda x: np.array(eval(x))/224.0)
        self.df['end_point'] = self.df['end_point'].apply(lambda x: np.array(eval(x))/224.0)
        self.df['angle'] = (self.df['angle_degrees'] % 360) / 180.0 - 1  # Normalize angles to [-1,1] range
        self.df['noise_level'] = self.df['noise_level'] / self.df['noise_level'].max()  # Normalize noise to [0,1]
        self.df['line_length'] = self.df['line_length'] / 224.0  # Normalize length to [0,1]
        print("Dataset loaded successfully.")
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.df.iloc[idx]['image_id']}.png")
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        target = {
            'angle': torch.tensor(self.df.iloc[idx]['angle'], dtype=torch.float32),
            'start_point': torch.tensor(self.df.iloc[idx]['start_point'], dtype=torch.float32),
            'end_point': torch.tensor(self.df.iloc[idx]['end_point'], dtype=torch.float32),
            'noise_level': torch.tensor(self.df.iloc[idx]['noise_level'], dtype=torch.float32),
            'line_length': torch.tensor(self.df.iloc[idx]['line_length'], dtype=torch.float32),
            'line_width': torch.tensor(self.df.iloc[idx]['line_width'], dtype=torch.float32)
        }
        return image, target

# Model
print("Defining ViTRegression model...")
class ViTRegression(torch.nn.Module):
    def __init__(self, pretrained_model='google/vit-base-patch16-224'):
        super().__init__()
        print(f"Initializing ViT model with {pretrained_model}...")
        config = ViTConfig.from_pretrained(pretrained_model)
        self.vit = ViTForImageClassification.from_pretrained(pretrained_model, config=config)
        
        print("Freezing base model parameters...")
        for param in self.vit.parameters():
            param.requires_grad = False
            
        print("Applying LoRA to attention layers...")
        for layer in self.vit.vit.encoder.layer:
            lora.mark_only_lora_as_trainable(layer.attention.attention)
            
        print("Initializing regression heads...")
        hidden_size = config.hidden_size
        
        # More powerful shared feature extractor
        self.shared_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )
        
        # Initialize heads with appropriate scaling
        self.angle_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size//2, 1)
        )
        
        self.coord_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 4)  # Predicts both start and end points
        )
        
        self.noise_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size//4),
            torch.nn.Linear(hidden_size//4, 1)
        )
        
        self.length_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size//4),
            torch.nn.Linear(hidden_size//4, 1)
        )
        
        self.width_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size//4),
            torch.nn.Linear(hidden_size//4, 1)
        )
        
        # Initialize with smaller weights
        for layer in [self.shared_head, self.angle_head, self.coord_head, self.noise_head, self.length_head, self.width_head]:
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    torch.nn.init.zeros_(m.bias)
                    
        print("Model initialization complete.")
        
    def forward(self, x):
        outputs = self.vit(x, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, 0]
        shared_features = self.shared_head(last_hidden)
        
        # Predict all outputs
        angle = torch.tanh(self.angle_head(shared_features)).squeeze()
        coords = torch.sigmoid(self.coord_head(shared_features))
        noise = torch.sigmoid(self.noise_head(shared_features)).squeeze()
        length = torch.sigmoid(self.length_head(shared_features)).squeeze()
        width = torch.sigmoid(self.width_head(shared_features)).squeeze()
        
        return {
            'angle': angle,
            'start_point': coords[:, :2],
            'end_point': coords[:, 2:],
            'noise_level': noise,
            'line_length': length,
            'line_width': width
        }

# Training
print("Defining training utilities...")
def weighted_mse_loss(preds, targets):
    # Use Huber loss for more robustness
    angle_loss = torch.nn.functional.huber_loss(preds['angle'], targets['angle'], reduction='mean', delta=0.1) * 2.0
    coord_loss = (torch.nn.functional.huber_loss(preds['start_point'], targets['start_point'], reduction='mean', delta=0.1) +
                 torch.nn.functional.huber_loss(preds['end_point'], targets['end_point'], reduction='mean', delta=0.1)) * 1.0
    other_loss = (torch.nn.functional.huber_loss(preds['noise_level'], targets['noise_level'], reduction='mean', delta=0.1) +
                 torch.nn.functional.huber_loss(preds['line_length'], targets['line_length'], reduction='mean', delta=0.1) +
                 torch.nn.functional.huber_loss(preds['line_width'], targets['line_width'], reduction='mean', delta=0.1)) * 0.5
    
    return angle_loss + coord_loss + other_loss

def save_model_and_artifacts(model, output_dir, train_losses, val_losses, inference_times, 
                            train_correlations, val_correlations, train_errors, val_errors):
    print("\nSaving model and artifacts...")
    model_path = os.path.join(output_dir, 'fine_tuned_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    lora_path = os.path.join(output_dir, 'lora_params.pth')
    torch.save({k: v for k, v in model.state_dict().items() if 'lora' in k}, lora_path)
    print(f"LoRA parameters saved to {lora_path}")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        return obj
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'inference_times': inference_times,
        'train_correlations': convert_for_json(train_correlations),
        'val_correlations': convert_for_json(val_correlations),
        'train_errors': convert_for_json(train_errors),
        'val_errors': convert_for_json(val_errors)
    }
    
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    print("Generating and saving figures...")
    save_figures(output_dir, train_losses, val_losses, inference_times, 
                train_correlations, val_correlations, train_errors, val_errors)
    print("Figures saved successfully.")

def save_figures(output_dir, train_losses, val_losses, inference_times, 
                train_correlations, val_correlations, train_errors, val_errors):
    # Loss curves
    print("  - Generating loss curves...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label='Test Loss')
    plt.title('Test Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(inference_times, label='Inference Time')
    plt.title('Inference Time over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Correlation plots
    print("  - Generating correlation plots...")
    features = ['angle', 'start_x', 'start_y', 'end_x', 'end_y', 'noise_level', 'line_length', 'line_width']
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(features):
        plt.subplot(3, 3, i+1)
        plt.plot([corr.get(feat, 0) for corr in train_correlations], label='Train')
        plt.plot([corr.get(feat, 0) for corr in val_correlations], label='Validation')
        plt.title(f'{feat} Correlation')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'))
    plt.close()
    
    # Error distribution plots
    print("  - Generating error distribution plots...")
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(features):
        plt.subplot(3, 3, i+1)
        if feat in train_errors:
            plt.hist(train_errors[feat], bins=50, alpha=0.5, label='Train')
        if feat in val_errors:
            plt.hist(val_errors[feat], bins=50, alpha=0.5, label='Validation')
        plt.title(f'{feat} Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'))
    plt.close()

def calculate_correlations(preds, targets):
    correlations = {}
    preds_np = {k: v.cpu().numpy() for k, v in preds.items()}
    targets_np = {k: v.cpu().numpy() for k, v in targets.items()}
    
    try:
        correlations['angle'] = np.corrcoef(preds_np['angle'], targets_np['angle'])[0, 1]
    except:
        correlations['angle'] = 0
    
    # Start and end points
    for point_type in ['start_point', 'end_point']:
        try:
            for i, coord in enumerate(['x', 'y']):
                key = f"{point_type.split('_')[0]}_{coord}"
                correlations[key] = np.corrcoef(preds_np[point_type][:, i], targets_np[point_type][:, i])[0, 1]
        except:
            for coord in ['x', 'y']:
                key = f"{point_type.split('_')[0]}_{coord}"
                correlations[key] = 0
    
    try:
        correlations['noise_level'] = np.corrcoef(preds_np['noise_level'], targets_np['noise_level'])[0, 1]
    except:
        correlations['noise_level'] = 0
    
    try:
        correlations['line_length'] = np.corrcoef(preds_np['line_length'], targets_np['line_length'])[0, 1]
    except:
        correlations['line_length'] = 0
    
    try:
        correlations['line_width'] = np.corrcoef(preds_np['line_width'], targets_np['line_width'])[0, 1]
    except:
        correlations['line_width'] = 0
    
    return correlations

def calculate_errors(preds, targets):
    errors = {}
    preds_np = {k: v.cpu().numpy() for k, v in preds.items()}
    targets_np = {k: v.cpu().numpy() for k, v in targets.items()}
    
    errors['angle'] = preds_np['angle'] - targets_np['angle']
    
    # Start and end points
    for point_type in ['start_point', 'end_point']:
        for i, coord in enumerate(['x', 'y']):
            key = f"{point_type.split('_')[0]}_{coord}"
            errors[key] = preds_np[point_type][:, i] - targets_np[point_type][:, i]
    
    errors['noise_level'] = preds_np['noise_level'] - targets_np['noise_level']
    errors['line_length'] = preds_np['line_length'] - targets_np['line_length']
    errors['line_width'] = preds_np['line_width'] - targets_np['line_width']
    
    return errors

def main():
    print("\nStarting main training process...")
    print("Loading and splitting dataset...")
    full_dataset = LineDataset(r"Path_to_Images\white_lines_varying_angles_lengths", excel_path)
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    print(f"Dataset split into {len(train_data)} training and {len(val_data)} validation samples")
    
    print("\nInitializing model, optimizer, and scheduler...")
    model = ViTRegression().to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    print("Training setup complete.")
    
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    train_losses = []
    val_losses = []
    inference_times = []
    train_correlations = []
    val_correlations = []
    train_errors = {}
    val_errors = {}
    
    print("\nStarting training loop...")
    for epoch in range(100):
        print(f"\nEpoch {epoch+1}/100")
        start_time = time.time()
        model.train()
        train_loss = 0
        epoch_train_preds = {k: [] for k in ['angle', 'start_point', 'end_point', 'noise_level', 'line_length', 'line_width']}
        epoch_train_targets = {k: [] for k in ['angle', 'start_point', 'end_point', 'noise_level', 'line_length', 'line_width']}
        
        print("  Training phase...")
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = {k: v.to(device) for k,v in targets.items()}
            
            optimizer.zero_grad()
            with autocast():
                preds = model(images)
                loss = weighted_mse_loss(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
            for k in preds.keys():
                epoch_train_preds[k].append(preds[k].detach())
                epoch_train_targets[k].append(targets[k].detach())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Concatenate all batch predictions and targets
        epoch_train_preds = {k: torch.cat(v) for k, v in epoch_train_preds.items()}
        epoch_train_targets = {k: torch.cat(v) for k, v in epoch_train_targets.items()}
        
        # Calculate correlations and errors
        train_corr = calculate_correlations(epoch_train_preds, epoch_train_targets)
        train_correlations.append(train_corr)
        train_errors = calculate_errors(epoch_train_preds, epoch_train_targets)
        
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        epoch_val_preds = {k: [] for k in ['angle', 'start_point', 'end_point', 'noise_level', 'line_length', 'line_width']}
        epoch_val_targets = {k: [] for k in ['angle', 'start_point', 'end_point', 'noise_level', 'line_length', 'line_width']}
        
        print("  Validation phase...")
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = {k: v.to(device) for k,v in targets.items()}
                
                preds = model(images)
                val_loss += weighted_mse_loss(preds, targets).item()
                
                for k in preds.keys():
                    epoch_val_preds[k].append(preds[k].detach())
                    epoch_val_targets[k].append(targets[k].detach())
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"    Validation batch {batch_idx + 1}/{len(val_loader)}")
        
        # Concatenate all batch predictions and targets
        epoch_val_preds = {k: torch.cat(v) for k, v in epoch_val_preds.items()}
        epoch_val_targets = {k: torch.cat(v) for k, v in epoch_val_targets.items()}
        
        # Calculate correlations and errors
        val_corr = calculate_correlations(epoch_val_preds, epoch_val_targets)
        val_correlations.append(val_corr)
        val_errors = calculate_errors(epoch_val_preds, epoch_val_targets)
        
        avg_val_loss = val_loss/len(val_loader)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        inference_times.append(epoch_time)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("  Feature correlations (Train/Val):")
        features_to_print = ['angle', 'start_x', 'start_y', 'end_x', 'end_y', 'noise_level', 'line_length', 'line_width']
        for feat in features_to_print:
            print(f"    {feat}: {train_corr.get(feat, 0):.3f}/{val_corr.get(feat, 0):.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print("  New best model saved!")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save all artifacts
    save_model_and_artifacts(model, output_dir, train_losses, val_losses, inference_times, 
                           train_correlations, val_correlations, train_errors, val_errors)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    print("Script started.")
    main()
    print("Script finished execution.")
