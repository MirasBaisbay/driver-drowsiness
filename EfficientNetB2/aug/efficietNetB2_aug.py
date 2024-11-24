import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class DrowsinessDataset(Dataset):
    """Custom Dataset for loading drowsiness detection images"""
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # Verify the required columns exist
        if 'image_path' not in self.data.columns or 'awake' not in self.data.columns:
            raise ValueError("CSV must contain 'image_path' and 'awake' columns")
        
        # Convert labels to integers if they aren't already
        self.data['awake'] = self.data['awake'].astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['awake']
        
        try:
            # Load and convert image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a black image of the expected size if loading fails
            image = Image.new('RGB', (128, 128), 'black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
       
      
class EarlyStopping:
    """Stop training when validation loss doesn't improve."""
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    """Train the model with early stopping and scheduler."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Step the scheduler
        scheduler.step(epoch_val_loss)

        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'model.pth')

        # Early stopping check
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    return train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('history.png')
    plt.close()

def visualize_predictions(model, val_loader, device, num_samples=8):
    """Visualize model predictions on sample images"""
    model.eval()
    
    # Get sample images and their predictions
    images, labels = next(iter(val_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()

    # Convert images for visualization
    images = images.cpu()
    
    # Create a figure
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Model Predictions on Sample Images', fontsize=16)
    
    # Plot images with predictions
    for idx, (img, pred, true_label) in enumerate(zip(images, predicted, labels)):
        row = idx // 4
        col = idx % 4
        
        # Denormalize the image
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[row, col].imshow(img)
        
        # Set color based on prediction correctness
        color = 'green' if pred == true_label else 'red'
        
        # Add prediction and true label
        title = f'Pred: {"Awake" if pred == 1 else "Drowsy"}\nTrue: {"Awake" if true_label == 1 else "Drowsy"}'
        axes[row, col].set_title(title, color=color)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def visualize_misclassifications(model, val_loader, device):
    """Visualize misclassified cases from the validation set."""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for img, true_label, pred_label in zip(images, labels, predicted):
                if true_label != pred_label:
                    misclassified_images.append(img.cpu())
                    misclassified_labels.append(true_label.cpu().item())
                    misclassified_preds.append(pred_label.cpu().item())

    # Visualize misclassified cases
    num_samples = min(8, len(misclassified_images))
    if num_samples > 0:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle('Misclassified Cases', fontsize=16)
        for idx in range(num_samples):
            img = misclassified_images[idx]
            true_label = misclassified_labels[idx]
            pred_label = misclassified_preds[idx]
            row = idx // 4
            col = idx % 4

            # Denormalize the image
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            axes[row, col].imshow(img)
            color = 'red'
            title = f'Pred: {"Awake" if pred_label == 1 else "Drowsy"}\nTrue: {"Awake" if true_label == 1 else "Drowsy"}'
            axes[row, col].set_title(title, color=color)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig('mis.png')
        plt.close()
    else:
        print("No misclassifications to visualize.")
        
        
def print_final_metrics(model, val_loader, device):
    """Compute and print final accuracy, F1-score, and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Drowsy', 'Awake'])
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\nFinal Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Drowsy', 'Awake'], yticklabels=['Drowsy', 'Awake'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('conf.png')
    plt.close()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Base directory for data
    base_dir = Path(r'C:\Users\Meiras\Desktop\DL\Driver-drowsiness-detection\driver')
    data_dir = base_dir / 'driver_data_cropped'


    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create datasets
    train_dataset = DrowsinessDataset(
        data_dir / 'train' / 'train_processed.csv',
        transform=train_transform
    )
    val_dataset = DrowsinessDataset(
        data_dir / 'valid' / 'valid_processed.csv',
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=weights)

    # Modify the classifier layer for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=2)
    )
    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Train model with early stopping and scheduler
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        device=device,
        patience=5  # Early stopping patience
    )

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Load best model and visualize predictions
    model.load_state_dict(torch.load('model.pth'))
    print("Visualizing model predictions...")
    visualize_predictions(model, val_loader, device)

    # Visualize misclassifications
    print("Visualizing misclassifications...")
    visualize_misclassifications(model, val_loader, device)

    # # Print final metrics and confusion matrix
    print("Computing final metrics...")
    print_final_metrics(model, val_loader, device)
    
if __name__ == '__main__':
    main()
