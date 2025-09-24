"""
PyTorch Multi-Layer Perceptron (MLP) for Land Use Classification
This implementation uses PyTorch for more flexibility and control over the neural network
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import time
import warnings
warnings.filterwarnings('ignore')

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MLPClassifier(nn.Module):
    """
    Custom MLP Neural Network built with PyTorch
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2, activation='relu'):
        super(MLPClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def get_model_info(self):
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }

def load_and_explore_data(file_path):
    """Load the dataset and perform exploratory data analysis"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.info())
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nClass distribution:")
    print(df['Label'].value_counts())
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def prepare_data_pytorch(df, scaler_type='standard', test_size=0.2, val_size=0.1, random_state=42):
    """Prepare data for PyTorch training with train/validation/test split"""
    print("\nPreparing data for PyTorch MLP...")
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['X', 'Y', 'Label']]
    X = df[feature_columns].values
    y = df['Label'].values
    
    print(f"Feature columns: {feature_columns}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Number of unique classes: {len(np.unique(y))}")
    print(f"Classes: {sorted(np.unique(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    if scaler_type == 'standard':
        scaler = StandardScaler()
        print("Using StandardScaler for feature normalization")
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("Using MinMaxScaler for feature normalization")
    
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeature scaling summary:")
    print(f"Original feature range: {X.min():.6f} to {X.max():.6f}")
    print(f"Scaled feature range: {X_scaled.min():.6f} to {X_scaled.max():.6f}")
    
    # Split data: first train/temp, then temp into val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=(test_size + val_size), 
        random_state=random_state, stratify=y_encoded
    )
    
    # Split temp into validation and test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_size),
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler, feature_columns

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Create PyTorch DataLoaders"""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=100, patience=10, device=device):
    """Train the PyTorch model"""
    
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining model on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss_avg)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    return model, history

def evaluate_model(model, test_loader, label_encoder, device=device):
    """Evaluate the trained model"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nTest Results:")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    class_names = label_encoder.classes_
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return y_true, y_pred, accuracy

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_mlp_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title='PyTorch MLP Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('pytorch_mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def cross_validate_pytorch(X, y, model_params, cv_folds=5, random_state=42):
    """Perform cross-validation with PyTorch model"""
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{cv_folds}...")
        
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_cv).to(device)
        X_val_tensor = torch.FloatTensor(X_val_cv).to(device)
        y_train_tensor = torch.LongTensor(y_train_cv).to(device)
        y_val_tensor = torch.LongTensor(y_val_cv).to(device)
        
        # Create model
        model = MLPClassifier(**model_params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Simple training for CV
        model.train()
        for epoch in range(50):  # Fewer epochs for CV
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_val_tensor).float().mean().item()
            cv_scores.append(accuracy)
    
    cv_scores = np.array(cv_scores)
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def main():
    """Main function to run PyTorch MLP analysis"""
    print("=" * 70)
    print("PyTorch Multi-Layer Perceptron (MLP) Classification Analysis")
    print("=" * 70)
    
    # Load data
    file_path = "DATA.csv"
    df = load_and_explore_data(file_path)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler, feature_names = prepare_data_pytorch(df)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    # Model parameters
    model_params = {
        'input_size': X_train.shape[1],
        'hidden_sizes': [128, 64, 32],
        'num_classes': len(label_encoder.classes_),
        'dropout_rate': 0.3,
        'activation': 'relu'
    }
    
    # Create model
    model = MLPClassifier(**model_params)
    print(f"\nModel Architecture:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Train model
    start_time = time.time()
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=200, patience=15, device=device
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    y_true, y_pred, test_accuracy = evaluate_model(trained_model, test_loader, label_encoder)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, label_encoder.classes_)
    
    # Cross-validation
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.hstack([y_train, y_val, y_test])
    cv_scores = cross_validate_pytorch(X_all, y_all, model_params, cv_folds=5)
    
    print(f"\n" + "="*70)
    print("PyTorch MLP Analysis Complete!")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Training Time: {training_time:.2f} seconds")
    print("="*70)
    
    return trained_model, history, label_encoder, scaler

if __name__ == "__main__":
    model, history, encoder, scaler = main()