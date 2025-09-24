"""
Enhanced Model Comparison: Random Forest vs XGBoost vs Scikit-learn MLP vs PyTorch MLP
This script compares all four models on the same dataset
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import time

# PyTorch MLP class
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(SimpleMLPClassifier, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_pytorch_model(X_train, X_test, y_train, y_test, epochs=100, device='cpu'):
    """Train PyTorch model"""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create model
    model = SimpleMLPClassifier(
        input_size=X_train.shape[1],
        hidden_sizes=[128, 64],
        num_classes=len(np.unique(y_train)),
        dropout_rate=0.2
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = (test_predicted == y_test_tensor).float().mean().item()
    
    return model, test_accuracy, test_predicted.cpu().numpy()

def compare_all_models():
    """Compare all four models"""
    
    print("Enhanced Model Comparison: RF vs XGBoost vs Scikit-learn MLP vs PyTorch MLP")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device: {device}")
    
    # Load and prepare data
    df = pd.read_csv("DATA.csv")
    
    # Features and target
    feature_columns = ['b1_da_nang', 'b2_da_nang', 'b3_da_nang', 'b4_da_nang', 
                      'b5_da_nang', 'b6_da_nang', 'b7_da_nang']
    X = df[feature_columns]
    y = df['Label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'use_scaled': False
        },
        'XGBoost': {
            'model': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                 random_state=42, eval_metric='logloss'),
            'use_scaled': False
        },
        'Scikit-learn MLP': {
            'model': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', 
                                 solver='adam', alpha=0.001, max_iter=500, 
                                 random_state=42, early_stopping=True),
            'use_scaled': True
        }
    }
    
    results = {}
    
    print("\\nTraining and evaluating models...")
    print("-" * 80)
    
    # Train traditional models
    for name, model_info in models.items():
        print(f"\\nTraining {name}...")
        start_time = time.time()
        
        model = model_info['model']
        
        if model_info['use_scaled']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"✓ {name} completed in {training_time:.2f} seconds")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    
    # Train PyTorch model
    print(f"\\nTraining PyTorch MLP...")
    start_time = time.time()
    
    pytorch_model, pytorch_accuracy, pytorch_predictions = train_pytorch_model(
        X_train_scaled, X_test_scaled, y_train, y_test, epochs=150, device=device
    )
    
    end_time = time.time()
    pytorch_time = end_time - start_time
    
    # PyTorch cross-validation (simplified)
    pytorch_cv_scores = []
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train_scaled, y_train):
        X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        _, cv_accuracy, _ = train_pytorch_model(
            X_train_cv, X_val_cv, y_train_cv, y_val_cv, epochs=50, device=device
        )
        pytorch_cv_scores.append(cv_accuracy)
    
    pytorch_cv_scores = np.array(pytorch_cv_scores)
    
    results['PyTorch MLP'] = {
        'accuracy': pytorch_accuracy,
        'cv_mean': pytorch_cv_scores.mean(),
        'cv_std': pytorch_cv_scores.std(),
        'training_time': pytorch_time,
        'predictions': pytorch_predictions
    }
    
    print(f"✓ PyTorch MLP completed in {pytorch_time:.2f} seconds")
    print(f"  Test Accuracy: {pytorch_accuracy:.4f}")
    print(f"  CV Score: {pytorch_cv_scores.mean():.4f} (+/- {pytorch_cv_scores.std()*2:.4f})")
    
    # Print comparison summary
    print("\\n" + "=" * 80)
    print("ENHANCED COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"{'Model':<20} {'Framework':<15} {'Test Acc':<10} {'CV Score':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    # Sort by test accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    frameworks = {
        'Random Forest': 'Scikit-learn',
        'XGBoost': 'XGBoost',
        'Scikit-learn MLP': 'Scikit-learn', 
        'PyTorch MLP': 'PyTorch'
    }
    
    for i, (name, metrics) in enumerate(sorted_results):
        rank = f"#{i+1}"
        framework = frameworks.get(name, 'Unknown')
        print(f"{rank} {name:<17} {framework:<15} {metrics['accuracy']:<10.4f} "
              f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}   {metrics['training_time']:<10.2f}")
    
    # Best model analysis
    best_model = sorted_results[0][0]
    best_accuracy = sorted_results[0][1]['accuracy']
    
    print(f"\\n🏆 BEST MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")
    print(f"   Framework: {frameworks[best_model]}")
    
    # Framework comparison
    print(f"\\nFramework Analysis:")
    print("-" * 40)
    
    sklearn_models = [name for name in results.keys() if 'Scikit-learn' in frameworks.get(name, '')]
    if sklearn_models:
        sklearn_avg = np.mean([results[name]['accuracy'] for name in sklearn_models])
        print(f"Scikit-learn Average: {sklearn_avg:.4f}")
    
    if 'PyTorch MLP' in results:
        pytorch_acc = results['PyTorch MLP']['accuracy']
        print(f"PyTorch MLP: {pytorch_acc:.4f}")
    
    if 'XGBoost' in results:
        xgb_acc = results['XGBoost']['accuracy']
        print(f"XGBoost: {xgb_acc:.4f}")
    
    # Detailed classification report for best model
    print(f"\\nDetailed Classification Report for {best_model}:")
    print("-" * 60)
    best_predictions = sorted_results[0][1]['predictions']
    print(classification_report(y_test, best_predictions, target_names=label_encoder.classes_))
    
    # Framework advantages
    print(f"\\nFramework Characteristics:")
    print("-" * 35)
    print("Scikit-learn:")
    print("  + Easy to use and well-documented")
    print("  + Consistent API across algorithms") 
    print("  + Good for rapid prototyping")
    print("  + Built-in preprocessing and metrics")
    
    print("\\nPyTorch:")
    print("  + Maximum flexibility and control")
    print("  + Custom architectures possible")
    print("  + GPU acceleration available")
    print("  + Research-oriented features")
    print("  - Requires more code")
    
    print("\\nXGBoost:")
    print("  + Often best performance on tabular data")
    print("  + Built-in regularization")
    print("  + Handles missing values")
    print("  + Industry standard for competitions")
    
    return results, label_encoder

if __name__ == "__main__":
    results, encoder = compare_all_models()