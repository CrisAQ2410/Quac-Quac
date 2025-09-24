"""
Simple PyTorch MLP Example for Land Use Classification
Minimal code to demonstrate PyTorch neural network usage
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def simple_pytorch_example():
    """Simple PyTorch MLP example"""
    print("Simple PyTorch MLP Land Use Classification")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv("DATA.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and target
    feature_columns = ['b1_da_nang', 'b2_da_nang', 'b3_da_nang', 'b4_da_nang', 
                      'b5_da_nang', 'b6_da_nang', 'b7_da_nang']
    X = df[feature_columns].values
    y = df['Label'].values
    
    # Encode labels and scale features
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create model
    input_size = X_train.shape[1]
    hidden_size = 128
    num_classes = len(label_encoder.classes_)
    
    model = SimpleMLPClassifier(input_size, hidden_size, num_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print(f"\nTraining PyTorch model...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_outputs = model(X_train_tensor)
        _, train_predicted = torch.max(train_outputs, 1)
        train_accuracy = (train_predicted == y_train_tensor).float().mean().item()
        
        # Test accuracy
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = (test_predicted == y_test_tensor).float().mean().item()
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed classification report
    y_pred_np = test_predicted.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_np, y_pred_np, target_names=label_encoder.classes_))
    
    # Example predictions
    print(f"\nExample predictions on first 5 test samples:")
    with torch.no_grad():
        for i in range(5):
            actual_class = label_encoder.classes_[y_test_np[i]]
            predicted_class = label_encoder.classes_[y_pred_np[i]]
            
            # Get prediction probabilities
            test_sample = X_test_tensor[i:i+1]
            outputs = model(test_sample)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item()
            
            print(f"Sample {i+1}: Actual={actual_class}, Predicted={predicted_class}, "
                  f"Confidence={confidence:.3f}")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    model, scaler, encoder = simple_pytorch_example()