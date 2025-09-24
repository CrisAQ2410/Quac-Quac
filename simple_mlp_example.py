"""
Simple example of how to use the MLP model for land use classification
This script demonstrates the basic usage without all the advanced analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

def simple_mlp_example():
    """Simple example of MLP usage"""
    
    print("Simple MLP Land Use Classification Example")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv("DATA.csv")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Prepare features and target
    feature_columns = ['b1_da_nang', 'b2_da_nang', 'b3_da_nang', 'b4_da_nang', 
                      'b5_da_nang', 'b6_da_nang', 'b7_da_nang']
    X = df[feature_columns]
    y = df['Label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features (important for neural networks!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create and train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers
        activation='relu',             # ReLU activation function
        solver='adam',                 # Adam optimizer
        alpha=0.001,                   # L2 regularization
        max_iter=500,                  # Maximum iterations
        random_state=42,               # For reproducibility
        early_stopping=True            # Stop if validation score doesn't improve
    )
    
    print("Training MLP model...")
    mlp.fit(X_train, y_train)
    
    # Make predictions
    y_pred = mlp.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Example of predicting new samples
    print("\nExample predictions on first 5 test samples:")
    for i in range(5):
        actual_class = label_encoder.classes_[y_test[i]]
        predicted_class = label_encoder.classes_[y_pred[i]]
        confidence = np.max(mlp.predict_proba([X_test[i]]))
        
        print(f"Sample {i+1}: Actual={actual_class}, Predicted={predicted_class}, "
              f"Confidence={confidence:.3f}")
    
    return mlp, scaler, label_encoder

if __name__ == "__main__":
    model, scaler, encoder = simple_mlp_example()