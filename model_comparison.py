"""
Model Comparison: Random Forest vs XGBoost vs MLP
This script compares the performance of all three models on the same dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import time

def compare_models():
    """Compare Random Forest, XGBoost, and MLP models"""
    
    print("Model Comparison: Random Forest vs XGBoost vs MLP")
    print("=" * 60)
    
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
    
    # Scale features for MLP (neural networks need scaled features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ),
        'MLP (Neural Network)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    }
    
    results = {}
    
    print("\nTraining and evaluating models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Use scaled data for MLP, original data for tree-based models
        if name == 'MLP (Neural Network)':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            # Cross-validation on scaled data
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Cross-validation on original data
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calculate metrics
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
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"{'Model':<25} {'Test Acc':<10} {'CV Score':<15} {'Time (s)':<10}")
    print("-" * 60)
    
    # Sort by test accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (name, metrics) in enumerate(sorted_results):
        rank = f"#{i+1}"
        print(f"{rank} {name:<22} {metrics['accuracy']:<10.4f} "
              f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}   {metrics['training_time']:<10.2f}")
    
    # Best model analysis
    best_model = sorted_results[0][0]
    best_accuracy = sorted_results[0][1]['accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")
    
    # Detailed classification report for best model
    print(f"\nDetailed Classification Report for {best_model}:")
    print("-" * 50)
    best_predictions = sorted_results[0][1]['predictions']
    print(classification_report(y_test, best_predictions, target_names=label_encoder.classes_))
    
    # Model characteristics analysis
    print("\nModel Characteristics:")
    print("-" * 30)
    print("Random Forest:")
    print("  + Good baseline performance")
    print("  + Handles mixed data types well")
    print("  + Feature importance available")
    print("  - May overfit with small datasets")
    
    print("\nXGBoost:")
    print("  + Often achieves high performance")
    print("  + Built-in regularization")
    print("  + Handles missing values")
    print("  - Requires hyperparameter tuning")
    
    print("\nMLP (Neural Network):")
    print("  + Can capture complex patterns")
    print("  + Flexible architecture")
    print("  + Good for non-linear relationships")
    print("  - Requires feature scaling")
    print("  - May need more data")
    print("  - Less interpretable")
    
    return results, label_encoder

if __name__ == "__main__":
    results, encoder = compare_models()