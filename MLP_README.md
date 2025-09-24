# Multi-Layer Perceptron (MLP) Model for Land Use Classification

## Overview
This document explains how to build and use a Multi-Layer Perceptron (MLP) neural network for land use classification using spectral band data.

## Dataset
- **Features**: 7 spectral bands (b1_da_nang through b7_da_nang)
- **Target**: 9 land use classes (aquaculture, barren, croplands, forest, grassland, residential land, rice paddies, scrub, water)
- **Size**: 900 samples, balanced dataset (100 samples per class)

## Files Created

### 1. `mlp_model.py` - Comprehensive MLP Implementation
**Features:**
- Complete data loading and exploration
- Feature scaling (StandardScaler/MinMaxScaler)
- Configurable MLP architecture
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Comprehensive evaluation metrics
- Visualization (training curves, confusion matrix, feature importance approximation)
- Statistical analysis

### 2. `simple_mlp_example.py` - Basic Usage Example
**Features:**
- Minimal code for quick MLP implementation
- Basic preprocessing and training
- Example predictions
- Easy to understand and modify

### 3. `model_comparison.py` - Model Performance Comparison
**Features:**
- Compares Random Forest, XGBoost, and MLP
- Side-by-side performance metrics
- Training time comparison
- Model characteristics analysis

## Key MLP Configuration

### Architecture
```python
MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
    activation='relu',                  # ReLU activation
    solver='adam',                     # Adam optimizer
    alpha=0.001,                       # L2 regularization
    learning_rate='adaptive',          # Adaptive learning rate
    max_iter=1000,                    # Maximum iterations
    early_stopping=True,              # Prevent overfitting
    validation_fraction=0.1,          # 10% for validation
    random_state=42                   # Reproducibility
)
```

### Data Preprocessing
1. **Feature Selection**: Exclude X, Y coordinates (keep only spectral bands)
2. **Label Encoding**: Convert categorical labels to numerical
3. **Feature Scaling**: **CRITICAL** - Neural networks require scaled features
   - StandardScaler: zero mean, unit variance
   - MinMaxScaler: scale to [0,1] range

## Performance Results

### Model Comparison (Test Accuracy)
1. **XGBoost**: 68.33%
2. **Random Forest**: 66.11% 
3. **MLP**: 63.33%

### MLP Specific Results
- **Training Accuracy**: 70.56%
- **Test Accuracy**: 64.44%
- **Cross-Validation**: 64.78% (±9.44%)
- **Training Time**: ~1 second
- **Convergence**: 62 iterations

## Why Use MLP for This Dataset?

### Advantages
✅ **Non-linear Pattern Recognition**: Can capture complex spectral relationships  
✅ **Flexible Architecture**: Easily adjustable hidden layers and neurons  
✅ **Universal Approximator**: Theoretically can approximate any function  
✅ **Good for Spectral Data**: Effective with continuous numerical features  

### Considerations
⚠️ **Feature Scaling Required**: Must normalize input features  
⚠️ **More Data Sensitive**: Performance improves with larger datasets  
⚠️ **Less Interpretable**: Black box compared to tree-based models  
⚠️ **Hyperparameter Sensitive**: Requires tuning for optimal performance  

## Usage Instructions

### Basic Usage
```python
# Run the simple example
python simple_mlp_example.py

# Run comprehensive analysis
python mlp_model.py

# Compare all models
python model_comparison.py
```

### Custom Configuration
```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load and scale your data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create custom MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),  # Larger network
    activation='tanh',                  # Different activation
    solver='lbfgs',                    # Different solver
    alpha=0.01,                        # More regularization
    max_iter=2000                      # More iterations
)

# Train
mlp.fit(X_scaled, y)
```

## Hyperparameter Tuning

The `mlp_model.py` includes automated hyperparameter tuning for:
- **hidden_layer_sizes**: Network architecture
- **activation**: relu, tanh
- **solver**: adam, lbfgs  
- **alpha**: L2 regularization strength
- **learning_rate**: constant, adaptive

## Visualization Output

The model generates several plots:
1. **Training Loss Curve**: Shows convergence
2. **Confusion Matrix**: Classification performance per class
3. **Feature Correlation Matrix**: Understanding feature relationships
4. **Feature Importance Approximation**: Based on first layer weights

## Tips for Better Performance

1. **More Data**: MLPs typically perform better with larger datasets
2. **Feature Engineering**: Consider additional spectral indices (NDVI, etc.)
3. **Architecture Tuning**: Experiment with different layer sizes
4. **Regularization**: Adjust alpha to prevent overfitting
5. **Early Stopping**: Use validation to prevent overtraining
6. **Cross-Validation**: Always validate with k-fold CV

## Next Steps

1. **Hyperparameter Optimization**: Run the grid search in `mlp_model.py`
2. **Feature Engineering**: Add spectral indices or texture features
3. **Ensemble Methods**: Combine MLP with Random Forest/XGBoost
4. **Deep Learning**: Try convolutional neural networks if you have spatial data
5. **Class Imbalance**: If real data is imbalanced, use class weights

## Conclusion

The MLP model provides a solid neural network approach for land use classification. While it doesn't outperform XGBoost on this particular dataset, it offers flexibility and can capture complex non-linear patterns that might be missed by tree-based models. The comprehensive implementation provides a foundation for further experimentation and improvement.