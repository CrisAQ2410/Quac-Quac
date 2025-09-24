# PyTorch MLP Models for Land Use Classification

## Overview
This directory now includes **PyTorch implementations** of Multi-Layer Perceptron (MLP) neural networks for land use classification, providing more flexibility and control compared to scikit-learn versions.

## 🔥 PyTorch Files Added

### 1. `pytorch_mlp_model.py` - Comprehensive PyTorch Implementation
**Advanced Features:**
- Custom neural network class with flexible architecture
- Proper train/validation/test splits
- Advanced training loop with early stopping
- Learning rate scheduling
- Cross-validation support
- GPU acceleration (CUDA) support
- Comprehensive visualization and evaluation

### 2. `simple_pytorch_example.py` - Basic PyTorch Usage
**Simple Features:**
- Minimal PyTorch MLP implementation
- Easy to understand and modify
- Basic training loop
- Quick results

### 3. `enhanced_model_comparison.py` - Complete Framework Comparison
**Comparison Features:**
- Random Forest vs XGBoost vs Scikit-learn MLP vs **PyTorch MLP**
- Framework-specific analysis
- Performance benchmarking

## 📊 Performance Comparison Results

| Rank | Model | Framework | Test Accuracy | CV Score | Training Time |
|------|--------|-----------|---------------|----------|---------------|
| 🥇 #1 | **XGBoost** | XGBoost | **68.33%** | 66.11%±4.2% | 2.13s |
| 🥈 #2 | **PyTorch MLP** | PyTorch | **67.22%** | 69.86%±5.1% | 5.98s |
| 🥉 #3 | Random Forest | Scikit-learn | 66.11% | 67.50%±1.1% | 0.84s |
| #4 | Scikit-learn MLP | Scikit-learn | 54.44% | 61.67%±3.7% | 1.08s |

### Key Insights:
- **PyTorch MLP ranks #2** - very close to XGBoost performance!
- **Best Cross-Validation**: PyTorch MLP (69.86%)
- **PyTorch shows excellent generalization** with high CV scores
- **More flexible** than scikit-learn MLP implementation

## 🚀 PyTorch Advantages

### ✅ **Flexibility**
- Custom network architectures
- Multiple activation functions (ReLU, Tanh, LeakyReLU, Sigmoid)
- Configurable dropout rates
- Custom loss functions and optimizers

### ✅ **Performance Features**
- GPU acceleration with CUDA
- Batch processing with DataLoaders
- Advanced optimizers (Adam, SGD, RMSprop)
- Learning rate scheduling
- Early stopping

### ✅ **Research Capabilities**
- Easy to experiment with new architectures
- Custom training loops
- Gradient analysis
- Model interpretability tools

## 🛠️ PyTorch Model Architecture

```python
class MLPClassifier(nn.Module):
    def __init__(self, input_size=7, hidden_sizes=[128, 64, 32], 
                 num_classes=9, dropout_rate=0.3):
        # Input Layer: 7 spectral bands
        # Hidden Layer 1: 128 neurons + ReLU + Dropout(0.3)
        # Hidden Layer 2: 64 neurons + ReLU + Dropout(0.3)  
        # Hidden Layer 3: 32 neurons + ReLU + Dropout(0.3)
        # Output Layer: 9 classes (land use types)
```

**Model Statistics:**
- **Total Parameters**: 11,657 trainable parameters
- **Architecture**: 7 → 128 → 64 → 32 → 9
- **Activation**: ReLU with 30% dropout
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Loss**: CrossEntropyLoss

## 📦 Installation & Setup

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other required packages
pip install scikit-learn pandas numpy matplotlib seaborn xgboost
```

## 🔧 Usage Examples

### Quick Start - Simple PyTorch
```python
# Run basic PyTorch example
python simple_pytorch_example.py
```

### Advanced Usage - Full PyTorch Implementation
```python
# Run comprehensive PyTorch analysis
python pytorch_mlp_model.py
```

### Compare All Models
```python
# Compare all frameworks
python enhanced_model_comparison.py
```

### Custom PyTorch Model
```python
import torch
import torch.nn as nn
from pytorch_mlp_model import MLPClassifier

# Create custom architecture
model = MLPClassifier(
    input_size=7,
    hidden_sizes=[256, 128, 64, 32],  # 4 hidden layers
    num_classes=9,
    dropout_rate=0.4,                 # Higher dropout
    activation='leaky_relu'           # Different activation
)

# Train with custom parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
```

## 🎯 GPU Acceleration

```python
# Automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model and data to GPU
model = model.to(device)
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
```

## 📈 Hyperparameter Tuning Options

### Architecture Tuning
- **Hidden Layers**: `[64], [128, 64], [256, 128, 64], [512, 256, 128, 64]`
- **Activation Functions**: `'relu'`, `'tanh'`, `'leaky_relu'`, `'sigmoid'`
- **Dropout Rates**: `0.0, 0.1, 0.2, 0.3, 0.5`

### Training Tuning
- **Learning Rates**: `0.001, 0.0005, 0.01, 0.005`
- **Batch Sizes**: `16, 32, 64, 128`
- **Optimizers**: `Adam`, `SGD`, `RMSprop`
- **Weight Decay**: `1e-5, 1e-4, 1e-3`

## 📊 Visualization Outputs

PyTorch models generate:
1. **Training History**: Loss and accuracy curves
2. **Confusion Matrix**: Per-class performance
3. **Model Architecture**: Network structure diagram
4. **Feature Analysis**: Input feature relationships

## 🔍 When to Use Each Framework

### Use **PyTorch** when:
- 🔬 Research and experimentation
- 🎛️ Need custom architectures
- 🚀 GPU acceleration required
- 🔧 Fine-grained control needed
- 📚 Learning deep learning concepts

### Use **Scikit-learn** when:
- ⚡ Quick prototyping
- 🎯 Simple, standard models
- 📖 Consistent, easy API
- 🔄 Pipeline integration
- 👥 Team collaboration

### Use **XGBoost** when:
- 🏆 Maximum performance on tabular data
- 📊 Structured/tabular datasets
- 🏅 Competition settings
- ⚖️ Built-in regularization needed

## 🚀 Next Steps with PyTorch

1. **Advanced Architectures**:
   - Residual connections
   - Attention mechanisms
   - Batch normalization

2. **Regularization Techniques**:
   - L1/L2 regularization
   - Batch normalization
   - Layer normalization

3. **Advanced Training**:
   - Learning rate scheduling
   - Gradient clipping
   - Mixed precision training

4. **Model Interpretation**:
   - Gradient-based feature importance
   - Layer-wise relevance propagation
   - Activation maximization

## 🎯 Conclusion

The **PyTorch implementation achieves 67.22% accuracy**, making it the **second-best performing model** after XGBoost. It offers:

- ✅ **High Performance**: Competitive with tree-based models
- ✅ **Flexibility**: Custom architectures and training procedures  
- ✅ **Scalability**: GPU acceleration and batch processing
- ✅ **Research Ready**: Foundation for advanced deep learning

PyTorch provides the **perfect balance** between performance and flexibility for neural network research and development! 🔥