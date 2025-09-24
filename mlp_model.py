import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

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

def prepare_data(df, scaler_type='standard'):
    """Prepare data for MLP modeling"""
    print("\nPreparing data for MLP modeling...")
    
    # Separate features and target
    # Exclude X, Y coordinates and Label from features
    feature_columns = [col for col in df.columns if col not in ['X', 'Y', 'Label']]
    X = df[feature_columns]
    y = df['Label']
    
    print(f"Feature columns: {feature_columns}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Number of unique classes: {y.nunique()}")
    print(f"Classes: {sorted(y.unique())}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features (very important for neural networks)
    if scaler_type == 'standard':
        scaler = StandardScaler()
        print("Using StandardScaler for feature normalization")
    else:
        scaler = MinMaxScaler()
        print("Using MinMaxScaler for feature normalization")
    
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame for easier handling
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    print(f"\nFeature scaling summary:")
    print(f"Original feature range: {X.min().min():.6f} to {X.max().max():.6f}")
    print(f"Scaled feature range: {X_scaled.min().min():.6f} to {X_scaled.max().max():.6f}")
    
    return X_scaled, y_encoded, label_encoder, scaler

def create_mlp_model(hidden_layers=(100, 50), activation='relu', solver='adam', 
                     alpha=0.0001, learning_rate='constant', max_iter=500, random_state=42):
    """Create and configure MLP classifier"""
    print(f"\nCreating MLP model with:")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Activation function: {activation}")
    print(f"Solver: {solver}")
    print(f"Regularization (alpha): {alpha}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterations: {max_iter}")
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    return model

def train_and_evaluate_mlp(X_train, X_test, y_train, y_test, model, label_encoder):
    """Train MLP model and evaluate performance"""
    print("\nTraining MLP model...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    print(f"Training completed!")
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Training loss: {model.loss_:.6f}")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Check: {train_accuracy - test_accuracy:.4f}")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Precision, Recall, F1-score per class
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_test_pred, average=None)
    
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:15} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, "
              f"F1: {f1[i]:.4f}, Support: {support[i]}")
    
    return y_train_pred, y_test_pred, train_accuracy, test_accuracy

def hyperparameter_tuning(X_train, y_train, cv_folds=5):
    """Perform hyperparameter tuning for MLP"""
    print(f"\nPerforming hyperparameter tuning with {cv_folds}-fold cross-validation...")
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Create base model
    mlp = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
    
    # Grid search
    grid_search = GridSearchCV(
        mlp, 
        param_grid, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def plot_training_history(model, X_train, y_train, X_val, y_val):
    """Plot training history if available"""
    try:
        if hasattr(model, 'loss_curve_'):
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(model.loss_curve_, 'b-', linewidth=2)
            plt.title('Training Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            # Plot validation curve if validation scores available
            if hasattr(model, 'validation_scores_'):
                plt.subplot(1, 2, 2)
                plt.plot(model.validation_scores_, 'g-', linewidth=2, label='Validation Accuracy')
                plt.title('Validation Accuracy Curve')
                plt.xlabel('Iterations')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('mlp_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"Could not plot training history: {e}")

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_feature_importance_approximation(model, feature_names, method='permutation'):
    """
    Approximate feature importance for MLP using different methods
    Note: MLP doesn't have built-in feature importance like tree-based models
    """
    print(f"\nCalculating feature importance approximation using {method} method...")
    
    if method == 'weights':
        # Use first layer weights as approximation
        if hasattr(model, 'coefs_'):
            first_layer_weights = np.abs(model.coefs_[0])
            feature_importance = np.mean(first_layer_weights, axis=1)
            
            # Normalize
            feature_importance = feature_importance / np.sum(feature_importance)
            
            # Plot
            plt.figure(figsize=(10, 6))
            indices = np.argsort(feature_importance)[::-1]
            plt.bar(range(len(feature_importance)), feature_importance[indices])
            plt.title('Feature Importance Approximation (First Layer Weights)')
            plt.xlabel('Features')
            plt.ylabel('Normalized Weight Magnitude')
            plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('mlp_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print importance values
            print("\nFeature importance (normalized):")
            for i in indices:
                print(f"{feature_names[i]:15}: {feature_importance[i]:.4f}")

def statistical_analysis(df):
    """Perform statistical analysis on the dataset"""
    print("\nPerforming statistical analysis...")
    
    # Feature correlation matrix
    feature_columns = [col for col in df.columns if col not in ['X', 'Y', 'Label']]
    correlation_matrix = df[feature_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('mlp_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Chi-square test for feature-label independence
    print(f"\nChi-square tests for feature-label independence:")
    for feature in feature_columns:
        # Discretize continuous features for chi-square test
        feature_binned = pd.cut(df[feature], bins=5, labels=False)
        contingency_table = pd.crosstab(feature_binned, df['Label'])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"{feature:15}: Chi2={chi2:.4f}, p-value={p_value:.4e}")

def main():
    """Main function to run the complete MLP analysis"""
    print("=" * 60)
    print("Multi-Layer Perceptron (MLP) Classification Analysis")
    print("=" * 60)
    
    # Load and explore data
    file_path = "DATA.csv"
    df = load_and_explore_data(file_path)
    
    # Statistical analysis
    statistical_analysis(df)
    
    # Prepare data
    X, y, label_encoder, scaler = prepare_data(df, scaler_type='standard')
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Create and train basic MLP model
    mlp_model = create_mlp_model(
        hidden_layers=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000
    )
    
    # Train and evaluate
    y_train_pred, y_test_pred, train_acc, test_acc = train_and_evaluate_mlp(
        X_train, X_test, y_train, y_test, mlp_model, label_encoder
    )
    
    # Plot results
    plot_training_history(mlp_model, X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_test_pred, label_encoder.classes_, 'MLP Confusion Matrix')
    plot_feature_importance_approximation(mlp_model, X.columns.tolist())
    
    # Hyperparameter tuning (optional - can be time consuming)
    print(f"\nDo you want to perform hyperparameter tuning? (This may take several minutes)")
    tune_params = input("Enter 'yes' to tune hyperparameters or 'no' to skip: ").lower().strip()
    
    if tune_params == 'yes':
        best_model, best_params = hyperparameter_tuning(X_train, y_train)
        
        # Evaluate best model
        print(f"\nEvaluating best model from hyperparameter tuning...")
        y_train_pred_best, y_test_pred_best, train_acc_best, test_acc_best = train_and_evaluate_mlp(
            X_train, X_test, y_train, y_test, best_model, label_encoder
        )
        
        plot_confusion_matrix(y_test, y_test_pred_best, label_encoder.classes_, 
                            'Best MLP Confusion Matrix (After Tuning)')
    
    # Cross-validation
    print(f"\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(mlp_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print(f"\n" + "="*60)
    print("MLP Analysis Complete!")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()