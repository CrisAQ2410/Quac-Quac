import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

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

def prepare_data(df):
    """Prepare data for Random Forest modeling"""
    print("\nPreparing data for modeling...")
    
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
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Check class distribution balance
    check_class_balance(y_train, y_test, label_encoder)
    
    return X_train, X_test, y_train, y_test, label_encoder, feature_columns

def check_class_balance(y_train, y_test, label_encoder):
    """Check class distribution balance across train/test splits"""
    print("\n" + "="*60)
    print("📊 CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Get original class names
    class_names = label_encoder.classes_
    
    # Create distribution analysis
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    # Create comprehensive distribution table
    distribution_df = pd.DataFrame({
        'Class': class_names,
        'Train_Count': train_counts.values,
        'Test_Count': test_counts.values,
        'Total_Count': train_counts.values + test_counts.values,
        'Train_Percentage': (train_counts.values / len(y_train) * 100).round(2),
        'Test_Percentage': (test_counts.values / len(y_test) * 100).round(2),
    })
    
    # Add ratio analysis
    distribution_df['Train_Test_Ratio'] = (distribution_df['Train_Count'] / distribution_df['Test_Count']).round(2)
    
    print("\n🔍 Detailed Class Distribution:")
    print(distribution_df.to_string(index=False))
    
    # Balance analysis
    print(f"\n📈 Balance Analysis:")
    print(f"   Total Training Samples: {len(y_train)}")
    print(f"   Total Testing Samples: {len(y_test)}")
    print(f"   Number of Classes: {len(class_names)}")
    
    # Check if perfectly balanced
    expected_train_per_class = len(y_train) / len(class_names)
    expected_test_per_class = len(y_test) / len(class_names)
    
    train_is_balanced = all(count == train_counts.iloc[0] for count in train_counts)
    test_is_balanced = all(count == test_counts.iloc[0] for count in test_counts)
    
    print(f"\n✅ Balance Status:")
    if train_is_balanced and test_is_balanced:
        print(f"   🎯 PERFECTLY BALANCED: Each class has exactly {train_counts.iloc[0]} training and {test_counts.iloc[0]} testing samples")
    else:
        print(f"   ⚠️  IMBALANCED DETECTED:")
        
        # Train set analysis
        train_min, train_max = train_counts.min(), train_counts.max()
        print(f"   📊 Training Set:")
        print(f"      - Expected per class: {expected_train_per_class:.1f}")
        print(f"      - Actual range: {train_min} - {train_max}")
        print(f"      - Imbalance ratio: {train_max/train_min:.2f}:1")
        
        # Test set analysis
        test_min, test_max = test_counts.min(), test_counts.max()
        print(f"   📊 Testing Set:")
        print(f"      - Expected per class: {expected_test_per_class:.1f}")
        print(f"      - Actual range: {test_min} - {test_max}")
        print(f"      - Imbalance ratio: {test_max/test_min:.2f}:1")
    
    # Identify most/least represented classes
    print(f"\n🏆 Class Representation:")
    most_train_idx = train_counts.idxmax()
    least_train_idx = train_counts.idxmin()
    most_test_idx = test_counts.idxmax()
    least_test_idx = test_counts.idxmin()
    
    print(f"   📈 Most samples in training: {class_names[most_train_idx]} ({train_counts.iloc[most_train_idx]} samples)")
    print(f"   📉 Least samples in training: {class_names[least_train_idx]} ({train_counts.iloc[least_train_idx]} samples)")
    print(f"   📈 Most samples in testing: {class_names[most_test_idx]} ({test_counts.iloc[most_test_idx]} samples)")
    print(f"   📉 Least samples in testing: {class_names[least_test_idx]} ({test_counts.iloc[least_test_idx]} samples)")
    
    # Statistical significance test
    from scipy.stats import chi2_contingency
    
    # Create contingency table
    contingency_table = np.array([train_counts.values, test_counts.values])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\n🧮 Statistical Analysis:")
    print(f"   Chi-square test for independence:")
    print(f"   - Chi2 statistic: {chi2:.4f}")
    print(f"   - P-value: {p_value:.6f}")
    print(f"   - Degrees of freedom: {dof}")
    
    if p_value > 0.05:
        print(f"   ✅ Train/Test splits are statistically similar (p > 0.05)")
    else:
        print(f"   ⚠️  Train/Test splits show significant difference (p ≤ 0.05)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if train_is_balanced and test_is_balanced:
        print(f"   ✅ No action needed - data is perfectly balanced")
        print(f"   ✅ Stratified sampling worked correctly")
    else:
        print(f"   ⚠️  Consider using stratified sampling for better balance")
        print(f"   ⚠️  Monitor per-class performance carefully")
        print(f"   ⚠️  Consider class weighting in model training")
        
        # Suggest class weights
        train_class_weights = len(y_train) / (len(class_names) * train_counts.values)
        print(f"   💻 Suggested class weights for Random Forest:")
        for i, (class_name, weight) in enumerate(zip(class_names, train_class_weights)):
            print(f"      {class_name}: {weight:.3f}")
    
    print("="*60)

def build_random_forest(X_train, y_train):
    """Build and train Random Forest model"""
    print("\nBuilding Random Forest model...")
    
    # Create Random Forest classifier with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,           # Number of trees
        max_depth=None,             # Maximum depth of trees
        min_samples_split=2,        # Minimum samples required to split
        min_samples_leaf=1,         # Minimum samples required at leaf node
        max_features='sqrt',        # Number of features to consider at each split
        bootstrap=True,             # Bootstrap samples when building trees
        random_state=42,            # For reproducibility
        n_jobs=-1                   # Use all available cores
    )
    
    # Train the model
    print("Training the model...")
    rf_model.fit(X_train, y_train)
    
    print("Model training completed!")
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Number of features: {rf_model.n_features_in_}")
    
    return rf_model

def evaluate_model(rf_model, X_test, y_test, label_encoder, feature_columns):
    """Evaluate the Random Forest model with comprehensive metrics"""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate accuracy (primary metric)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get class names for better reporting
    class_names = label_encoder.classes_
    
    # Detailed classification report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔍 Confusion Matrix:")
    print("Rows: Actual classes, Columns: Predicted classes")
    
    # Create a formatted confusion matrix with class names
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Overall metrics summary
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print("\n📈 Overall Performance Metrics:")
    print(f"   Weighted Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Weighted Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   Weighted F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"   Macro Precision:    {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"   Macro Recall:       {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"   Macro F1-Score:     {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    print("\n📋 Per-Class Performance:")
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'F1-Score': per_class_f1,
        'Support': per_class_support
    })
    
    # Format the dataframe for better display
    metrics_df['Precision'] = metrics_df['Precision'].apply(lambda x: f"{x:.4f}")
    metrics_df['Recall'] = metrics_df['Recall'].apply(lambda x: f"{x:.4f}")
    metrics_df['F1-Score'] = metrics_df['F1-Score'].apply(lambda x: f"{x:.4f}")
    
    print(metrics_df.to_string(index=False))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🌟 Feature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")
    
    # Model performance summary
    metrics_summary = {
        'accuracy': accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    
    return metrics_summary, feature_importance, cm, class_names

def main():
    """Main function to run the Random Forest analysis"""
    print("=== Random Forest Classification Analysis ===")
    
    # File path
    file_path = "DATA.csv"
    
    try:
        # 1. Load and explore data
        df = load_and_explore_data(file_path)
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test, label_encoder, feature_columns = prepare_data(df)
        
        # 3. Build Random Forest model
        rf_model = build_random_forest(X_train, y_train)
        
        # 4. Evaluate model with comprehensive metrics
        metrics_summary, feature_importance, cm, class_names = evaluate_model(
            rf_model, X_test, y_test, label_encoder, feature_columns
        )
        
        print("\n" + "="*60)
        print("🏆 FINAL MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🎯 Primary Metric - Accuracy: {metrics_summary['accuracy']:.4f} ({metrics_summary['accuracy']*100:.2f}%)")
        print(f"📊 Weighted F1-Score: {metrics_summary['weighted_f1']:.4f} ({metrics_summary['weighted_f1']*100:.2f}%)")
        print(f"🎖️  Macro F1-Score: {metrics_summary['macro_f1']:.4f} ({metrics_summary['macro_f1']*100:.2f}%)")
        print("="*60)
        print("✅ Random Forest model has been successfully built and evaluated!")
        
        return rf_model, label_encoder, feature_columns, metrics_summary
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

if __name__ == "__main__":
    model, encoder, features, metrics = main()
    if metrics is not None:
        print(f"\n🎯 QUICK PERFORMANCE OVERVIEW:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   F1-Score (Weighted): {metrics['weighted_f1']:.4f} ({metrics['weighted_f1']*100:.2f}%)")
        print(f"   F1-Score (Macro): {metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)")
