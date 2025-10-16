"""
Anti Money Laundering Detection System - Model Training
This script trains and compares multiple ML models to detect suspicious transactions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os

# Import LightGBM and CatBoost
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("ANTI MONEY LAUNDERING DETECTION SYSTEM - MODEL TRAINING")
print("=" * 60)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1/10] Loading dataset...")
df = pd.read_csv('aml_large_dataset_10k.csv')
print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: CHECK DATA QUALITY
# ============================================================================
print("\n[STEP 2/10] Checking data quality...")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

# Check target variable
print(f"\nTarget variable distribution:")
print(df['label'].value_counts())
print(f"\nClass balance: {df['label'].value_counts(normalize=True) * 100}")

# ============================================================================
# STEP 3: CLEAN & PREPARE DATA
# ============================================================================
print("\n[STEP 3/10] Cleaning and preparing data...")

# Drop ID columns that are not useful for prediction
columns_to_drop = ['transaction_id', 'account_id']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(f"✓ Dropped columns: {[col for col in columns_to_drop if col in df.columns]}")

# Separate features and target
X = df_clean.drop('label', axis=1)
y = df_clean['label']

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical columns: {categorical_columns}")
print(f"Numerical columns: {numerical_columns}")

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {len(le.classes_)} unique values")

print(f"\n✓ Total features after encoding: {X.shape[1]}")

# ============================================================================
# STEP 4: CREATE CORRELATION HEATMAP 
# (To justify why 'label' column is selected for target)
# ============================================================================
print("\n[STEP 4/10] Creating correlation heatmap to justify target selection...")

# Combine numeric features + target label
corr_features = df_clean[numerical_columns + ['label']]

# Compute correlation matrix
corr_matrix = corr_features.corr()

# Sort by correlation strength with the target variable
corr_with_target = corr_matrix['label'].sort_values(ascending=False)
print("\n✓ Top correlations with target variable (label):")
print(corr_with_target)

# Identify strongest correlations
strong_correlations = corr_with_target[abs(corr_with_target) > 0.1]
print(f"\n✓ Features with correlation > 0.1: {len(strong_correlations)-1}")
print("This justifies 'label' as target - it has measurable relationships with input features")

# Plot heatmap with target emphasis
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient'},
    center=0
)
plt.title("Feature Correlation Heatmap\n(Justifying 'label' as Target Variable)", 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")
plt.show()

# ============================================================================
# STEP 5: SPLIT DATA
# ============================================================================
print("\n[STEP 5/10] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Suspicious in train: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"✓ Suspicious in test: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# STEP 6: SCALE FEATURES
# ============================================================================
print("\n[STEP 6/10] Scaling numeric features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# ============================================================================
# STEP 7: TRAIN 3 MODELS (RandomForest, LightGBM, CatBoost)
# ============================================================================
print("\n[STEP 7/10] Training 3 models...")
print("=" * 60)

models = {}
training_results = {}

# Model 1: Random Forest
print("\n[Model 1/3] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
models['RandomForest'] = rf_model
print("✓ Random Forest training completed!")

# Model 2: LightGBM
if LIGHTGBM_AVAILABLE:
    print("\n[Model 2/3] Training LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=15,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    models['LightGBM'] = lgb_model
    print("✓ LightGBM training completed!")
else:
    print("\n[Model 2/3] Skipping LightGBM (not installed)")

# Model 3: CatBoost
if CATBOOST_AVAILABLE:
    print("\n[Model 3/3] Training CatBoost Classifier...")
    cb_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    cb_model.fit(X_train_scaled, y_train)
    models['CatBoost'] = cb_model
    print("✓ CatBoost training completed!")
else:
    print("\n[Model 3/3] Skipping CatBoost (not installed)")

# ============================================================================
# STEP 8: EVALUATE MODELS AND SELECT BEST ONE
# ============================================================================
print("\n[STEP 8/10] Evaluating models and selecting best performer...")
print("=" * 60)

best_model_name = None
best_model = None
best_accuracy = 0
best_metrics = {}

for model_name, model in models.items():
    print(f"\n--- Evaluating {model_name} ---")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    training_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Track best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model
        best_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred
        }

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"🏆 Best Accuracy: {best_accuracy*100:.2f}%")
print("=" * 60)

# Display comparison table
print("\n📊 MODEL COMPARISON TABLE:")
print("-" * 80)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 80)
for model_name, metrics in training_results.items():
    marker = "🏆 " if model_name == best_model_name else "   "
    print(f"{marker}{model_name:<12} {metrics['accuracy']:.4f}      {metrics['precision']:.4f}      "
          f"{metrics['recall']:.4f}      {metrics['f1_score']:.4f}      {metrics['roc_auc']:.4f}")
print("-" * 80)

# ============================================================================
# STEP 9: DISPLAY RESULTS (for best model)
# ============================================================================
print(f"\n[STEP 9/10] Displaying detailed results for {best_model_name}...")
print("=" * 60)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_metrics['y_pred'])
print(cm)
print("\n[TN  FP]  (True Negative | False Positive)")
print("[FN  TP]  (False Negative | True Positive)")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, best_metrics['y_pred'], target_names=['Normal', 'Suspicious']))

# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Suspicious'],
            yticklabels=['Normal', 'Suspicious'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print(f"\n[STEP 10/10] Saving best model ({best_model_name})...")
print("=" * 60)

os.makedirs('model', exist_ok=True)

model_data = {
    'model': best_model,
    'model_name': best_model_name,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_columns': list(X.columns),
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns,
    'accuracy': best_metrics['accuracy'],
    'precision': best_metrics['precision'],
    'recall': best_metrics['recall'],
    'f1_score': best_metrics['f1_score'],
    'roc_auc': best_metrics['roc_auc'],
    'all_model_results': training_results
}

with open('model/aml_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved successfully to 'model/aml_model.pkl'")
print(f"✓ Model Type: {best_model_name}")
print(f"✓ Model Accuracy: {best_metrics['accuracy']*100:.2f}%")
print(f"✓ Precision: {best_metrics['precision']*100:.2f}%")
print(f"✓ Recall: {best_metrics['recall']*100:.2f}%")
print(f"✓ F1-Score: {best_metrics['f1_score']*100:.2f}%")
print(f"✓ ROC-AUC: {best_metrics['roc_auc']*100:.2f}%")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
