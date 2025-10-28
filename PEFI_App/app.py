import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# For XGBoost (install: pip install xgboost)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# For Neural Network
from sklearn.neural_network import MLPRegressor

# For Classification approach
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("="*70)
print("🌱 PRODUCT ECO-FRIENDLINESS INDEX (PEFI) - ML PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 STEP 1: Loading Dataset...")

# Load the dataset (make sure to run the dataset generator first!)
try:
    df = pd.read_csv('PEFI_Indian_Products_Dataset.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Total products: {len(df)}")
    print(f"   - Features: {len(df.columns)}")
except FileNotFoundError:
    print("❌ Dataset file not found. Please run the dataset generator first!")
    exit()

# Display basic info
print(f"\n📊 Dataset Overview:")
print(df.head())
print(f"\n📈 Dataset Info:")
print(df.info())
print(f"\n📉 Missing Values:")
print(df.isnull().sum())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("📊 STEP 2: Exploratory Data Analysis")
print("="*70)

# Target variable distribution
print(f"\n🎯 Target Variable: Eco_Friendliness_Index")
print(f"   - Mean: {df['Eco_Friendliness_Index'].mean():.2f}")
print(f"   - Median: {df['Eco_Friendliness_Index'].median():.2f}")
print(f"   - Std Dev: {df['Eco_Friendliness_Index'].std():.2f}")
print(f"   - Range: [{df['Eco_Friendliness_Index'].min()}, {df['Eco_Friendliness_Index'].max()}]")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Target distribution
axes[0, 0].hist(df['Eco_Friendliness_Index'], bins=30, color='green', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Eco-Friendliness Index Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Eco-Friendliness Index')
axes[0, 0].set_ylabel('Frequency')

# 2. Category distribution
category_counts = df['Category'].value_counts()
axes[0, 1].barh(category_counts.index, category_counts.values, color='skyblue')
axes[0, 1].set_title('Products by Category', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Count')

# 3. Material Type vs Eco Index
material_eco = df.groupby('Material_Type')['Eco_Friendliness_Index'].mean().sort_values()
axes[0, 2].barh(material_eco.index, material_eco.values, color='coral')
axes[0, 2].set_title('Avg Eco-Index by Material', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Avg Eco-Friendliness Index')

# 4. Carbon Footprint vs Eco Index
axes[1, 0].scatter(df['Carbon_Footprint_kgCO2'], df['Eco_Friendliness_Index'], 
                   alpha=0.5, c=df['Eco_Friendliness_Index'], cmap='RdYlGn')
axes[1, 0].set_title('Carbon Footprint vs Eco-Index', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Carbon Footprint (kg CO2)')
axes[1, 0].set_ylabel('Eco-Friendliness Index')

# 5. Recyclability vs Eco Index
axes[1, 1].scatter(df['Recyclability_Percent'], df['Eco_Friendliness_Index'], 
                   alpha=0.5, c=df['Eco_Friendliness_Index'], cmap='RdYlGn')
axes[1, 1].set_title('Recyclability vs Eco-Index', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Recyclability %')
axes[1, 1].set_ylabel('Eco-Friendliness Index')

# 6. Source Country distribution
country_counts = df['Source_Country'].value_counts().head(8)
axes[1, 2].bar(range(len(country_counts)), country_counts.values, color='gold')
axes[1, 2].set_xticks(range(len(country_counts)))
axes[1, 2].set_xticklabels(country_counts.index, rotation=45, ha='right')
axes[1, 2].set_title('Top Source Countries', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('PEFI_EDA.png', dpi=300, bbox_inches='tight')
print(f"\n✅ EDA visualizations saved as 'PEFI_EDA.png'")

# ============================================================================
# STEP 3: DATA PREPROCESSING & ENCODING
# ============================================================================
print("\n" + "="*70)
print("🔧 STEP 3: Data Preprocessing & Encoding")
print("="*70)

# Create a copy for processing
df_processed = df.copy()

# Drop Product_ID and Product_Name (not useful for prediction)
X = df_processed.drop(['Product_ID', 'Product_Name', 'Eco_Friendliness_Index'], axis=1)
y = df_processed['Eco_Friendliness_Index']

print(f"\n📋 Features selected: {list(X.columns)}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n🏷️  Categorical features: {categorical_cols}")
print(f"🔢 Numerical features: {numerical_cols}")

# Encode categorical variables
print(f"\n⚙️  Encoding categorical variables...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"   - {col}: {len(le.classes_)} unique values encoded")

# Check for any remaining non-numeric data
print(f"\n✅ All features encoded. Data types:")
print(X.dtypes)

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("✂️ STEP 4: Train-Test Split")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Dataset Split:")
print(f"   - Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Feature Scaling (important for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Feature scaling completed")

# ============================================================================
# STEP 5: MODEL TRAINING - REGRESSION
# ============================================================================
print("\n" + "="*70)
print("🤖 STEP 5: Training Regression Models")
print("="*70)

models = {}
results = {}

# -------------------------
# Model 1: Random Forest Regressor
# -------------------------
print(f"\n🌲 Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'RMSE': rf_rmse,
    'MAE': rf_mae,
    'R²': rf_r2,
    'predictions': rf_pred
}

print(f"   ✅ Random Forest Results:")
print(f"      - RMSE: {rf_rmse:.2f}")
print(f"      - MAE: {rf_mae:.2f}")
print(f"      - R² Score: {rf_r2:.4f}")

# -------------------------
# Model 2: XGBoost (if available)
# -------------------------
if XGBOOST_AVAILABLE:
    print(f"\n🚀 Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'RMSE': xgb_rmse,
        'MAE': xgb_mae,
        'R²': xgb_r2,
        'predictions': xgb_pred
    }
    
    print(f"   ✅ XGBoost Results:")
    print(f"      - RMSE: {xgb_rmse:.2f}")
    print(f"      - MAE: {xgb_mae:.2f}")
    print(f"      - R² Score: {xgb_r2:.4f}")

# -------------------------
# Model 3: Neural Network
# -------------------------
print(f"\n🧠 Training Neural Network (MLP)...")
nn_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)

nn_mse = mean_squared_error(y_test, nn_pred)
nn_rmse = np.sqrt(nn_mse)
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)

models['Neural Network'] = nn_model
results['Neural Network'] = {
    'RMSE': nn_rmse,
    'MAE': nn_mae,
    'R²': nn_r2,
    'predictions': nn_pred
}

print(f"   ✅ Neural Network Results:")
print(f"      - RMSE: {nn_rmse:.2f}")
print(f"      - MAE: {nn_mae:.2f}")
print(f"      - R² Score: {nn_r2:.4f}")

# ============================================================================
# STEP 6: MODEL COMPARISON
# ============================================================================
print("\n" + "="*70)
print("📊 STEP 6: Model Comparison")
print("="*70)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['RMSE'] for m in results.keys()],
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'R² Score': [results[m]['R²'] for m in results.keys()]
})

print(f"\n{comparison_df.to_string(index=False)}")

# Find best model
best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name} (Highest R² Score: {comparison_df['R² Score'].max():.4f})")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE (Random Forest)
# ============================================================================
print("\n" + "="*70)
print("📈 STEP 7: Feature Importance Analysis")
print("="*70)

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🔍 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 8: VISUALIZE RESULTS
# ============================================================================
print("\n" + "="*70)
print("📊 STEP 8: Visualizing Results")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(comparison_df))
ax1.bar(x_pos, comparison_df['R² Score'], color=['#2ecc71', '#3498db', '#e74c3c'][:len(comparison_df)])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance Comparison (R² Score)', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1])
for i, v in enumerate(comparison_df['R² Score']):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# 2. Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(10)
ax2.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['Feature'])
ax2.set_xlabel('Importance Score')
ax2.set_title('Top 10 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

# 3. Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
best_predictions = results[best_model_name]['predictions']
ax3.scatter(y_test, best_predictions, alpha=0.6, color='purple')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Eco-Friendliness Index')
ax3.set_ylabel('Predicted Eco-Friendliness Index')
ax3.set_title(f'Actual vs Predicted ({best_model_name})', fontsize=12, fontweight='bold')
ax3.text(0.05, 0.95, f'R² = {results[best_model_name]["R²"]:.4f}', 
         transform=ax3.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Residual Plot
ax4 = axes[1, 1]
residuals = y_test - best_predictions
ax4.scatter(best_predictions, residuals, alpha=0.6, color='orange')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Eco-Friendliness Index')
ax4.set_ylabel('Residuals')
ax4.set_title(f'Residual Plot ({best_model_name})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('PEFI_Model_Results.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Model results visualizations saved as 'PEFI_Model_Results.png'")

# ============================================================================
# STEP 9: CLASSIFICATION APPROACH (BONUS)
# ============================================================================
print("\n" + "="*70)
print("🎯 STEP 9: Classification Approach (Eco-Rating Categories)")
print("="*70)

# Create categories: Poor (0-40), Fair (41-60), Good (61-80), Excellent (81-100)
def categorize_eco_index(score):
    if score <= 40:
        return 'Poor'
    elif score <= 60:
        return 'Fair'
    elif score <= 80:
        return 'Good'
    else:
        return 'Excellent'

y_train_cat = y_train.apply(categorize_eco_index)
y_test_cat = y_test.apply(categorize_eco_index)

print(f"\n📊 Category Distribution:")
print(y_train_cat.value_counts())

# Train Random Forest Classifier
print(f"\n🌲 Training Random Forest Classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_classifier.fit(X_train, y_train_cat)
y_pred_cat = rf_classifier.predict(X_test)

# Classification metrics
accuracy = accuracy_score(y_test_cat, y_pred_cat)
print(f"\n✅ Classification Accuracy: {accuracy:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test_cat, y_pred_cat))

# Confusion Matrix
cm = confusion_matrix(y_test_cat, y_pred_cat, labels=['Poor', 'Fair', 'Good', 'Excellent'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Poor', 'Fair', 'Good', 'Excellent'],
            yticklabels=['Poor', 'Fair', 'Good', 'Excellent'])
plt.title('Confusion Matrix - Eco-Rating Classification', fontsize=14, fontweight='bold')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.tight_layout()
plt.savefig('PEFI_Classification_ConfusionMatrix.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Confusion matrix saved as 'PEFI_Classification_ConfusionMatrix.png'")

# ============================================================================
# STEP 10: SAVE MODELS & ENCODERS
# ============================================================================
print("\n" + "="*70)
print("💾 STEP 10: Saving Models & Encoders")
print("="*70)

import pickle

# Save the best regression model
with open('best_pefi_model.pkl', 'wb') as f:
    pickle.dump(models[best_model_name], f)
print(f"✅ Best regression model saved: best_pefi_model.pkl")

# Save classifier
with open('pefi_classifier.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)
print(f"✅ Classifier model saved: pefi_classifier.pkl")

# Save encoders and scaler
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Label encoders saved: label_encoders.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved: scaler.pkl")

# ============================================================================
# STEP 11: EXAMPLE PREDICTION
# ============================================================================
print("\n" + "="*70)
print("🔮 STEP 11: Example Prediction")
print("="*70)

# Take a sample product from test set
sample_idx = 0
sample_product = X_test.iloc[sample_idx:sample_idx+1]
actual_score = y_test.iloc[sample_idx]

# Predict using best model
if best_model_name == 'Neural Network':
    sample_scaled = scaler.transform(sample_product)
    predicted_score = models[best_model_name].predict(sample_scaled)[0]
else:
    predicted_score = models[best_model_name].predict(sample_product)[0]

predicted_category = categorize_eco_index(predicted_score)
actual_category = categorize_eco_index(actual_score)

print(f"\n🎯 Sample Prediction:")
print(f"   - Actual Eco-Index: {actual_score:.1f} ({actual_category})")
print(f"   - Predicted Eco-Index: {predicted_score:.1f} ({predicted_category})")
print(f"   - Prediction Error: {abs(actual_score - predicted_score):.2f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✨ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)

print(f"\n📁 Files Generated:")
print(f"   1. PEFI_EDA.png - Exploratory Data Analysis")
print(f"   2. PEFI_Model_Results.png - Model Performance & Predictions")
print(f"   3. PEFI_Classification_ConfusionMatrix.png - Classification Results")
print(f"   4. best_pefi_model.pkl - Best Regression Model")
print(f"   5. pefi_classifier.pkl - Classification Model")
print(f"   6. label_encoders.pkl - Feature Encoders")
print(f"   7. scaler.pkl - Feature Scaler")

print(f"\n🏆 Best Performing Model: {best_model_name}")
print(f"   - R² Score: {results[best_model_name]['R²']:.4f}")
print(f"   - RMSE: {results[best_model_name]['RMSE']:.2f}")
print(f"   - MAE: {results[best_model_name]['MAE']:.2f}")

print(f"\n🎯 Classification Accuracy: {accuracy:.4f}")

print("\n" + "="*70)
print("🌱 Your PEFI Model is Ready for Deployment!")
print("="*70)
