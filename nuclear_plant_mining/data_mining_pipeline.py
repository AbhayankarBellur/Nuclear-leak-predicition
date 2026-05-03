# data_mining_pipeline.py
# Comprehensive Data Mining Pipeline for Nuclear Plant Leak Detection
import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                            mean_squared_error, r2_score, roc_auc_score,
                            precision_recall_curve, roc_curve)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import joblib

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directories
OUTPUT_DIR = "data/mining_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 NUCLEAR PLANT DATA MINING PIPELINE")
print("=" * 50)

# ===============================================
# STEP 1: LOAD AND PREPARE DATA
# ===============================================
print("\n📊 STEP 1: Loading Processed Dataset")

df = pd.read_csv("data/processed/nuclear_sensors_processed.csv", parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Create risk levels based on leak events and severity
def create_risk_levels(df):
    """Create risk categories based on leak events and sensor readings"""
    conditions = [
        (~df['ground_truth_leak_flag']) & (df['sump_flow_lpm'] <= 0.1),  # Normal
        (~df['ground_truth_leak_flag']) & (df['sump_flow_lpm'] > 0.1),   # Elevated
        (df['ground_truth_leak_flag']) & (df['ground_truth_leak_rate_lpm'] <= 20),  # Medium Risk
        (df['ground_truth_leak_flag']) & (df['ground_truth_leak_rate_lpm'] > 20)    # High Risk
    ]
    choices = ['Normal', 'Elevated', 'Medium Risk', 'High Risk']
    return np.select(conditions, choices, default='Normal')

df['risk_level'] = create_risk_levels(df)

# Key features for mining (select interpretable features)
key_features = [
    'sump_flow_lpm', 'airborne_gamma_uSvph', 'particulate_cpm',
    'makeup_flow_lpm', 'letdown_flow_lpm', 'rcs_mass_balance_lpm',
    'pressurizer_level_pct', 'pressurizer_pressure_MPa',
    'rcs_hot_leg_temp_C', 'rcs_cold_leg_temp_C', 'temp_gradient_C',
    'containment_humidity_pct', 'pump_vibration_mm_s',
    'radiation_index', 'hydraulic_imbalance_index', 'thermal_stress_index'
]

# Add PCA components
pca_cols = [col for col in df.columns if col.startswith('pc_')]
mining_features = key_features + pca_cols[:5]  # Top 5 PCA components

print(f"Selected {len(mining_features)} features for mining")
print(f"Risk level distribution:\n{df['risk_level'].value_counts()}")

# ===============================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================================
print("\n📈 STEP 2: Exploratory Data Analysis")

# Summary statistics
summary_stats = df[key_features].describe()
summary_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))

# Create comprehensive EDA plots
fig = plt.figure(figsize=(20, 15))

# 1. Correlation heatmap
plt.subplot(2, 3, 1)
corr_matrix = df[key_features[:12]].corr()  # Top 12 for readability
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, 
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap - Key Sensors")

# 2. Risk level distribution
plt.subplot(2, 3, 2)
risk_counts = df['risk_level'].value_counts()
colors = ['green', 'yellow', 'orange', 'red']
plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
        colors=colors[:len(risk_counts)])
plt.title("Risk Level Distribution")

# 3. Sump flow boxplot by risk level
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='risk_level', y='sump_flow_lpm')
plt.xticks(rotation=45)
plt.title("Sump Flow by Risk Level")

# 4. Radiation levels over time
plt.subplot(2, 3, 4)
df_sample = df.iloc[::100]  # Sample for performance
plt.plot(df_sample.index, df_sample['airborne_gamma_uSvph'], alpha=0.7, linewidth=0.8)
leak_events = df_sample[df_sample['ground_truth_leak_flag']]
plt.scatter(leak_events.index, leak_events['airborne_gamma_uSvph'], 
           color='red', alpha=0.8, s=10, label='Leak Events')
plt.title("Radiation Levels Over Time")
plt.legend()
plt.xticks(rotation=45)

# 5. Feature distributions
plt.subplot(2, 3, 5)
features_to_plot = ['sump_flow_lpm', 'radiation_index', 'hydraulic_imbalance_index']
for i, feature in enumerate(features_to_plot):
    plt.hist(df[feature].dropna(), bins=30, alpha=0.6, label=feature)
plt.legend()
plt.title("Key Feature Distributions")
plt.xlabel("Standardized Values")

# 6. PCA visualization
plt.subplot(2, 3, 6)
risk_colors = {'Normal': 'green', 'Elevated': 'yellow', 'Medium Risk': 'orange', 'High Risk': 'red'}
for risk in df['risk_level'].unique():
    mask = df['risk_level'] == risk
    if f'pc_01' in df.columns and f'pc_02' in df.columns:
        plt.scatter(df.loc[mask, 'pc_01'], df.loc[mask, 'pc_02'], 
                   c=risk_colors[risk], alpha=0.6, s=1, label=risk)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: PC1 vs PC2 by Risk Level")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_comprehensive.png"), dpi=150, bbox_inches='tight')
plt.close()

print("✅ EDA plots saved")

# ===============================================
# STEP 3: CLASSIFICATION - LEAK DETECTION
# ===============================================
print("\n🎯 STEP 3: Classification - Risk Level Prediction")

# Prepare data for classification
X_class = df[mining_features].fillna(df[mining_features].median())
y_class = df['risk_level']

# Convert to binary for some algorithms (leak vs no leak)
y_binary = (df['ground_truth_leak_flag'] == True).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_class, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Define classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Store results
classification_results = []

# Multi-class classification
print("Multi-class Classification (Risk Levels):")
for name, clf in classifiers.items():
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = clf.score(X_test, y_test)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        
        classification_results.append({
            'Algorithm': name,
            'Task': 'Multi-class',
            'Accuracy': accuracy,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std()
        })

        
        print(f"{name}: Accuracy = {accuracy:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        
        # Save detailed report for best performing model
        if name == 'Random Forest':
            print(f"\nDetailed Classification Report - {name}:")
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=clf.classes_, yticklabels=clf.classes_)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Feature importance
            if hasattr(clf, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': mining_features,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
                plt.title(f'Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    except Exception as e:
        print(f"Error with {name}: {e}")

# Binary classification (Leak Detection)
print(f"\nBinary Classification (Leak Detection):")
for name, clf in classifiers.items():
    try:
        clf.fit(X_train_bin, y_train_bin)
        y_pred_bin = clf.predict(X_test_bin)
        
        accuracy = clf.score(X_test_bin, y_test_bin)
        
        # AUC score
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test_bin)[:, 1]
            auc_score = roc_auc_score(y_test_bin, y_proba)
        else:
            y_proba = clf.decision_function(X_test_bin)
            auc_score = roc_auc_score(y_test_bin, y_proba)
        
        classification_results.append({
            'Algorithm': name,
            'Task': 'Binary',
            'Accuracy': accuracy,
            'AUC': auc_score,
            'CV_Mean': np.nan,
            'CV_Std': np.nan
        })
        
        print(f"{name}: Accuracy = {accuracy:.3f}, AUC = {auc_score:.3f}")
        
    except Exception as e:
        print(f"Error with {name}: {e}")

# Save classification results
results_df = pd.DataFrame(classification_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"), index=False)

# ===============================================
# STEP 4: REGRESSION - PREDICT LEAK RATE
# ===============================================
print("\n📊 STEP 4: Regression - Leak Rate Prediction")

# Prepare data for regression (only leak events)
leak_data = df[df['ground_truth_leak_flag'] == True].copy()
if len(leak_data) > 50:  # Ensure we have enough data
    
    X_reg = leak_data[mining_features].fillna(leak_data[mining_features].median())
    y_reg = leak_data['ground_truth_leak_rate_lpm']
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Define regressors
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    regression_results = []
    
    print("Leak Rate Prediction:")
    for name, reg in regressors.items():
        try:
            reg.fit(X_train_reg, y_train_reg)
            y_pred_reg = reg.predict(X_test_reg)
            
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            r2 = r2_score(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mse)
            
            regression_results.append({
                'Algorithm': name,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            })
            
            print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
            
            # Plot for best model
            if name == 'Random Forest':
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
                plt.plot([y_test_reg.min(), y_test_reg.max()], 
                        [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
                plt.xlabel('Actual Leak Rate (LPM)')
                plt.ylabel('Predicted Leak Rate (LPM)')
                plt.title(f'Regression Results - {name}')
                plt.savefig(os.path.join(OUTPUT_DIR, "regression_scatter.png"), 
                           dpi=150, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Save regression results
    reg_results_df = pd.DataFrame(regression_results)
    reg_results_df.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)

else:
    print("Insufficient leak data for regression analysis")

# ===============================================
# STEP 5: ASSOCIATION RULE MINING
# ===============================================
print("\n🔍 STEP 5: Association Rule Mining")

try:
    # Create binned versions for association rules
    df_assoc = df.copy()
    
    # Bin continuous variables into categories
    numeric_cols = ['sump_flow_lpm', 'airborne_gamma_uSvph', 'pressurizer_level_pct', 
                   'rcs_hot_leg_temp_C', 'radiation_index']
    
    for col in numeric_cols:
        if col in df_assoc.columns:
            df_assoc[f'{col}_bin'] = pd.cut(df_assoc[col], bins=3, 
                                          labels=[f'{col}_Low', f'{col}_Med', f'{col}_High'])
    
    # Add risk level
    df_assoc['risk_high'] = (df_assoc['risk_level'].isin(['Medium Risk', 'High Risk']))
    df_assoc['leak_event'] = df_assoc['ground_truth_leak_flag']
    
    # Select categorical columns for association rules
    assoc_cols = [col for col in df_assoc.columns if col.endswith('_bin')] + \
                ['risk_high', 'leak_event']
    
    # One-hot encode
    df_encoded = pd.get_dummies(df_assoc[assoc_cols])
    df_encoded = df_encoded.astype(bool)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True, max_len=3)
    
    if len(frequent_itemsets) > 0:
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
        
        # Filter for interesting rules (involving leak events or high risk)
        interesting_rules = rules[
            (rules['antecedents'].astype(str).str.contains('leak_event_True|risk_high_True')) |
            (rules['consequents'].astype(str).str.contains('leak_event_True|risk_high_True'))
        ]
        
        if len(interesting_rules) > 0:
            # Sort by confidence and lift
            top_rules = interesting_rules.sort_values(['confidence', 'lift'], ascending=False).head(20)
            
            # Save rules
            top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv(
                os.path.join(OUTPUT_DIR, "association_rules.csv"), index=False
            )
            
            print("Top Association Rules:")
            for idx, rule in top_rules.head(5).iterrows():
                antecedent = list(rule['antecedents'])[0] if rule['antecedents'] else "None"
                consequent = list(rule['consequents'])[0] if rule['consequents'] else "None"
                print(f"{antecedent} → {consequent}")
                print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
        else:
            print("No interesting association rules found")
    else:
        print("No frequent itemsets found")

except Exception as e:
    print(f"Error in association rule mining: {e}")

# ===============================================
# STEP 6: ADVANCED DIMENSIONALITY REDUCTION
# ===============================================
print("\n🎨 STEP 6: Dimensionality Reduction Visualization")

# Prepare data for PCA visualization
X_viz = df[key_features].fillna(df[key_features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_viz)

# Apply PCA
pca_viz = PCA(n_components=3)
X_pca_viz = pca_viz.fit_transform(X_scaled)

# Create visualization plots
fig = plt.figure(figsize=(15, 10))

# 2D PCA plot
plt.subplot(2, 2, 1)
risk_colors = {'Normal': 'green', 'Elevated': 'gold', 'Medium Risk': 'orange', 'High Risk': 'red'}
for risk in df['risk_level'].unique():
    mask = df['risk_level'] == risk
    plt.scatter(X_pca_viz[mask, 0], X_pca_viz[mask, 1], 
               c=risk_colors[risk], alpha=0.6, s=10, label=risk)
plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA: Risk Levels')
plt.legend()

# Explained variance
plt.subplot(2, 2, 2)
plt.bar(range(1, len(pca_viz.explained_variance_ratio_) + 1), 
        pca_viz.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')

# Loading plot (feature contributions)
plt.subplot(2, 2, 3)
loadings = pca_viz.components_.T * np.sqrt(pca_viz.explained_variance_)
for i, feature in enumerate(key_features):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
              head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, 
            fontsize=8, ha='center', va='center')
plt.xlabel('PC1 Loading')
plt.ylabel('PC2 Loading')
plt.title('PCA Feature Loadings')
plt.grid(True, alpha=0.3)

# Time series of first principal component
plt.subplot(2, 2, 4)
df_sample = df.iloc[::50]  # Sample for performance
pc1_sample = X_pca_viz[::50, 0]
plt.plot(df_sample.index, pc1_sample, alpha=0.7, linewidth=1)
leak_mask = df_sample['ground_truth_leak_flag']
plt.scatter(df_sample.index[leak_mask], pc1_sample[leak_mask], 
           color='red', alpha=0.8, s=20, label='Leak Events')
plt.xlabel('Time')
plt.ylabel('PC1 Score')
plt.title('PC1 Over Time (Leak Events Highlighted)')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dimensionality_reduction.png"), 
           dpi=150, bbox_inches='tight')
plt.close()

# ===============================================
# STEP 7: MODEL COMPARISON SUMMARY
# ===============================================
print("\n📋 STEP 7: Model Comparison Summary")

# Create comprehensive results table
if 'results_df' in locals() and len(results_df) > 0:
    print("\nClassification Results Summary:")
    print(results_df.to_string(index=False))
    
    # Best performing models
    best_multiclass = results_df[results_df['Task'] == 'Multi-class'].loc[
        results_df[results_df['Task'] == 'Multi-class']['Accuracy'].idxmax()
    ]
    best_binary = results_df[results_df['Task'] == 'Binary'].loc[
        results_df[results_df['Task'] == 'Binary']['AUC'].idxmax()
    ]
    
    print(f"\nBest Multi-class Model: {best_multiclass['Algorithm']} (Accuracy: {best_multiclass['Accuracy']:.3f})")
    print(f"Best Binary Model: {best_binary['Algorithm']} (AUC: {best_binary['AUC']:.3f})")

if 'reg_results_df' in locals() and len(reg_results_df) > 0:
    print(f"\nRegression Results Summary:")
    print(reg_results_df.to_string(index=False))
    
    best_regression = reg_results_df.loc[reg_results_df['R²'].idxmax()]
    print(f"Best Regression Model: {best_regression['Algorithm']} (R²: {best_regression['R²']:.3f})")

# Final summary plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Classification accuracy comparison
if 'results_df' in locals() and len(results_df) > 0:
    multiclass_results = results_df[results_df['Task'] == 'Multi-class']
    ax1.bar(multiclass_results['Algorithm'], multiclass_results['Accuracy'])
    ax1.set_title('Classification Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)

# Regression performance comparison
if 'reg_results_df' in locals() and len(reg_results_df) > 0:
    ax2.bar(reg_results_df['Algorithm'], reg_results_df['R²'])
    ax2.set_title('Regression R² Comparison')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)

# Risk level distribution over time
df_daily = df.groupby([df.index.date, 'risk_level']).size().unstack(fill_value=0)
df_daily.plot(kind='area', stacked=True, ax=ax3, 
              color=['green', 'gold', 'orange', 'red'])
ax3.set_title('Risk Level Distribution Over Time')
ax3.set_xlabel('Date')
ax3.set_ylabel('Count')

# Feature correlation with leak events
leak_corr = df[key_features + ['ground_truth_leak_flag']].corr()['ground_truth_leak_flag'].abs().sort_values(ascending=False)[1:11]
ax4.barh(range(len(leak_corr)), leak_corr.values)
ax4.set_yticks(range(len(leak_corr)))
ax4.set_yticklabels(leak_corr.index)
ax4.set_title('Feature Correlation with Leak Events')
ax4.set_xlabel('Absolute Correlation')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_summary.png"), 
           dpi=150, bbox_inches='tight')
plt.close()

# ===============================================
# FINAL SUMMARY
# ===============================================
print("\n" + "="*50)
print("🎉 DATA MINING PIPELINE COMPLETED!")
print("="*50)
print(f"✅ Results saved to: {OUTPUT_DIR}")
print(f"✅ Dataset analyzed: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✅ Mining features: {len(mining_features)}")
print(f"✅ Risk levels identified: {df['risk_level'].nunique()}")
print(f"✅ Leak events detected: {df['ground_truth_leak_flag'].sum()}")

# List all generated files
print(f"\n📁 Generated Files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {filename}")

print(f"\n🚀 Ready for presentation and analysis!")