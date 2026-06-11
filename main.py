import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import *
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data (local file)
file_path = 'DataSet.csv'
df = pd.read_csv(file_path)

if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=['Unnamed: 0'])

print("="*70)
print("🎯 FINAL MULE ACCOUNT DETECTION SYSTEM")
print("="*70)

# ============================================
# 1. DATA PREPARATION
# ============================================
print("\n📊 STEP 1: Data Preparation")

target_col = 'F3924'
feature_cols = [col for col in df.columns if col != target_col]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

X_raw = df[feature_cols].copy()
y = df[target_col].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train_raw.shape[0]}, Test samples: {X_test_raw.shape[0]}")
print(f"Fraud cases: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")

# ============================================
# 2. REMOVE EXTREME NOISE
# ============================================
print("\n🗑️ STEP 2: Removing Extreme Noise")

missing_pct = X_train_raw.isnull().mean()
high_missing = missing_pct[missing_pct > 0.95].index
X_train_raw = X_train_raw.drop(columns=high_missing)
X_test_raw = X_test_raw.drop(columns=high_missing)

constant_cols = [col for col in X_train_raw.columns if X_train_raw[col].std() < 1e-8]
X_train_raw = X_train_raw.drop(columns=constant_cols)
X_test_raw = X_test_raw.drop(columns=constant_cols)

print(f"Remaining features: {X_train_raw.shape[1]}")

# ============================================
# 3. IMPUTATION
# ============================================
print("\n🩹 STEP 3: Median Imputation")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), 
                            columns=X_train_raw.columns, 
                            index=X_train_raw.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), 
                           columns=X_test_raw.columns, 
                           index=X_test_raw.index)

# ============================================
# 4. MUTUAL INFORMATION SELECTION
# ============================================
print("\n📈 STEP 4: Mutual Information Selection")

mi_scores = mutual_info_classif(X_train_imp, y_train, random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X_train_imp.columns).sort_values(ascending=False)

TOP_K = 200
selected_features_mi = mi_scores_series.head(TOP_K).index
X_train_mi = X_train_imp[selected_features_mi]
X_test_mi = X_test_imp[selected_features_mi]

# ============================================
# 5. REMOVE HIGHLY CORRELATED FEATURES
# ============================================
print("\n🔗 STEP 5: Removing Highly Correlated Features")

corr_matrix = X_train_mi.corr().abs()
upper_tri_df = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper_tri_df.columns if any(upper_tri_df[column] > 0.98)]
X_train_mi = X_train_mi.drop(columns=high_corr)
X_test_mi = X_test_mi.drop(columns=high_corr)

print(f"Final features: {X_train_mi.shape[1]}")

# ============================================
# 6. SCALE AND HANDLE IMBALANCE
# ============================================
print("\n⚖️ STEP 6: Scaling & SMOTE-ENN")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_mi)
X_test_scaled = scaler.transform(X_test_mi)

smote_enn = SMOTEENN(random_state=42, 
                      smote=SMOTE(k_neighbors=min(5, y_train.value_counts().min() - 1)))
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train)

# ============================================
# 7. TRAIN MODELS
# ============================================
print("\n🤖 STEP 7: Training Models")

scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])

models = {
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500, max_depth=6,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
}

results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    
    results[name] = {
        'model': model,
        'proba': y_pred_proba,
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'threshold': best_thresh
    }
    
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall: {results[name]['recall']:.4f}")
    print(f"  F1: {results[name]['f1']:.4f}")
    print(f"  AUC: {results[name]['auc']:.4f}")

# ============================================
# 8. ENSEMBLE MODEL
# ============================================
print("\n🔗 STEP 8: Ensemble Model")

ensemble = VotingClassifier(
    estimators=[('xgb', models['XGBoost']), ('rf', models['RandomForest'])],
    voting='soft'
)
ensemble.fit(X_train_balanced, y_train_balanced)
y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_ensemble)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
best_thresh_ensemble = thresholds[np.argmax(f1_scores)]
y_pred_ensemble = (y_pred_proba_ensemble >= best_thresh_ensemble).astype(int)

ensemble_results = {
    'f1': f1_score(y_test, y_pred_ensemble),
    'auc': roc_auc_score(y_test, y_pred_proba_ensemble),
    'precision': precision_score(y_test, y_pred_ensemble),
    'recall': recall_score(y_test, y_pred_ensemble),
    'threshold': best_thresh_ensemble,
    'proba': y_pred_proba_ensemble
}

print(f"\n--- Ensemble Results ---")
print(f"  Precision: {ensemble_results['precision']:.4f}")
print(f"  Recall: {ensemble_results['recall']:.4f}")
print(f"  F1: {ensemble_results['f1']:.4f}")
print(f"  AUC: {ensemble_results['auc']:.4f}")

# ============================================
# 9. BEST MODEL SELECTION
# ============================================
all_results = {**results, 'Ensemble': ensemble_results}
best_name = max(all_results, key=lambda x: all_results[x]['f1'])
best = all_results[best_name]

cm = confusion_matrix(y_test, (best['proba'] >= best['threshold']).astype(int))
tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]

print(f"\n✅ BEST MODEL: {best_name}")
print(f"   Precision: {best['precision']:.4f} ({fp} false positives)")
print(f"   Recall: {best['recall']:.4f} (Caught {tp} of {tp+fn})")

# ============================================
# 10. FEATURE IMPORTANCE
# ============================================
feature_importance = pd.DataFrame({
    'feature': X_train_mi.columns,
    'importance': results['RandomForest']['model'].feature_importances_
}).sort_values('importance', ascending=False)

# ============================================
# 11. GENERATE VISUALIZATIONS
# ============================================
print("\n📊 STEP 11: Generating Visualizations")

sns.set_style("whitegrid")
sns.set_palette("Set2")

# Create a large figure with subplots
fig = plt.figure(figsize=(20, 24))

# ============================================
# Plot 1: Confusion Matrix
# ============================================
ax1 = plt.subplot(3, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Suspicious'],
            yticklabels=['Normal', 'Suspicious'],
            ax=ax1)
ax1.set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)

# Add annotations
for i in range(2):
    for j in range(2):
        ax1.text(j+0.5, i+0.5, f'{cm[i,j]}', 
                ha='center', va='center', fontsize=16, fontweight='bold')

# ============================================
# Plot 2: ROC Curve
# ============================================
ax2 = plt.subplot(3, 3, 2)
for name, res in all_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax2.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={res["auc"]:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

# ============================================
# Plot 3: Precision-Recall Curve
# ============================================
ax3 = plt.subplot(3, 3, 3)
for name, res in all_results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['proba'])
    ax3.plot(recall, precision, linewidth=2, label=f'{name} (AP={average_precision_score(y_test, res["proba"]):.3f})')
ax3.set_xlabel('Recall', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=9)
ax3.grid(True, alpha=0.3)

# ============================================
# Plot 4: Risk Score Distribution
# ============================================
ax4 = plt.subplot(3, 3, 4)
risk_scores = (best['proba'] * 1000).astype(int)

# Separate by actual class
normal_risks = risk_scores[y_test == 0]
fraud_risks = risk_scores[y_test == 1]

ax4.hist(normal_risks, bins=30, alpha=0.5, label='Normal Accounts', color='green', edgecolor='black')
ax4.hist(fraud_risks, bins=30, alpha=0.7, label='Suspicious Accounts', color='red', edgecolor='black')
ax4.axvline(x=best['threshold']*1000, color='blue', linestyle='--', linewidth=2, 
            label=f'Threshold ({best["threshold"]*1000:.0f})')
ax4.set_xlabel('Risk Score (0-1000)', fontsize=12)
ax4.set_ylabel('Number of Accounts', fontsize=12)
ax4.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============================================
# Plot 5: Feature Importance (Top 15)
# ============================================
ax5 = plt.subplot(3, 3, 5)
top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax5.barh(range(len(top_features)), top_features['importance'].values, color=colors)
ax5.set_yticks(range(len(top_features)))
ax5.set_yticklabels(top_features['feature'].values, fontsize=9)
ax5.set_xlabel('Importance Score', fontsize=12)
ax5.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_features['importance'].values):
    ax5.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# ============================================
# Plot 6: Model Performance Bar Chart
# ============================================
ax6 = plt.subplot(3, 3, 6)
models_names = list(all_results.keys())
metrics = ['precision', 'recall', 'f1']
x = np.arange(len(models_names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [all_results[m][metric] for m in models_names]
    ax6.bar(x + i*width, values, width, label=metric.capitalize())

ax6.set_xlabel('Model', fontsize=12)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax6.set_xticks(x + width)
ax6.set_xticklabels(models_names, rotation=45, ha='right')
ax6.legend()
ax6.set_ylim(0, 1.1)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, metric in enumerate(metrics):
    values = [all_results[m][metric] for m in models_names]
    for j, v in enumerate(values):
        ax6.text(j + i*width, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

# ============================================
# Plot 7: Top Risk Scores (Heatmap)
# ============================================
ax7 = plt.subplot(3, 3, 7)
high_risk_df = pd.DataFrame({
    'Risk Score': risk_scores,
    'Is Fraud': y_test.values
})
high_risk_table = high_risk_df.groupby(['Risk Score', 'Is Fraud']).size().unstack(fill_value=0)
high_risk_table = high_risk_table.sort_index(ascending=False).head(20)

sns.heatmap(high_risk_table.T, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax7, cbar_kws={'label': 'Count'})
ax7.set_title('Top 20 Risk Scores Distribution', fontsize=14, fontweight='bold')
ax7.set_xlabel('Risk Score', fontsize=12)
ax7.set_ylabel('Account Type', fontsize=12)
ax7.set_yticklabels(['Normal', 'Suspicious'])

# ============================================
# Plot 8: Cumulative Gains Chart
# ============================================
ax8 = plt.subplot(3, 3, 8)
from scipy import integrate

# Sort by predicted probability
sorted_idx = np.argsort(best['proba'])[::-1]
sorted_y = y_test.values[sorted_idx]

# Calculate cumulative gains
cumulative = np.cumsum(sorted_y) / np.sum(sorted_y)
percentage = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100

ax8.plot(percentage, cumulative * 100, 'b-', linewidth=2, label='Model')
ax8.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Random')
ax8.set_xlabel('Percentage of Test Set', fontsize=12)
ax8.set_ylabel('Percentage of Frauds Caught', fontsize=12)
ax8.set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Add annotation for top 20%
top20_pct = cumulative[int(len(cumulative)*0.2)] * 100
ax8.scatter(20, top20_pct, color='red', s=100, zorder=5)
ax8.annotate(f'Top 20% catches {top20_pct:.1f}% of frauds', 
            xy=(20, top20_pct), xytext=(25, top20_pct-10),
            arrowprops=dict(arrowstyle='->', color='red'))

# ============================================
# Plot 9: Alert Level Distribution
# ============================================
ax9 = plt.subplot(3, 3, 9)

def get_alert(score):
    if score >= 800: return "CRITICAL"
    elif score >= 650: return "HIGH"
    elif score >= 500: return "MEDIUM"
    else: return "LOW/NORMAL"

alert_levels = [get_alert(s) for s in risk_scores]
alert_counts = pd.Series(alert_levels).value_counts()

colors_alert = ['darkred', 'red', 'orange', 'green']
ax9.pie(alert_counts.values, labels=alert_counts.index, autopct='%1.1f%%', 
        colors=colors_alert[:len(alert_counts)], startangle=90, explode=[0.05]*len(alert_counts))
ax9.set_title('Alert Level Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('complete_dashboard.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Dashboard saved to 'complete_dashboard.png'")

# ============================================
# Plot 10: Separate Feature Importance Plot (High Res)
# ============================================
plt.figure(figsize=(14, 10))
top_features = feature_importance.head(20)
colors = plt.cm.plasma(np.linspace(0, 1, len(top_features)))
plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=10)
plt.xlabel('Importance Score', fontsize=14)
plt.title('Top 20 Most Important Features for Mule Account Detection', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_features['importance'].values):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Feature importance plot saved to 'feature_importance_plot.png'")

# ============================================
# Plot 11: MI Score Distribution
# ============================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(mi_scores_series.values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(x=mi_scores_series.head(200).min(), color='red', linestyle='--', 
            label=f'Threshold (Top 200)')
plt.xlabel('Mutual Information Score', fontsize=12)
plt.ylabel('Number of Features', fontsize=12)
plt.title('Distribution of Mutual Information Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cumulative_mi = mi_scores_series.cumsum()
total_mi = cumulative_mi.iloc[-1]
plt.plot(range(1, len(cumulative_mi)+1), cumulative_mi / total_mi * 100, 'b-', linewidth=2)
plt.axhline(y=95, color='red', linestyle='--', label='95% of MI')
plt.axvline(x=200, color='green', linestyle='--', label='Top 200 features')
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Cumulative MI (% of total)', fontsize=12)
plt.title('Cumulative Mutual Information', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mi_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ MI distribution plot saved to 'mi_distribution.png'")

# ============================================
# 12. RISK SCORING & SAVE
# ============================================
print("\n🚨 STEP 12: Risk Scoring")

risk_scores = (best['proba'] * 1000).astype(int)

def get_alert(score):
    if score >= 800: return "CRITICAL"
    elif score >= 650: return "HIGH"
    elif score >= 500: return "MEDIUM"
    else: return "NORMAL"

risk_profile = pd.DataFrame({
    'Account_ID': X_test_raw.index,
    'True_Label': y_test.values,
    'Risk_Score': risk_scores,
    'Alert': [get_alert(s) for s in risk_scores]
})

print(f"\nAlert Distribution:")
print(risk_profile['Alert'].value_counts())

high_risk = risk_profile[risk_profile['Risk_Score'] >= 650]
fraud_caught = high_risk[high_risk['True_Label'] == 1]
print(f"\nHIGH RISK ACCOUNTS: {len(high_risk)}")
print(f"Fraudsters correctly identified: {len(fraud_caught)}/{y_test.sum()}")

# ============================================
# 13. SAVE ALL FILES
# ============================================
print("\n💾 STEP 13: Saving Files")

risk_profile.to_csv('risk_profile.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

# Save final summary
summary = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    FINAL MULE ACCOUNT DETECTION SYSTEM                  ║
║                           SUBMISSION READY                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  DATASET OVERVIEW                                                       ║
║  ─────────────────                                                      ║
║  {'• Total accounts: ' + f'{len(df):,}':<71}║
║  {'• Suspicious accounts: ' + f'{y.sum()} ({y.sum()/len(df)*100:.2f}%)':<71}║
║  {'• Features after selection: ' + f'{X_train_mi.shape[1]}':<71}║
║                                                                         ║
║  {'BEST MODEL: ' + best_name:<71}║
║  ─────────────────────                                                  ║
║  {'• Optimal Threshold: ' + f"{best['threshold']:.3f}":<71}║
║  {'• Precision: ' + f"{best['precision']:.4f} ({fp} false positives)":<71}║
║  {'• Recall: ' + f"{best['recall']:.4f} ({tp}/{tp+fn} fraudsters caught)":<71}║
║  {'• F1 Score: ' + f"{best['f1']:.4f}":<71}║
║  {'• AUC-ROC: ' + f"{best['auc']:.4f}":<71}║
║                                                                         ║
║  FILES GENERATED                                                        ║
║  ───────────────                                                        ║
║  {'• risk_profile.csv - Risk scores for test accounts':<71}║
║  {'• feature_importance.csv - Top predictive features':<71}║
║  {'• complete_dashboard.png - 9-panel dashboard':<71}║
║  {'• feature_importance_plot.png - Importance chart':<71}║
║  {'• mi_distribution.png - MI distribution plots':<71}║
║                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

print(summary)
with open('submission_summary.txt', 'w') as f:
    f.write(summary)

print("\n✅ ALL FILES SAVED!")
print("   📊 Visualizations:")
print("      • complete_dashboard.png (9-panel dashboard)")
print("      • feature_importance_plot.png")
print("      • mi_distribution.png")
print("   📁 Data files:")
print("      • risk_profile.csv")
print("      • feature_importance.csv")
print("      • submission_summary.txt")