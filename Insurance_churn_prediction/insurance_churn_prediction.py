# ============================================================
# Insurance Customer Churn Prediction
# Author: Yuxuan Li
# Description: End-to-end ML pipeline to predict insurance
#              customer churn using Python and scikit-learn
#              Includes PCA for dimensionality reduction
# ============================================================

# ── 0. Import Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

import os
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("  INSURANCE CUSTOMER CHURN PREDICTION")
print("=" * 60)

# ── 1. Load & Explore Data ───────────────────────────────────
print("\n[1] Loading and exploring data...")
df = pd.read_csv('insurance_churn.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nChurn distribution:\n{df['churn'].value_counts()}")
print(f"Churn rate: {df['churn'].mean():.1%}")

# ── 2. EDA ───────────────────────────────────────────────────
print("\n[2] Creating EDA visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Insurance Churn – Exploratory Data Analysis', fontsize=16, fontweight='bold')

palette = {0: '#2E6DA4', 1: '#E05C2A'}
labels  = {0: 'Retained', 1: 'Churned'}

ax = axes[0, 0]
counts = df['churn'].value_counts()
bars = ax.bar(['Retained', 'Churned'], counts.values, color=['#2E6DA4', '#E05C2A'], edgecolor='white')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val), ha='center', fontweight='bold')
ax.set_title('Churn Distribution', fontweight='bold')
ax.set_ylabel('Count')

ax = axes[0, 1]
for churn_val, group in df.groupby('churn'):
    ax.hist(group['tenure_years'], bins=15, alpha=0.6, label=labels[churn_val], color=palette[churn_val])
ax.set_title('Tenure Years by Churn', fontweight='bold')
ax.set_xlabel('Tenure (years)')
ax.legend()

ax = axes[0, 2]
churn_by_complaint = df.groupby('complaint_count')['churn'].mean() * 100
ax.bar(churn_by_complaint.index, churn_by_complaint.values, color='#E05C2A', edgecolor='white')
ax.set_title('Churn Rate by Complaint Count', fontweight='bold')
ax.set_xlabel('Number of Complaints')
ax.set_ylabel('Churn Rate (%)')

ax = axes[1, 0]
churn_by_policy = df.groupby('policy_type')['churn'].mean() * 100
ax.bar(churn_by_policy.index, churn_by_policy.values,
       color=['#2E6DA4', '#4A9DBF', '#E05C2A', '#F0A060'], edgecolor='white')
ax.set_title('Churn Rate by Policy Type', fontweight='bold')
ax.set_xlabel('Policy Type')
ax.set_ylabel('Churn Rate (%)')

ax = axes[1, 1]
churn_by_sat = df.groupby('satisfaction_score')['churn'].mean() * 100
ax.bar(churn_by_sat.index.astype(str), churn_by_sat.values, color='#E05C2A', edgecolor='white')
ax.set_title('Churn Rate by Satisfaction Score', fontweight='bold')
ax.set_xlabel('Satisfaction Score')
ax.set_ylabel('Churn Rate (%)')

ax = axes[1, 2]
for churn_val, group in df.groupby('churn'):
    ax.hist(group['premium_increase_pct'], bins=15, alpha=0.6, label=labels[churn_val], color=palette[churn_val])
ax.set_title('Premium Increase % by Churn', fontweight='bold')
ax.set_xlabel('Premium Increase (%)')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/01_eda.png")

# ── 3. Feature Engineering ───────────────────────────────────
print("\n[3] Feature engineering...")

df_model = df.copy()
le = LabelEncoder()
df_model['policy_type_enc'] = le.fit_transform(df_model['policy_type'])
df_model['region_enc']      = le.fit_transform(df_model['region'])
df_model['claim_frequency']    = df_model['num_claims'] / (df_model['tenure_years'] + 1)
df_model['premium_per_policy'] = df_model['annual_premium'] / df_model['num_policies']
df_model['high_risk']          = ((df_model['complaint_count'] > 0) & (df_model['num_claims'] > 1)).astype(int)

print("  New features created:")
print("  - claim_frequency    : claims per year of tenure")
print("  - premium_per_policy : premium divided by number of policies")
print("  - high_risk          : flag for customers with complaints AND multiple claims")

feature_cols = [
    'age', 'tenure_years', 'num_policies', 'annual_premium',
    'num_claims', 'claim_amount', 'complaint_count', 'contacted_support',
    'premium_increase_pct', 'satisfaction_score', 'years_since_last_claim',
    'digital_engagement', 'policy_type_enc', 'region_enc',
    'claim_frequency', 'premium_per_policy', 'high_risk'
]

X = df_model[feature_cols]
y = df_model['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Training set: {X_train.shape[0]} samples")
print(f"  Test set:     {X_test.shape[0]} samples")

# ── 4. PCA ───────────────────────────────────────────────────
print("\n[4] Applying PCA for dimensionality reduction...")

pca_full = PCA()
pca_full.fit(X_train_scaled)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(explained_variance >= 0.95) + 1
print(f"  Components needed to explain 95% variance: {n_components_95} (out of {X_train_scaled.shape[1]})")

pca = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_train_2d = pca_2d.fit_transform(X_train_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PCA – Dimensionality Reduction', fontsize=14, fontweight='bold')

ax = axes[0]
components = np.arange(1, len(explained_variance) + 1)
ax.plot(components, explained_variance * 100, 'o-', color='#2E6DA4', lw=2, markersize=5)
ax.axhline(y=95, color='#E05C2A', linestyle='--', lw=1.5, label='95% threshold')
ax.axvline(x=n_components_95, color='#2EAA6D', linestyle='--', lw=1.5,
           label=f'{n_components_95} components selected')
ax.fill_between(components, explained_variance * 100, alpha=0.1, color='#2E6DA4')
ax.set_title('Cumulative Explained Variance', fontweight='bold')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance (%)')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
colors_pca = {0: '#2E6DA4', 1: '#E05C2A'}
for churn_val in [0, 1]:
    mask = y_train == churn_val
    ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
               c=colors_pca[churn_val], label=labels[churn_val],
               alpha=0.4, s=20, edgecolors='none')
var1 = pca_2d.explained_variance_ratio_[0] * 100
var2 = pca_2d.explained_variance_ratio_[1] * 100
ax.set_title('PCA – 2D Visualization of Customer Segments', fontweight='bold')
ax.set_xlabel(f'PC1 ({var1:.1f}% variance)')
ax.set_ylabel(f'PC2 ({var2:.1f}% variance)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/02_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/02_pca.png")

# ── 5. Train Models ──────────────────────────────────────────
print("\n[5] Training models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    X_tr = X_train_pca if name == 'Logistic Regression' else X_train
    X_te = X_test_pca  if name == 'Logistic Regression' else X_test

    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(model,
                          X_train_pca if name == 'Logistic Regression' else X_train,
                          y_train, cv=5, scoring='roc_auc').mean()

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': acc, 'auc': auc, 'cv_auc': cv
    }
    print(f"  {name:25s}  Accuracy: {acc:.1%}  ROC-AUC: {auc:.3f}  CV-AUC: {cv:.3f}")

# ── 6. Model Comparison ──────────────────────────────────────
print("\n[6] Generating model comparison plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')

colors_roc = ['#2E6DA4', '#E05C2A', '#2EAA6D']
ax = axes[0]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=col, lw=2)
ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
ax.set_title('ROC Curves', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

names   = list(results.keys())
aucs    = [results[n]['auc']    for n in names]
cv_aucs = [results[n]['cv_auc'] for n in names]
x = np.arange(len(names))
w = 0.35
ax = axes[1]
bars1 = ax.bar(x - w/2, aucs,    w, label='Test AUC', color='#2E6DA4', edgecolor='white')
bars2 = ax.bar(x + w/2, cv_aucs, w, label='CV AUC',   color='#E05C2A', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_title('Test AUC vs Cross-Validation AUC', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)

best_name = max(results, key=lambda n: results[n]['accuracy'])
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
ConfusionMatrixDisplay(cm, display_labels=['Retained', 'Churned']).plot(
    ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title(f'Confusion Matrix\n{best_name} (Best Model)', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/03_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/03_model_comparison.png")

# ── 7. Feature Importance ────────────────────────────────────
print("\n[7] Plotting feature importance (Random Forest)...")

rf_model    = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors_fi = ['#E05C2A' if imp > importances.quantile(0.75) else '#2E6DA4' for imp in importances.values]
ax.barh(importances.index, importances.values, color=colors_fi, edgecolor='white')
ax.set_title('Feature Importance – Random Forest\n(Orange = Top 25% most important features)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/04_feature_importance.png")

# ── 8. Business Insights ─────────────────────────────────────
print("\n[8] Generating business insights...")

df_model['churn_prob']   = results['Gradient Boosting']['model'].predict_proba(df_model[feature_cols])[:, 1]
df_model['risk_segment'] = pd.cut(df_model['churn_prob'],
                                   bins=[0, 0.1, 0.25, 0.5, 1.0],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Business Insights: Customer Risk Segmentation', fontsize=14, fontweight='bold')

seg_colors = ['#2EAA6D', '#F0C040', '#E05C2A', '#8B0000']
seg_counts  = df_model['risk_segment'].value_counts()

axes[0].pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
            colors=seg_colors, startangle=90, pctdistance=0.75)
axes[0].set_title('Customer Risk Segments', fontweight='bold')

seg_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical']
premium_at_risk = df_model.groupby('risk_segment')['annual_premium'].sum() / 1_000
vals = [premium_at_risk.get(s, 0) for s in seg_order]
bars = axes[1].bar(seg_order, vals, color=seg_colors, edgecolor='white')
for bar, val in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'€{val:.0f}K', ha='center', fontweight='bold', fontsize=9)
axes[1].set_title('Total Annual Premium at Risk by Segment', fontweight='bold')
axes[1].set_ylabel('Total Premium (thousands €)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_business_insights.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/05_business_insights.png")

# ── 9. Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL SUMMARY")
print("=" * 60)
print(f"\n  PCA: reduced {len(feature_cols)} features → {n_components_95} components (95% variance retained)")
for name, res in results.items():
    print(f"\n  {name}")
    print(f"    Accuracy : {res['accuracy']:.1%}")
    print(f"    ROC-AUC  : {res['auc']:.3f}")
    print(f"    CV-AUC   : {res['cv_auc']:.3f}")

best_name = max(results, key=lambda n: results[n]['accuracy'])
print(f"\n  Best model: {best_name} (Accuracy = {results[best_name]['accuracy']:.1%})")
top_features = importances.tail(5).index.tolist()[::-1]
print(f"\n  Top 5 churn drivers:")
for i, f in enumerate(top_features, 1):
    print(f"    {i}. {f}")
print("\n  All outputs saved to: outputs/")
print("=" * 60)
