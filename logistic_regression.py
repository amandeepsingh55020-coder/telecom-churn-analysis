
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.facecolor': '#0A0E1A', 'axes.facecolor':  '#111827',
    'axes.edgecolor':   '#1F2937', 'axes.labelcolor': '#E5E7EB',
    'xtick.color':      '#6B7280', 'ytick.color':     '#6B7280',
    'text.color':       '#E5E7EB', 'grid.color':      '#1F2937',
    'grid.linewidth':   0.8,       'font.family':     'monospace',
})
CHURN_COLOR  = '#FF4560'
RETAIN_COLOR = '#00E396'
ACCENT_COLOR = '#FEB019'
BLUE_COLOR   = '#008FFB'
PURPLE_COLOR = '#775DD0'

#  1. Load & Preprocess
df = pd.read_csv('telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

print("=" * 55)
print("  CUSTOMER CHURN — LOGISTIC REGRESSION MODEL")
print("=" * 55)
print(f"\n Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

#  2. Encode Categorical Features 
le = LabelEncoder()
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
            'InternetService', 'OnlineSecurity', 'TechSupport',
            'StreamingTV', 'Contract', 'PaperlessBilling', 'PaymentMethod']

df_model = df.copy()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

feature_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'InternetService',
                'OnlineSecurity', 'TechSupport', 'StreamingTV',
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges']

X = df_model[feature_cols]
y = (df['Churn'] == 'Yes').astype(int)

print(f"\n Features used     : {len(feature_cols)}")
print(f"   Class distribution: {y.value_counts().to_dict()}")
print(f"   Churn rate        : {y.mean()*100:.1f}%")

#  3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Train set : {X_train.shape[0]} samples")
print(f"   Test set  : {X_test.shape[0]} samples")

#  4. Scale Features 
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

#  5. Train Model 
# class_weight='balanced' handles imbalanced dataset automatically
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=1.0                   
)
model.fit(X_train_sc, y_train)
print("\n Model trained successfully!")

# 6. Predictions 
y_pred       = model.predict(X_test_sc)
y_pred_proba = model.predict_proba(X_test_sc)[:, 1]

#  7. Evaluation Metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 40)
print("  MODEL PERFORMANCE METRICS")
print("=" * 40)
print(f"  Accuracy  : {accuracy*100:.1f}%")
print(f"  Precision : {precision*100:.1f}%")
print(f"  Recall    : {recall*100:.1f}%")
print(f"  F1-Score  : {f1*100:.1f}%")
print(f"  AUC-ROC   : {auc:.3f}")
print("=" * 40)
print(f"\n Classification Report:\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

# ── 8. Cross-Validation ───────────────────────────────────────
cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='roc_auc')
print(f" 5-Fold Cross-Val AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 9. Feature Importance (Coefficients
importance_df = pd.DataFrame({
    'Feature':    feature_cols,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(f"\n Top 5 Churn Drivers:\n{importance_df.head().to_string(index=False)}")
print(f"\n Top 5 Retention Drivers:\n{importance_df.tail().to_string(index=False)}")

#  10. High-Risk Customer Identification 
X_test_df = X_test.copy()
X_test_df['ChurnProbability'] = y_pred_proba
X_test_df['ActualChurn']      = y_test.values
X_test_df['CustomerID']       = df.iloc[X_test.index]['customerID'].values

high_risk = (
    X_test_df[X_test_df['ChurnProbability'] >= 0.8]
    .sort_values('ChurnProbability', ascending=False)
    [['CustomerID', 'ChurnProbability', 'ActualChurn']]
)
print(f"\n  High-Risk Customers (prob ≥ 80%): {len(high_risk)}")
print(high_risk.head(10).to_string(index=False))
=
#  PLOTS
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Logistic Regression — Model Results',
             fontsize=16, fontweight='bold', color='#E5E7EB')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1 — Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(y_test, y_pred)
labels_cm = [['True\nNegative', 'False\nPositive'],
             ['False\nNegative', 'True\nPositive']]
colors_cm = [[RETAIN_COLOR, ACCENT_COLOR],
             [CHURN_COLOR,  BLUE_COLOR]]
for i in range(2):
    for j in range(2):
        ax1.add_patch(plt.Rectangle((j, 1-i), 1, 1, fill=True,
                      color=colors_cm[i][j], alpha=0.25))
        ax1.text(j+0.5, 1.5-i, str(cm[i][j]), ha='center', va='center',
                 fontsize=20, fontweight='bold', color=colors_cm[i][j])
        ax1.text(j+0.5, 1.15-i, labels_cm[i][j], ha='center', va='center',
                 fontsize=8, color='#6B7280')
ax1.set_xlim(0, 2); ax1.set_ylim(0, 2)
ax1.set_xticks([0.5, 1.5]); ax1.set_xticklabels(['Pred: No', 'Pred: Yes'])
ax1.set_yticks([0.5, 1.5]); ax1.set_yticklabels(['Actual: Yes', 'Actual: No'])
ax1.set_title('Confusion Matrix', fontsize=12, pad=12)

# Plot 2  ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color=BLUE_COLOR, linewidth=2.5,
         label=f'AUC = {auc:.3f}')
ax2.plot([0,1], [0,1], color='#374151', linestyle='--', linewidth=1.5,
         label='Random Classifier')
ax2.fill_between(fpr, tpr, alpha=0.1, color=BLUE_COLOR)
ax2.set_title('ROC Curve', fontsize=12, pad=12)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(facecolor='#111827', edgecolor='#1F2937', labelcolor='#E5E7EB')
ax2.grid(alpha=0.4)

# Plot 3  Metrics Bar
ax3 = fig.add_subplot(gs[0, 2])
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
metric_vals  = [accuracy, precision, recall, f1, auc]
bar_colors   = [RETAIN_COLOR, BLUE_COLOR, ACCENT_COLOR, PURPLE_COLOR, CHURN_COLOR]
bars = ax3.bar(metric_names, metric_vals, color=bar_colors,
               edgecolor='#0A0E1A', linewidth=1.5, width=0.6)
for bar, val in zip(bars, metric_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='#E5E7EB')
ax3.set_title('Model Performance Metrics', fontsize=12, pad=12)
ax3.set_ylim(0, 1.15)
ax3.grid(axis='y', alpha=0.4)
ax3.tick_params(axis='x', rotation=15)

# Plot 4  Feature Importance
ax4 = fig.add_subplot(gs[1, :2])
top_features = importance_df.head(8)
bot_features = importance_df.tail(4)
fi_plot      = pd.concat([top_features, bot_features])
bar_clrs     = [CHURN_COLOR if v > 0 else RETAIN_COLOR for v in fi_plot['Coefficient']]
ax4.barh(fi_plot['Feature'], fi_plot['Coefficient'],
         color=bar_clrs, edgecolor='#0A0E1A', linewidth=1.2, height=0.6)
ax4.axvline(x=0, color='#4B5563', linestyle='--', linewidth=1)
ax4.set_title('Feature Coefficients (+ = increases churn risk)',
              fontsize=12, pad=12)
ax4.set_xlabel('Coefficient Value')
ax4.grid(axis='x', alpha=0.4)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=CHURN_COLOR,  label='Increases churn risk'),
                   Patch(facecolor=RETAIN_COLOR, label='Reduces churn risk')]
ax4.legend(handles=legend_elements, facecolor='#111827',
           edgecolor='#1F2937', labelcolor='#E5E7EB')

# Plot 5  Churn Probability Distribution
ax5 = fig.add_subplot(gs[1, 2])
retained_proba = y_pred_proba[y_test == 0]
churned_proba  = y_pred_proba[y_test == 1]
ax5.hist(retained_proba, bins=25, color=RETAIN_COLOR, alpha=0.65,
         label='Retained', edgecolor='#0A0E1A')
ax5.hist(churned_proba,  bins=25, color=CHURN_COLOR,  alpha=0.65,
         label='Churned',  edgecolor='#0A0E1A')
ax5.axvline(x=0.5, color='#FEB019', linestyle='--',
            linewidth=1.5, label='Decision Threshold (0.5)')
ax5.set_title('Predicted Churn Probability', fontsize=12, pad=12)
ax5.set_xlabel('Churn Probability')
ax5.set_ylabel('Count')
ax5.legend(facecolor='#111827', edgecolor='#1F2937',
           labelcolor='#E5E7EB', fontsize=8)
ax5.grid(axis='y', alpha=0.4)

plt.savefig('plots/model_plots.png', dpi=150, bbox_inches='tight',
            facecolor='#0A0E1A')
plt.show()
print("\n✅ Model complete! Plots saved to plots/model_plots.png")

# Save high-risk predictions
high_risk.to_csv('high_risk_customers.csv', index=False)
print("  High-risk customers saved: high_risk_customers.csv")
