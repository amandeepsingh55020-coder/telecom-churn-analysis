# ============================================================
#  Customer Churn Analysis — EDA
#  Dataset : telco_churn.csv
#  Author  : (Your Name)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0A0E1A',
    'axes.facecolor':   '#111827',
    'axes.edgecolor':   '#1F2937',
    'axes.labelcolor':  '#E5E7EB',
    'xtick.color':      '#6B7280',
    'ytick.color':      '#6B7280',
    'text.color':       '#E5E7EB',
    'grid.color':       '#1F2937',
    'grid.linewidth':   0.8,
    'font.family':      'monospace',
})
CHURN_COLOR   = '#FF4560'
RETAIN_COLOR  = '#00E396'
ACCENT_COLOR  = '#FEB019'
BLUE_COLOR    = '#008FFB'

# ── 1. Load Data ─────────────────────────────────────────────
df = pd.read_csv('telco_churn.csv')
print("=" * 55)
print("  CUSTOMER CHURN ANALYSIS — EXPLORATORY DATA ANALYSIS")
print("=" * 55)
print(f"\n📦 Dataset Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📋 Columns         : {list(df.columns)}")
print(f"\n🔍 Missing Values  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n📊 Data Types      :\n{df.dtypes.value_counts()}")

# ── 2. Basic Cleaning ─────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# ── 3. Overall Churn Rate ─────────────────────────────────────
churn_counts = df['Churn'].value_counts()
churn_rate   = df['Churn_Binary'].mean() * 100
print(f"\n📉 Overall Churn Rate : {churn_rate:.1f}%")
print(f"   Churned   : {churn_counts['Yes']:,}")
print(f"   Retained  : {churn_counts['No']:,}")

# ── 4. Churn by Contract Type ─────────────────────────────────
contract_churn = (
    df.groupby('Contract')['Churn_Binary']
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'ChurnRate', 'count': 'Customers'})
    .sort_values('ChurnRate', ascending=False)
)
contract_churn['ChurnRate'] *= 100
print(f"\n📋 Churn Rate by Contract Type:\n{contract_churn.round(1)}")

# ── 5. Churn by Payment Method ────────────────────────────────
payment_churn = (
    df.groupby('PaymentMethod')['Churn_Binary']
    .mean()
    .sort_values(ascending=False) * 100
)
print(f"\n💳 Churn Rate by Payment Method:\n{payment_churn.round(1)}")

# ── 6. Churn by Tenure Bucket ─────────────────────────────────
df['TenureBucket'] = pd.cut(
    df['tenure'],
    bins=[0, 6, 12, 24, 36, 48, 60, 72],
    labels=['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '60+']
)
tenure_churn = df.groupby('TenureBucket')['Churn_Binary'].mean() * 100
print(f"\n📅 Churn Rate by Tenure Bucket (months):\n{tenure_churn.round(1)}")

# ── 7. Correlation of Numeric Features with Churn ─────────────
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr = df[numeric_cols + ['Churn_Binary']].corr()['Churn_Binary'].drop('Churn_Binary')
print(f"\n🔗 Correlation with Churn:\n{corr.round(3)}")

# ============================================================
#  PLOTS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Customer Churn — Exploratory Data Analysis',
             fontsize=16, fontweight='bold', color='#E5E7EB', y=1.01)

# Plot 1 — Pie chart overall churn
ax = axes[0, 0]
ax.pie(
    churn_counts,
    labels=['Retained', 'Churned'],
    colors=[RETAIN_COLOR, CHURN_COLOR],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': '#0A0E1A', 'linewidth': 2},
    textprops={'color': '#E5E7EB', 'fontsize': 11}
)
ax.set_title('Overall Churn Distribution', fontsize=12, pad=12)

# Plot 2 — Churn by Contract
ax = axes[0, 1]
bars = ax.bar(contract_churn.index, contract_churn['ChurnRate'],
              color=[CHURN_COLOR, ACCENT_COLOR, RETAIN_COLOR], edgecolor='#0A0E1A',
              linewidth=1.5, width=0.55)
for bar, val in zip(bars, contract_churn['ChurnRate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#E5E7EB')
ax.set_title('Churn Rate by Contract Type', fontsize=12, pad=12)
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, max(contract_churn['ChurnRate']) * 1.2)
ax.grid(axis='y', alpha=0.4)
ax.tick_params(axis='x', rotation=10)

# Plot 3 — Churn by Payment Method
ax = axes[0, 2]
colors_pm = [CHURN_COLOR, ACCENT_COLOR, BLUE_COLOR, RETAIN_COLOR]
bars = ax.barh(payment_churn.index, payment_churn.values,
               color=colors_pm, edgecolor='#0A0E1A', linewidth=1.5, height=0.55)
for bar, val in zip(bars, payment_churn.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', ha='left', va='center', fontsize=9, color='#E5E7EB')
ax.set_title('Churn Rate by Payment Method', fontsize=12, pad=12)
ax.set_xlabel('Churn Rate (%)')
ax.grid(axis='x', alpha=0.4)

# Plot 4 — Churn by Tenure Bucket
ax = axes[1, 0]
ax.plot(tenure_churn.index, tenure_churn.values,
        color=CHURN_COLOR, linewidth=2.5, marker='o',
        markerfacecolor='#FF4560', markersize=7)
ax.fill_between(range(len(tenure_churn)), tenure_churn.values,
                alpha=0.15, color=CHURN_COLOR)
ax.set_title('Churn Rate by Tenure (months)', fontsize=12, pad=12)
ax.set_ylabel('Churn Rate (%)')
ax.set_xlabel('Tenure Bucket')
ax.set_xticks(range(len(tenure_churn)))
ax.set_xticklabels(tenure_churn.index)
ax.grid(axis='y', alpha=0.4)

# Plot 5 — Monthly Charges distribution
ax = axes[1, 1]
churned     = df[df['Churn'] == 'Yes']['MonthlyCharges']
not_churned = df[df['Churn'] == 'No']['MonthlyCharges']
ax.hist(not_churned, bins=30, color=RETAIN_COLOR, alpha=0.65, label='Retained', edgecolor='#0A0E1A')
ax.hist(churned,     bins=30, color=CHURN_COLOR,  alpha=0.65, label='Churned',  edgecolor='#0A0E1A')
ax.set_title('Monthly Charges Distribution', fontsize=12, pad=12)
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Count')
ax.legend(facecolor='#111827', edgecolor='#1F2937', labelcolor='#E5E7EB')
ax.grid(axis='y', alpha=0.4)

# Plot 6 — Tenure vs Monthly Charges scatter
ax = axes[1, 2]
scatter_no  = df[df['Churn'] == 'No']
scatter_yes = df[df['Churn'] == 'Yes']
ax.scatter(scatter_no['tenure'],  scatter_no['MonthlyCharges'],
           color=RETAIN_COLOR, alpha=0.25, s=15, label='Retained')
ax.scatter(scatter_yes['tenure'], scatter_yes['MonthlyCharges'],
           color=CHURN_COLOR,  alpha=0.45, s=15, label='Churned')
ax.set_title('Tenure vs Monthly Charges', fontsize=12, pad=12)
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Monthly Charges ($)')
ax.legend(facecolor='#111827', edgecolor='#1F2937', labelcolor='#E5E7EB')
ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('plots/eda_plots.png', dpi=150, bbox_inches='tight',
            facecolor='#0A0E1A')
plt.show()
print("\n✅ EDA complete! Plots saved to plots/eda_plots.png")
