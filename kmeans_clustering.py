# ============================================================
#  Customer Churn Analysis — K-Means Clustering
#  Dataset : telco_churn.csv
#  Author  : (Your Name)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0A0E1A', 'axes.facecolor':  '#111827',
    'axes.edgecolor':   '#1F2937', 'axes.labelcolor': '#E5E7EB',
    'xtick.color':      '#6B7280', 'ytick.color':     '#6B7280',
    'text.color':       '#E5E7EB', 'grid.color':      '#1F2937',
    'grid.linewidth':   0.8,       'font.family':     'monospace',
})
CLUSTER_COLORS = ['#FF4560', '#FEB019', '#00E396', '#008FFB']

# ── 1. Load & Prepare Data ────────────────────────────────────
df = pd.read_csv('telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

print("=" * 55)
print("  CUSTOMER CHURN ANALYSIS — K-MEANS CLUSTERING")
print("=" * 55)

# ── 2. Feature Engineering for Clustering ─────────────────────
# Encode categorical features
le = LabelEncoder()
df['Contract_enc']    = le.fit_transform(df['Contract'])       # 0,1,2
df['PaymentMethod_enc']= le.fit_transform(df['PaymentMethod']) # 0,1,2,3
df['InternetService_enc'] = le.fit_transform(df['InternetService'])

# Count number of add-on services
service_cols = ['OnlineSecurity', 'TechSupport', 'StreamingTV']
df['NumServices'] = (df[service_cols] == 'Yes').sum(axis=1)

# Features used for clustering
features = ['tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract_enc', 'PaymentMethod_enc', 'NumServices']
X = df[features].copy()

# ── 3. Scale Features ─────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n✅ Features scaled: {features}")

# ── 4. Elbow Method — Find Optimal k ──────────────────────────
print("\n🔍 Running Elbow Method (k = 1 to 10)...")
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    print(f"   k={k:2d}  WCSS={km.inertia_:,.0f}")

# ── 5. Fit K-Means with k=4 ───────────────────────────────────
optimal_k = 4
print(f"\n✅ Optimal k selected: {optimal_k} (Elbow method)")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ── 6. Cluster Analysis ───────────────────────────────────────
cluster_summary = df.groupby('Cluster').agg(
    Customers      = ('Cluster', 'count'),
    ChurnRate      = ('Churn_Binary', 'mean'),
    AvgTenure      = ('tenure', 'mean'),
    AvgMonthly     = ('MonthlyCharges', 'mean'),
    AvgServices    = ('NumServices', 'mean'),
).round(2)
cluster_summary['ChurnRate'] = (cluster_summary['ChurnRate'] * 100).round(1)

# Assign meaningful labels based on churn rate
labels_map = {
    cluster_summary['ChurnRate'].idxmax(): 'High Risk',
    cluster_summary['ChurnRate'].nlargest(2).index[-1]: 'Medium Risk',
    cluster_summary['ChurnRate'].nsmallest(2).index[-1]: 'Loyal Base',
    cluster_summary['ChurnRate'].idxmin(): 'Premium Stable',
}
cluster_summary['Label'] = cluster_summary.index.map(labels_map)
df['ClusterLabel'] = df['Cluster'].map(labels_map)

print(f"\n📊 Cluster Summary:\n{cluster_summary.to_string()}")

# ── 7. PCA for 2D Visualization ───────────────────────────────
pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
print(f"\n📐 PCA Variance Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ============================================================
#  PLOTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('K-Means Customer Segmentation (k=4)',
             fontsize=16, fontweight='bold', color='#E5E7EB')

# Plot 1 — Elbow Curve
ax = axes[0, 0]
ax.plot(K_range, wcss, color='#008FFB', linewidth=2.5,
        marker='o', markerfacecolor='#FEB019', markersize=7)
ax.axvline(x=optimal_k, color='#FF4560', linestyle='--',
           linewidth=1.5, label=f'Optimal k={optimal_k}')
ax.set_title('Elbow Method — Optimal k', fontsize=12, pad=12)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('WCSS (Inertia)')
ax.legend(facecolor='#111827', edgecolor='#1F2937', labelcolor='#E5E7EB')
ax.grid(alpha=0.4)

# Plot 2 — PCA Scatter
ax = axes[0, 1]
for i, (cluster_id, label) in enumerate(labels_map.items()):
    mask = df['Cluster'] == cluster_id
    ax.scatter(df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'],
               color=CLUSTER_COLORS[i], alpha=0.45, s=18, label=label)
# Plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
for i, (cluster_id, label) in enumerate(labels_map.items()):
    ax.scatter(centroids_pca[cluster_id, 0], centroids_pca[cluster_id, 1],
               color=CLUSTER_COLORS[i], s=200, marker='*',
               edgecolors='white', linewidths=0.8, zorder=5)
ax.set_title('Customer Segments (PCA 2D)', fontsize=12, pad=12)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend(facecolor='#111827', edgecolor='#1F2937', labelcolor='#E5E7EB',
          markerscale=1.5, fontsize=9)
ax.grid(alpha=0.3)

# Plot 3 — Cluster Churn Rate
ax = axes[1, 0]
labels = [labels_map.get(i, f'Cluster {i}') for i in range(optimal_k)]
rates  = [cluster_summary.loc[i, 'ChurnRate'] for i in range(optimal_k)]
bars   = ax.bar(labels, rates, color=CLUSTER_COLORS,
                edgecolor='#0A0E1A', linewidth=1.5, width=0.55)
for bar, val in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color='#E5E7EB')
ax.set_title('Churn Rate per Cluster', fontsize=12, pad=12)
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, max(rates) * 1.25)
ax.grid(axis='y', alpha=0.4)
ax.tick_params(axis='x', rotation=10)

# Plot 4 — Avg Tenure vs Avg Monthly Charges (bubble)
ax = axes[1, 1]
for i, (cluster_id, label) in enumerate(labels_map.items()):
    row  = cluster_summary.loc[cluster_id]
    size = row['Customers'] / 3
    ax.scatter(row['AvgTenure'], row['AvgMonthly'],
               s=size, color=CLUSTER_COLORS[i], alpha=0.8,
               edgecolors='white', linewidths=1, zorder=5)
    ax.annotate(label,
                (row['AvgTenure'], row['AvgMonthly']),
                textcoords='offset points', xytext=(8, 4),
                fontsize=9, color=CLUSTER_COLORS[i])
ax.set_title('Avg Tenure vs Monthly Charges (Bubble = Size)',
             fontsize=11, pad=12)
ax.set_xlabel('Avg Tenure (months)')
ax.set_ylabel('Avg Monthly Charges ($)')
ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('plots/kmeans_plots.png', dpi=150, bbox_inches='tight',
            facecolor='#0A0E1A')
plt.show()
print("\n✅ Clustering complete! Plots saved to plots/kmeans_plots.png")
print(f"   Segment column 'ClusterLabel' added to dataframe.")
df.to_csv('telco_churn_clustered.csv', index=False)
print("   Clustered data saved: telco_churn_clustered.csv")
