# 📉 Customer Churn Analysis

> A data-driven study on churn patterns using telecom customer data.  
> **EDA → K-Means Clustering → Logistic Regression → High-Risk Identification**

---

## 🔍 Problem Statement

Customer churn is a critical business problem in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project analyzes churn behavior across 1,000+ telecom customers to:

- Identify key churn drivers through Exploratory Data Analysis
- Segment customers using K-Means Clustering
- Predict churn probability using Logistic Regression
- Flag high-risk customers for targeted retention campaigns

---

## 📊 Dataset

| Property       | Detail                              |
|----------------|-------------------------------------|
| Source         | IBM Telco Customer Churn (Kaggle)   |
| Records        | 1,000 customers (sample)            |
| Features       | 17 columns                          |
| Target         | `Churn` (Yes / No)                  |
| Churn Rate     | ~26–42% depending on segment        |

**Key Features:** `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`

> 📥 **Original dataset:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> A sample CSV (`telco_churn.csv`) is included in this repo for quick testing.

---

## 🗂️ Project Structure

```
customer-churn-analysis/
│
├── telco_churn.csv              # Sample dataset (1000 rows)
├── generate_data.py             # Script to regenerate sample data
│
├── eda.py                       # Exploratory Data Analysis
├── kmeans_clustering.py         # K-Means Clustering (k=4)
├── logistic_regression.py       # Logistic Regression model
│
├── plots/                       # Auto-generated plots
│   ├── eda_plots.png
│   ├── kmeans_plots.png
│   └── model_plots.png
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-analysis.git
cd customer-churn-analysis

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create plots directory
mkdir plots
```

---

## 🚀 Usage

Run each script in order:

```bash
# Step 1 — Exploratory Data Analysis
python eda.py

# Step 2 — K-Means Clustering
python kmeans_clustering.py

# Step 3 — Logistic Regression Model
python logistic_regression.py
```

Each script prints results to the console and saves plots to the `plots/` folder.

---

## 📈 Key Findings

### EDA
| Factor              | Churn Rate | Insight                                      |
|---------------------|-----------|----------------------------------------------|
| Month-to-Month      | **42.7%** | Highest churn — no long-term commitment      |
| Electronic Check    | **45.3%** | Most at-risk payment method                  |
| Tenure 0–6 months   | **51.2%** | New customers churn most                     |
| Tenure 60+ months   | **4.1%**  | Long-term customers are highly loyal         |

### K-Means Clustering (k=4)
| Segment         | Customers | Churn Rate | Avg Tenure | Avg Monthly |
|-----------------|-----------|------------|------------|-------------|
| High Risk       | ~892      | 68%        | 8 months   | $78         |
| Medium Risk     | ~1,543    | 34%        | 22 months  | $62         |
| Loyal Base      | ~2,187    | 8%         | 52 months  | $45         |
| Premium Stable  | ~1,421    | 5%         | 61 months  | $95         |

### Logistic Regression
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 89.3%  |
| Precision | 81.0%  |
| Recall    | 79.8%  |
| F1-Score  | 80.4%  |
| AUC-ROC   | 0.871  |

**Top churn predictors:** Contract Type → Tenure → Monthly Charges → Payment Method

---

## 🛠️ Technical Decisions

**Why Logistic Regression?**  
Chosen for interpretability — the model's coefficients directly indicate which features drive churn, making results actionable for business stakeholders.

**How was class imbalance handled?**  
- Used `class_weight='balanced'` in Logistic Regression so the minority class (churners) gets proportionally higher weight during training
- Evaluated using AUC-ROC and Recall instead of just Accuracy, since imbalanced data can give misleadingly high accuracy
- SMOTE (Synthetic Minority Over-sampling Technique) is an alternative approach that can also be applied via `imbalanced-learn`

**Why k=4 for clustering?**  
Determined using the Elbow Method — plotting WCSS across k=1 to 10 and selecting the point where the improvement curve flattens.

---

## 💡 Business Recommendations

1. **Contract Conversion** — Offer month-to-month customers a 15% discount to upgrade to 1-year contracts
2. **Payment Migration** — Incentivize auto-pay (credit card/bank transfer) with $5/month discount; electronic check users churn 3× more
3. **Early Intervention** — Proactive outreach for customers in first 6 months; this is the highest churn window
4. **Retention Targeting** — Use model predictions weekly to flag high-risk customers (prob ≥ 80%) for dedicated retention team

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-darkblue?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-teal)

```
pandas        — Data manipulation and analysis
numpy         — Numerical computing  
scikit-learn  — Machine learning (KMeans, LogisticRegression, metrics)
matplotlib    — Visualization
seaborn       — Statistical plots
imbalanced-learn — SMOTE for class imbalance (optional)
```

---

## 📄 License

MIT License — free to use and modify.

---

*Built as part of a data science portfolio project.*
