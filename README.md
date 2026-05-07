# 🏥 Insurance Customer Churn Prediction

A complete end-to-end machine learning project to predict customer churn in the insurance industry using Python and scikit-learn. Includes PCA for dimensionality reduction and a full business risk segmentation.

---

## 📌 Project Overview

Customer churn — when a policyholder cancels or fails to renew their insurance — is a major business challenge. Retaining existing customers is significantly cheaper than acquiring new ones. This project builds and compares multiple machine learning models to identify customers at high risk of churning, enabling insurers to take proactive retention action.

---

## 🎯 Business Objective

> **Predict which insurance customers are likely to churn, and identify the key drivers of churn to inform targeted retention strategies.**

---

## 📁 Project Structure

```
insurance-churn-prediction/
│
├── insurance_churn.csv               # Simulated dataset (3,000 customers)
├── insurance_churn_prediction.py     # Main ML pipeline script
│
├── outputs/
│   ├── 01_eda.png                    # Exploratory Data Analysis charts
│   ├── 02_pca.png                    # PCA scree plot & 2D visualization
│   ├── 03_model_comparison.png       # ROC curves & model performance
│   ├── 04_feature_importance.png     # Random Forest feature importances
│   └── 05_business_insights.png      # Customer risk segmentation
│
└── README.md
```

---

## 📊 Dataset

Simulated dataset of **3,000 insurance customers** with a realistic churn rate of **20.7%**.

| Feature | Description |
|---|---|
| `age` | Customer age |
| `tenure_years` | Years as a customer |
| `num_policies` | Number of active policies |
| `annual_premium` | Annual premium paid (€) |
| `num_claims` | Number of claims filed |
| `claim_amount` | Total claim amount (€) |
| `policy_type` | Life / Health / Auto / Home |
| `region` | North / South / East / West |
| `complaint_count` | Number of complaints filed |
| `contacted_support` | Whether customer contacted support (0/1) |
| `premium_increase_pct` | % increase in premium at last renewal |
| `satisfaction_score` | Customer satisfaction rating (1–5) |
| `years_since_last_claim` | Years since most recent claim |
| `digital_engagement` | Whether customer uses digital channels (0/1) |
| `churn` | Target variable — 1 = churned, 0 = retained |

---

## 🔧 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution of churn vs retention
- Churn rate by complaint count, policy type, and satisfaction score
- Premium increase and tenure distributions by churn status

### 2. Feature Engineering
Three new features created to capture business-relevant patterns:

| New Feature | Description |
|---|---|
| `claim_frequency` | Claims per year of tenure |
| `premium_per_policy` | Annual premium divided by number of policies |
| `high_risk` | Binary flag for customers with complaints AND multiple claims |

### 3. PCA – Dimensionality Reduction
Principal Component Analysis (PCA) applied before modelling:
- All features standardised using `StandardScaler`
- **14 out of 17 components** retain 95% of the total variance
- 2D PCA scatter plot visualises separation between churned and retained customers
- PCA-reduced features used to train the Logistic Regression model

### 4. Models Trained & Results

| Model | Accuracy | ROC-AUC | CV-AUC |
|---|---|---|---|
| **Logistic Regression (with PCA)** | **96.7%** | **0.994** | **0.992** |
| Gradient Boosting | 96.0% | 0.991 | 0.985 |
| Random Forest | 93.5% | 0.981 | 0.973 |

> ✅ **Best model: Logistic Regression with PCA** (~95% prediction accuracy)

### 5. Business Risk Segmentation
Customers segmented into four risk tiers based on predicted churn probability:

| Segment | Churn Probability | Recommended Action |
|---|---|---|
| 🟢 Low Risk | < 10% | Standard service |
| 🟡 Medium Risk | 10–25% | Proactive engagement |
| 🟠 High Risk | 25–50% | Personal outreach, loyalty offer |
| 🔴 Critical | > 50% | Immediate intervention |

---

## 📈 Key Findings

**Top 5 churn drivers:**
1. `complaint_count` — even one complaint dramatically raises churn risk
2. `premium_increase_pct` — customers are most sensitive to price increases at renewal
3. `tenure_years` — newer customers are at significantly higher risk
4. `satisfaction_score` — low satisfaction is a strong early warning signal
5. `claim_frequency` — frequent claimers who feel underserved are more likely to leave

**Business recommendations:**
- Resolve complaints **immediately** — complaint count is the single strongest churn predictor
- Be transparent about premium increases and offer justification before renewal
- Invest in onboarding for **new customers** in their first 1–3 years
- Monitor satisfaction scores regularly and act on low scores proactively

---

## 🛠️ How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the pipeline
```bash
python insurance_churn_prediction.py
```

---

## 🧠 Skills Demonstrated

- **Python**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Machine Learning**: Logistic Regression, Random Forest, Gradient Boosting
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **Data Science workflow**: EDA → Feature Engineering → PCA → Modelling → Evaluation → Business Insights
- **Model evaluation**: ROC-AUC, Cross-validation, Confusion Matrix
- **Business thinking**: Translating model outputs into actionable retention strategies

---

## 👤 Author

**Yuxuan Li**  
MSc Data Science – Leiden University  
BSc Insurance: Actuarial & Risk Management  
📧 liyuxuan0803@hotmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yuxuan-li-/)
