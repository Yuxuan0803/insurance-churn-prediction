# 🏥 Insurance Customer Churn Prediction

A complete end-to-end machine learning project to predict customer churn in the insurance industry using Python and scikit-learn.

---

## 📌 Project Overview

Customer churn — when a policyholder cancels or fails to renew their insurance — is a major business challenge. Retaining existing customers is significantly cheaper than acquiring new ones. This project builds and compares multiple machine learning models to identify customers at high risk of churning, enabling insurers to take proactive retention action.

---

## 🎯 Business Objective

> **Predict which insurance customers are likely to churn, and identify the key drivers of churn to inform retention strategies.**

---

## 📁 Project Structure

```
insurance-churn-prediction/
│
├── insurance_churn.csv               # Dataset (2,000 customers)
├── generate_data.py                  # Script to regenerate the dataset
├── insurance_churn_prediction.py     # Main ML pipeline script
├── insurance_churn_notebook.ipynb    # Jupyter Notebook (interactive version)
│
├── outputs/
│   ├── 01_eda.png                    # Exploratory Data Analysis charts
│   ├── 02_model_comparison.png       # ROC curves & model performance
│   ├── 03_feature_importance.png     # Random Forest feature importances
│   └── 04_business_insights.png      # Customer risk segmentation
│
└── README.md
```

---

## 📊 Dataset

The dataset contains **2,000 insurance customers** with the following features:

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
| `churn` | Target variable — 1 = churned, 0 = retained |

**Churn rate: ~10.9%** — realistic for the insurance industry.

---

## 🔧 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution of churn vs retention
- Churn rate by complaint count, policy type, and premium increase
- Feature correlation heatmap

### 2. Feature Engineering
Three new features were created to improve model performance:
- `claim_frequency` = claims per year of tenure
- `premium_per_policy` = annual premium / number of policies
- `high_risk` = binary flag for customers with both complaints and multiple claims

### 3. Models Trained
| Model | Test AUC | CV AUC |
|---|---|---|
| Logistic Regression | 0.765 | 0.726 |
| Random Forest | 0.687 | 0.647 |
| Gradient Boosting | 0.693 | 0.672 |

> ✅ **Best model: Logistic Regression** (AUC = 0.765)

### 4. Business Risk Segmentation
Customers were segmented into four risk tiers based on predicted churn probability:
- 🟢 Low Risk (< 10%)
- 🟡 Medium Risk (10–25%)
- 🟠 High Risk (25–50%)
- 🔴 Critical (> 50%)

---

## 📈 Key Findings

**Top churn drivers:**
1. `premium_increase_pct` — customers are most sensitive to price increases
2. `annual_premium` — higher-premium customers churn more
3. `premium_per_policy` — cost per policy matters
4. `age` — older customers show different churn behaviour
5. `tenure_years` — longer-tenure customers are more loyal

**Business recommendation:**
- Prioritise outreach to **High Risk** and **Critical** segments
- Review and justify premium increases before renewal
- Resolve complaints quickly — even 1 complaint significantly raises churn risk

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

### Or open the notebook
```bash
jupyter notebook insurance_churn_notebook.ipynb
```

---

## 🧠 Skills Demonstrated

- **Python**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Machine Learning**: Logistic Regression, Random Forest, Gradient Boosting
- **Data Science workflow**: EDA → Feature Engineering → Modelling → Evaluation → Business Insights
- **Model evaluation**: ROC-AUC, Cross-validation, Confusion Matrix
- **Business thinking**: Translating model outputs into actionable recommendations

---

## 👤 Author

**Yuxuan Li**  
Biostatistician - SGS  
MSc Data Science – Leiden University   
📧 liyuxuan0803@hotmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yuxuan-li-/)
