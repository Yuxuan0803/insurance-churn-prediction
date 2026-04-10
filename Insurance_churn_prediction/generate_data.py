import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(22, 70, n)
tenure_years = np.random.randint(1, 20, n)
num_policies = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1])
annual_premium = np.random.randint(500, 5000, n)
num_claims = np.random.poisson(0.8, n)
claim_amount = num_claims * np.random.randint(200, 3000, n)
policy_type = np.random.choice(['Life', 'Health', 'Auto', 'Home'], n, p=[0.3, 0.3, 0.25, 0.15])
region = np.random.choice(['North', 'South', 'East', 'West'], n)
complaint_count = np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05])
contacted_support = np.random.choice([0, 1], n, p=[0.6, 0.4])
premium_increase_pct = np.random.uniform(0, 30, n)

# Churn logic: higher churn if high claims, complaints, premium increase, short tenure
churn_prob = (
    0.05
    + 0.003 * num_claims
    + 0.08 * complaint_count
    + 0.005 * premium_increase_pct
    - 0.01 * tenure_years
    + 0.03 * contacted_support
    - 0.002 * num_policies
)
churn_prob = np.clip(churn_prob, 0.03, 0.85)
churn = np.random.binomial(1, churn_prob, n)

df = pd.DataFrame({
    'age': age,
    'tenure_years': tenure_years,
    'num_policies': num_policies,
    'annual_premium': annual_premium,
    'num_claims': num_claims,
    'claim_amount': claim_amount,
    'policy_type': policy_type,
    'region': region,
    'complaint_count': complaint_count,
    'contacted_support': contacted_support,
    'premium_increase_pct': premium_increase_pct.round(2),
    'churn': churn
})

df.to_csv('insurance_churn.csv', index=False)
print(f"Dataset created: {len(df)} rows, churn rate: {df['churn'].mean():.1%}")
