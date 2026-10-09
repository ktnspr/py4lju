import numpy as np
import pandas as pd


def generate_vehicle_fraud_data(n=10000, seed=42):
    """Generate synthetic motor insurance claims with a binary `fraud` label.

    A latent fraud propensity drives several observable features, so the
    classifiers have genuine signal to learn from. `occupation`, `region`,
    `vehicle_type`, `coverage_type` and `vehicle_value` carry no signal on
    their own; `vehicle_value` matters only through the claim-to-value ratio.

    The result looks like a raw export: a `claim_id`, 50 duplicate rows, the
    placeholder 0 for 30 unknown ages and missing `schufa_score` values. A
    missing `schufa_score` is about four times as likely for fraudulent claims.
    """
    rng = np.random.default_rng(seed)

    # Stage 1: latent fraud propensity
    propensity = rng.normal(0, 1, n)

    # Stage 2: features correlated with the propensity
    policyholder_age = np.clip(50 - 5 * propensity + rng.normal(0, 8, n), 18, 80).astype(int)
    schufa_score = np.clip(600 - 60 * propensity + rng.normal(0, 40, n), 300, 850).astype(int)
    customer_since_years = np.clip(15 - 4 * propensity + rng.normal(0, 3, n), 0, 30).astype(int)
    number_of_prior_claims = np.clip(2 + propensity + rng.normal(0, 1.2, n), 0, 5).astype(int)
    reporting_delay_days = np.clip(25 + 10 * propensity + rng.normal(0, 5, n), 0, 60).astype(int)
    police_report = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(0.8 * propensity))).astype(int)
    witnesses_present = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(0.6 * propensity))).astype(int)

    # Features without fraud signal
    occupation = rng.choice(['Employed', 'Self-employed', 'Unemployed', 'Student'], n)
    region = rng.choice([f'PLZ_{i:02d}' for i in range(1, 11)], n)
    vehicle_value = rng.uniform(1000, 80000, n)
    vehicle_type = rng.choice(['Compact', 'Mid-range', 'Luxury', 'SUV', 'Sports car'], n)
    coverage_type = rng.choice(['Liability', 'Partial coverage', 'Comprehensive'], n)

    # Fraudsters inflate their claims
    claim_amount = (rng.gamma(shape=2, scale=5000, size=n) + 100) * np.exp(0.6 * propensity)

    # Fraud label from a logistic model
    no_report = police_report == 0
    no_witness = witnesses_present == 0
    log_odds = (
        -8.5
        + 4.0 * claim_amount / vehicle_value
        + 2.5 * (850 - schufa_score) / 550
        + 1.5 * reporting_delay_days / 60
        + np.where(no_report & no_witness, 1.5, np.where(no_report | no_witness, 0.3, 0.0))
        + 0.5 * (customer_since_years < 3)
        + 0.4 * (number_of_prior_claims >= 3)
        + 0.3 * (policyholder_age < 30)
    )
    fraud = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-log_odds))).astype(int)

    df = pd.DataFrame({
        'claim_id': np.arange(100001, 100001 + n),
        'policyholder_age': policyholder_age,
        'occupation': occupation,
        'region': region,
        'schufa_score': schufa_score,
        'customer_since_years': customer_since_years,
        'vehicle_value': vehicle_value.round(2),
        'vehicle_type': vehicle_type,
        'coverage_type': coverage_type,
        'number_of_prior_claims': number_of_prior_claims,
        'claim_amount': claim_amount.round(2),
        'police_report': police_report,
        'witnesses_present': witnesses_present,
        'reporting_delay_days': reporting_delay_days,
        'fraud': fraud,
    })

    # Raw export: placeholder 0 for unknown ages
    df.loc[rng.choice(n, 30, replace=False), 'policyholder_age'] = 0

    # Raw export: missing credit scores, more frequent for fraudulent claims
    schufa_missing = rng.uniform(0, 1, n) < np.where(fraud == 1, 0.15, 0.04)
    df['schufa_score'] = df['schufa_score'].astype('Int64').mask(schufa_missing)

    # Raw export: duplicate rows from a double export, in random order
    duplicates = df.loc[rng.choice(n, 50, replace=False)]
    return pd.concat([df, duplicates]).sample(frac=1, random_state=seed).reset_index(drop=True)
