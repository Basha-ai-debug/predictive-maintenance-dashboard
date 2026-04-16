import numpy as np
import pandas as pd

def classify_risk(anomaly_scores):
    risk_levels = []
    for score in anomaly_scores:
        if score < 0.3:
            risk_levels.append('Low Risk')
        elif score < 0.6:
            risk_levels.append('Medium Risk')
        else:
            risk_levels.append('High Risk')
    return risk_levels

def get_high_risk_machines(df, threshold=0.6):
    high_risk = df[df['Anomaly_Score'] >= threshold]
    return high_risk['Machine_ID'].value_counts().to_dict()

def get_risk_summary(df):
    risk_counts = df['Risk_Level'].value_counts()
    total = len(df)
    return {
        'low_risk_pct': (risk_counts.get('Low Risk', 0) / total) * 100,
        'medium_risk_pct': (risk_counts.get('Medium Risk', 0) / total) * 100,
        'high_risk_pct': (risk_counts.get('High Risk', 0) / total) * 100,
        'total_records': total,
        'high_risk_machines': len(df[df['Risk_Level'] == 'High Risk']['Machine_ID'].unique())
    }
