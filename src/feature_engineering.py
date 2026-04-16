import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    features = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
                'Network_Latency_ms', 'Packet_Loss_%', 'Error_Rate_%',
                'Quality_Control_Defect_Rate_%', 'Predictive_Maintenance_Score']
    
    for feature in features:
        if feature in df.columns:
            df[f'{feature}_Rolling_Mean'] = df[feature].rolling(window=10, min_periods=1).mean()
            df[f'{feature}_Rolling_Std'] = df[feature].rolling(window=10, min_periods=1).std()
    
    if 'Vibration_Hz' in df.columns and 'Power_Consumption_kW' in df.columns:
        df['Vibration_Power_Ratio'] = df['Vibration_Hz'] / (df['Power_Consumption_kW'] + 0.01)
    
    if 'Error_Rate_%' in df.columns:
        df['Error_Change_Rate'] = df['Error_Rate_%'].diff().fillna(0)
    
    return df

def get_feature_columns():
    return ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
            'Network_Latency_ms', 'Packet_Loss_%', 'Error_Rate_%',
            'Quality_Control_Defect_Rate_%', 'Predictive_Maintenance_Score',
            'Temperature_C_Rolling_Mean', 'Vibration_Hz_Rolling_Mean',
            'Power_Consumption_kW_Rolling_Mean', 'Error_Rate_%_Rolling_Mean',
            'Vibration_Power_Ratio', 'Error_Change_Rate']
