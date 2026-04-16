import pandas as pd
import numpy as np

def load_data(file_path='data/Thales_Group_Manufacturing.csv'):
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    return df

def get_machine_list(df):
    return sorted(df['Machine_ID'].unique())

def filter_by_machine(df, machine_id):
    if machine_id == 'All':
        return df
    return df[df['Machine_ID'] == machine_id].copy()

def filter_by_mode(df, operation_mode):
    if operation_mode == 'All':
        return df
    return df[df['Operation_Mode'] == operation_mode].copy()

def filter_by_date_range(df, start_date, end_date):
    mask = (df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)
    return df[mask].copy()
