import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, get_machine_list, filter_by_machine, filter_by_mode
from src.feature_engineering import create_features
from src.anomaly_detector import AnomalyDetector
from src.risk_classifier import classify_risk, get_high_risk_machines, get_risk_summary

st.set_page_config(page_title='Predictive Maintenance Dashboard', page_icon='??', layout='wide')

st.title('Predictive Maintenance and Anomaly Detection System')
st.markdown('### 6G-Integrated Smart Manufacturing | Thales Group')
st.markdown('---')

@st.cache_data
def load_and_process_data():
    df = load_data('data/Thales_Group_Manufacturing.csv')
    df = create_features(df)
    return df

try:
    df = load_and_process_data()
    st.success(f'Data loaded successfully! {len(df)} records, {df["Machine_ID"].nunique()} machines')
    st.write(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
except Exception as e:
    st.error(f'Error loading data: {e}')
    st.stop()

st.sidebar.header('Controls')

machines = ['All'] + get_machine_list(df)
selected_machine = st.sidebar.selectbox('Select Machine', machines)

modes = ['All'] + sorted(df['Operation_Mode'].unique())
selected_mode = st.sidebar.selectbox('Operation Mode', modes)

risk_threshold = st.sidebar.slider('Risk Threshold', 0.0, 1.0, 0.6, 0.05)

time_window = st.sidebar.selectbox('Time Window', ['Last Day', 'Last 3 Days', 'Last Week', 'Last 2 Weeks', 'All Data'])

filtered_df = filter_by_machine(df, selected_machine)
filtered_df = filter_by_mode(filtered_df, selected_mode)

if time_window == 'Last Day':
    cutoff = df['DateTime'].max() - timedelta(days=1)
    filtered_df = filtered_df[filtered_df['DateTime'] >= cutoff]
elif time_window == 'Last 3 Days':
    cutoff = df['DateTime'].max() - timedelta(days=3)
    filtered_df = filtered_df[filtered_df['DateTime'] >= cutoff]
elif time_window == 'Last Week':
    cutoff = df['DateTime'].max() - timedelta(days=7)
    filtered_df = filtered_df[filtered_df['DateTime'] >= cutoff]
elif time_window == 'Last 2 Weeks':
    cutoff = df['DateTime'].max() - timedelta(days=14)
    filtered_df = filtered_df[filtered_df['DateTime'] >= cutoff]

if len(filtered_df) > 0:
    feature_cols = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
                    'Network_Latency_ms', 'Packet_Loss_%', 'Error_Rate_%']
    X = filtered_df[feature_cols].fillna(filtered_df[feature_cols].mean())
    
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(X)
    predictions, anomaly_scores = detector.predict(X)
    
    filtered_df = filtered_df.copy()
    filtered_df['Anomaly_Score'] = anomaly_scores
    filtered_df['Risk_Level'] = classify_risk(anomaly_scores)
    
    risk_summary = get_risk_summary(filtered_df)
    high_risk_machines = get_high_risk_machines(filtered_df, risk_threshold)
    
    st.markdown('## Key Performance Indicators')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric('Total Records', f'{risk_summary["total_records"]:,}')
    with col2:
        st.metric('High Risk %', f'{risk_summary["high_risk_pct"]:.1f}%')
    with col3:
        st.metric('Medium Risk %', f'{risk_summary["medium_risk_pct"]:.1f}%')
    with col4:
        st.metric('Low Risk %', f'{risk_summary["low_risk_pct"]:.1f}%')
    with col5:
        st.metric('High-Risk Machines', f'{risk_summary["high_risk_machines"]}')
    
    st.markdown('---')
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Risk Overview', 'Machine Anomaly', 'Alerts', 'Historical Trends', 'Sensor Analysis'])
    
    with tab1:
        st.markdown('### Risk Distribution Dashboard')
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = filtered_df['Risk_Level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                         title='Risk Distribution', 
                         color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'],
                         hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_by_mode = filtered_df.groupby(['Operation_Mode', 'Risk_Level']).size().reset_index(name='Count')
            fig = px.bar(risk_by_mode, x='Operation_Mode', y='Count', color='Risk_Level',
                        title='Risk Distribution by Operation Mode',
                        color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            high_risk_df = pd.DataFrame(list(high_risk_machines.items()), columns=['Machine', 'Anomaly Count'])
            if len(high_risk_df) > 0:
                fig = px.bar(high_risk_df.head(10), x='Machine', y='Anomaly Count', 
                            title='Top 10 High-Risk Machines', 
                            color='Anomaly Count', color_continuous_scale='Reds',
                            text='Anomaly Count')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No high-risk machines detected')
        
        with col2:
            fig = px.histogram(filtered_df, x='Anomaly_Score', nbins=30, 
                              title='Anomaly Score Distribution',
                              color_discrete_sequence=['#3498db'])
            fig.add_vline(x=risk_threshold, line_dash='dash', line_color='red',
                         annotation_text=f'Threshold: {risk_threshold}')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if selected_machine != 'All':
            machine_data = filtered_df[filtered_df['Machine_ID'] == selected_machine].copy()
            
            if len(machine_data) > 0:
                st.markdown(f'### Machine {selected_machine} - Detailed Analysis')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_anomaly = machine_data['Anomaly_Score'].mean()
                    st.metric('Avg Anomaly Score', f'{avg_anomaly:.3f}')
                with col2:
                    max_anomaly = machine_data['Anomaly_Score'].max()
                    st.metric('Max Anomaly Score', f'{max_anomaly:.3f}')
                with col3:
                    risk_pct = (machine_data['Risk_Level'] == 'High Risk').mean() * 100
                    st.metric('High Risk %', f'{risk_pct:.1f}%')
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    subplot_titles=('Anomaly Score Trend', 'Error Rate Trend'),
                                    vertical_spacing=0.1)
                
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Anomaly_Score'],
                                          mode='lines', name='Anomaly Score',
                                          line=dict(color='red', width=2)), row=1, col=1)
                fig.add_hline(y=risk_threshold, line_dash='dash', line_color='orange', row=1, col=1)
                
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Error_Rate_%'],
                                          mode='lines', name='Error Rate',
                                          line=dict(color='blue', width=2)), row=2, col=1)
                
                fig.update_layout(height=600, title_text=f'Machine {selected_machine} - Time Series Analysis')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('#### Sensor Readings Over Time')
                cols = st.columns(4)
                sensors = [
                    ('Temperature_C', 'Temperature (C)', '#e74c3c'),
                    ('Vibration_Hz', 'Vibration (Hz)', '#2ecc71'),
                    ('Power_Consumption_kW', 'Power (kW)', '#3498db'),
                    ('Network_Latency_ms', 'Network Latency (ms)', '#f39c12')
                ]
                
                for i, (sensor, title, color) in enumerate(sensors):
                    with cols[i % 4]:
                        fig = px.area(machine_data, x='DateTime', y=sensor, title=title,
                                     color_discrete_sequence=[color])
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f'No data available for Machine {selected_machine}')
        else:
            st.info('Please select a specific machine to view detailed anomaly analysis')
    
    with tab3:
        st.markdown('### Maintenance Alerts and Priority')
        
        high_risk_alerts = filtered_df[filtered_df['Anomaly_Score'] >= risk_threshold]
        
        if len(high_risk_alerts) > 0:
            st.warning(f'High-risk records detected: {len(high_risk_alerts)}. Immediate attention required.')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('#### Recent High-Risk Events')
                alert_df = high_risk_alerts[['DateTime', 'Machine_ID', 'Anomaly_Score', 'Error_Rate_%', 
                                              'Temperature_C', 'Vibration_Hz', 'Operation_Mode']].head(15)
                st.dataframe(alert_df, use_container_width=True)
            
            with col2:
                st.markdown('#### Critical Sensor Values')
                critical_sensors = high_risk_alerts[['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
                                                      'Error_Rate_%', 'Packet_Loss_%']].describe()
                st.dataframe(critical_sensors, use_container_width=True)
            
            st.markdown('#### Maintenance Priority List')
            priority_machines = high_risk_alerts.groupby('Machine_ID').agg({
                'Anomaly_Score': 'mean',
                'Error_Rate_%': 'mean',
                'Temperature_C': 'mean'
            }).round(2).sort_values('Anomaly_Score', ascending=False)
            
            priority_machines.columns = ['Avg Anomaly Score', 'Avg Error Rate', 'Avg Temperature']
            st.dataframe(priority_machines.head(15), use_container_width=True)
            
            st.markdown('#### Risk Heatmap by Machine and Hour')
            high_risk_alerts['Hour'] = high_risk_alerts['DateTime'].dt.hour
            heatmap_data = high_risk_alerts.groupby(['Machine_ID', 'Hour']).size().reset_index(name='Risk Count')
            pivot_data = heatmap_data.pivot(index='Machine_ID', columns='Hour', values='Risk Count').fillna(0)
            
            fig = px.imshow(pivot_data, 
                           title='Risk Heatmap: Machine vs Hour of Day',
                           labels=dict(x='Hour of Day', y='Machine ID', color='Risk Count'),
                           color_continuous_scale='Reds',
                           aspect='auto')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success('No high-risk alerts at current threshold')
    
    with tab4:
        st.markdown('### Historical Risk Analysis')
        
        if selected_machine != 'All':
            machine_data = filtered_df[filtered_df['Machine_ID'] == selected_machine].copy()
            
            if len(machine_data) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    machine_data['Hour'] = machine_data['DateTime'].dt.hour
                    machine_data['Day'] = machine_data['DateTime'].dt.day_name()
                    hourly_risk = machine_data.groupby('Hour')['Anomaly_Score'].mean().reset_index()
                    fig = px.line(hourly_risk, x='Hour', y='Anomaly_Score', 
                                 title='Risk by Hour of Day', markers=True,
                                 line_shape='spline')
                    fig.add_hrect(y0=0.6, y1=1.0, line_width=0, fillcolor='red', opacity=0.1)
                    fig.add_hrect(y0=0.3, y1=0.6, line_width=0, fillcolor='orange', opacity=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    daily_risk = machine_data.groupby('Day')['Anomaly_Score'].mean().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                    fig = px.bar(daily_risk, x=daily_risk.index, y='Anomaly_Score',
                                title='Risk by Day of Week', color='Anomaly_Score',
                                color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    mode_risk = machine_data.groupby('Operation_Mode')['Anomaly_Score'].mean().reset_index()
                    fig = px.bar(mode_risk, x='Operation_Mode', y='Anomaly_Score', 
                                title='Average Risk by Operation Mode', 
                                color='Anomaly_Score', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    rolling_risk = machine_data.set_index('DateTime')['Anomaly_Score'].rolling(window=50).mean()
                    fig = px.line(x=rolling_risk.index, y=rolling_risk.values,
                                 title='Rolling Average Risk (50-period)',
                                 labels={'x': 'DateTime', 'y': 'Rolling Risk Score'})
                    fig.add_hline(y=risk_threshold, line_dash='dash', line_color='red')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f'No data available for Machine {selected_machine}')
        else:
            st.info('Please select a specific machine for historical analysis')
    
    with tab5:
        st.markdown('### Advanced Sensor Analytics')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(filtered_df, x='Temperature_C', y='Vibration_Hz', 
                            color='Risk_Level', size='Anomaly_Score',
                            title='Temperature vs Vibration (Color = Risk Level)',
                            color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
                            opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filtered_df, x='Power_Consumption_kW', y='Error_Rate_%',
                            color='Risk_Level', size='Anomaly_Score',
                            title='Power Consumption vs Error Rate',
                            color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
                            opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr_data = filtered_df[['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
                                      'Network_Latency_ms', 'Packet_Loss_%', 'Error_Rate_%',
                                      'Quality_Control_Defect_Rate_%', 'Anomaly_Score']].corr()
            fig = px.imshow(corr_data, text_auto=True, aspect='auto',
                           title='Correlation Matrix of All Sensors',
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            box_data = filtered_df.melt(id_vars=['Risk_Level'], 
                                        value_vars=['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 'Error_Rate_%'],
                                        var_name='Sensor', value_name='Value')
            fig = px.box(box_data, x='Sensor', y='Value', color='Risk_Level',
                        title='Sensor Distribution by Risk Level',
                        color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### Multi-Sensor Time Series Comparison')
        if selected_machine != 'All':
            machine_data = filtered_df[filtered_df['Machine_ID'] == selected_machine].copy()
            if len(machine_data) > 0:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=('Temperature and Vibration', 'Power and Error Rate', 'Network Metrics'),
                                    vertical_spacing=0.1)
                
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Temperature_C'],
                                         name='Temperature', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Vibration_Hz'],
                                         name='Vibration', line=dict(color='blue')), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Power_Consumption_kW'],
                                         name='Power', line=dict(color='green')), row=2, col=1)
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Error_Rate_%'],
                                         name='Error Rate', line=dict(color='orange')), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Network_Latency_ms'],
                                         name='Network Latency', line=dict(color='purple')), row=3, col=1)
                fig.add_trace(go.Scatter(x=machine_data['DateTime'], y=machine_data['Packet_Loss_%'],
                                         name='Packet Loss', line=dict(color='brown')), row=3, col=1)
                
                fig.update_layout(height=800, title_text=f'Machine {selected_machine} - Multi-Sensor Analysis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f'No data available for Machine {selected_machine}')
        else:
            st.info('Select a specific machine to view multi-sensor time series')
else:
    st.warning('No data available with current filters')

st.markdown('---')
st.markdown('### Executive Summary')
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f'Total Machines: {df["Machine_ID"].nunique()}')
with col2:
    st.warning(f'High-Risk Machines: {risk_summary["high_risk_machines"]}')
with col3:
    st.success(f'Risk Threshold: {risk_threshold}')

if risk_summary["high_risk_machines"] > 0:
    st.error('URGENT - Immediate inspection required for high-risk machines. Schedule maintenance within 24 hours.')
else:
    st.success('NORMAL - All machines operating within normal parameters. Continue regular monitoring.')
