import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="NDVI Analysis Dashboard")

# Function to extract date from filename
def extract_date(filename):
    date_str = filename.split('_ndvi')[1].split(',')[0]
    return datetime.strptime(date_str, '%Y-%m-%d')

# Function to process a single GeoJSON file
def process_geojson(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    date = extract_date(os.path.basename(file_path))
    records = []
    for feature in data['features']:
        props = feature['properties']
        records.append({
            'date': date,
            'Label': props['Label'],
            'ACRES': props['ACRES'],
            'ndvi_mean': props['_ndvimean'],
            'ndvi_median': props['_ndvimedian'],
            'ndvi_stdev': props['_ndvistdev']
        })
    return records

# Load data
@st.cache_data
def load_data():
    geojson_files = glob.glob('NDVI Zonal Stats/*.geojson')
    all_records = []
    for file_path in geojson_files:
        all_records.extend(process_geojson(file_path))
    df = pd.DataFrame(all_records)
    return df.sort_values('date')

df = load_data()

# Sidebar controls
st.sidebar.header('Visualization Controls')

# Pasture selection
all_pastures = sorted(df['Label'].unique())
selected_pastures = st.sidebar.multiselect(
    'Select Pastures to Display',
    all_pastures,
    default=all_pastures[:3]  # Default to first 3 pastures
)

# Date range selection
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Metric selection
metric = st.sidebar.selectbox(
    'Select Metric',
    ['ndvi_mean', 'ndvi_median', 'ndvi_stdev'],
    format_func=lambda x: x.replace('ndvi_', 'NDVI ').upper()
)

# Main content
st.title('NDVI Analysis Dashboard')

# Time series plot using Plotly
def create_time_series():
    fig = go.Figure()
    
    for pasture in selected_pastures:
        mask = (df['Label'] == pasture) & \
               (df['date'] >= pd.Timestamp(date_range[0])) & \
               (df['date'] <= pd.Timestamp(date_range[1]))
        
        pasture_data = df[mask]
        
        fig.add_trace(go.Scatter(
            x=pasture_data['date'],
            y=pasture_data[metric],
            name=pasture,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='NDVI Values Over Time',
        xaxis_title='Date',
        yaxis_title=metric.replace('ndvi_', 'NDVI ').upper(),
        hovermode='x unified'
    )
    return fig

# Display time series plot
st.plotly_chart(create_time_series(), use_container_width=True)

# Statistics and Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader('Pasture Health Status')
    
    # Calculate current and historical metrics
    current_values = df[df['Label'].isin(selected_pastures)].groupby('Label')[metric].last()
    historical_avg = df[df['Label'].isin(selected_pastures)].groupby('Label')[metric].mean()
    
    # Define health thresholds (adjust these based on your specific needs)
    def get_health_status(value):
        if value >= 0.6: return "🟢 Healthy"
        elif value >= 0.4: return "🟡 Moderate"
        else: return "🔴 Needs Attention"
    
    # Calculate trending (comparing last 2 measurements)
    recent_trend = df[df['Label'].isin(selected_pastures)].sort_values('date').groupby('Label')[metric].agg(
        lambda x: '📈' if x.iloc[-1] > x.iloc[-2] else '📉' if x.iloc[-1] < x.iloc[-2] else '➡️'
    )
    
    # Combine into status dataframe
    status_df = pd.DataFrame({
        'Current Value': current_values.round(3),
        'Historical Avg': historical_avg.round(3),
        'Status': current_values.map(get_health_status),
        'Trend': recent_trend
    })
    
    st.dataframe(status_df, use_container_width=True)
    
    # Show benchmark comparison
    st.subheader('Benchmark Comparison')
    best_performer = df.groupby('Label')[metric].mean().nlargest(1).index[0]
    benchmark_df = df[df['Label'] == best_performer][metric].mean()
    
    for pasture in selected_pastures:
        pasture_avg = df[df['Label'] == pasture][metric].mean()
        performance_ratio = (pasture_avg / benchmark_df) * 100
        st.metric(
            label=pasture,
            value=f"{pasture_avg:.3f}",
            delta=f"{performance_ratio:.1f}% of best performer"
        )

with col2:
    st.subheader('Monthly Patterns')
    df['month'] = df['date'].dt.month
    monthly_means = df[df['Label'].isin(selected_pastures)].groupby(['Label', 'month'])[metric].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(monthly_means, cmap='YlGn', annot=True, fmt='.2f', ax=ax)
    plt.title('Monthly NDVI Patterns')
    st.pyplot(fig)

# Additional Analysis
st.subheader('Comparative Analysis')
comparison_type = st.selectbox(
    'Select Analysis Type',
    ['Year-over-Year Comparison', 'Pasture Size vs NDVI', 'NDVI Distribution']
)

if comparison_type == 'Year-over-Year Comparison':
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    yearly_comparison = df[df['Label'].isin(selected_pastures)].groupby(['Label', 'year', 'month'])[metric].mean().reset_index()
    fig = px.line(yearly_comparison, 
                  x='month', 
                  y=metric, 
                  color='Label',
                  line_dash='year',
                  title='Year-over-Year Comparison')
    st.plotly_chart(fig)

elif comparison_type == 'Pasture Size vs NDVI':
    fig = px.scatter(
        df[df['Label'].isin(selected_pastures)],
        x='ACRES',
        y=metric,
        color='Label',
        title='Pasture Size vs NDVI'
    )
    st.plotly_chart(fig)

else:
    fig = px.box(
        df[df['Label'].isin(selected_pastures)],
        x='Label',
        y=metric,
        title='NDVI Distribution by Pasture'
    )
    st.plotly_chart(fig)

# Add after the existing comparative analysis section
st.subheader('Growth and Recovery Analysis')

analysis_type = st.selectbox(
    'Select Growth Analysis',
    ['Spring Growth Rate', 'Fall Recovery', 'Seasonal Transitions'],
    key='growth_analysis'
)

# Calculate growth rates (daily change in NDVI)
df['ndvi_change'] = df.groupby('Label')[metric].diff() / \
                    df.groupby('Label')['date'].diff().dt.days

if analysis_type == 'Spring Growth Rate':
    # Focus on March through June
    spring_mask = df['date'].dt.month.isin([3, 4, 5, 6])
    spring_data = df[spring_mask & df['Label'].isin(selected_pastures)].copy()
    
    # Calculate average daily growth rate for spring months
    spring_growth = spring_data.groupby(['Label', 'date'])['ndvi_change'].mean().reset_index()
    
    fig = px.box(spring_growth, 
                 x='Label', 
                 y='ndvi_change',
                 title='Spring Growth Rates by Pasture (March-June)')
    fig.update_layout(
        yaxis_title='Daily NDVI Change Rate',
        showlegend=True
    )
    st.plotly_chart(fig)
    
    # Show top performers
    avg_spring_growth = spring_growth.groupby('Label')['ndvi_change'].mean().sort_values(ascending=False)
    st.write("Top Spring Growth Performers:")
    st.dataframe(avg_spring_growth.head().round(4))

elif analysis_type == 'Fall Recovery':
    # Focus on August through October
    fall_mask = df['date'].dt.month.isin([8, 9, 10])
    fall_data = df[fall_mask & df['Label'].isin(selected_pastures)].copy()
    
    # Calculate recovery metrics
    fall_recovery = fall_data.groupby(['Label', 'date'])[metric].mean().reset_index()
    
    fig = go.Figure()
    for pasture in selected_pastures:
        pasture_data = fall_recovery[fall_recovery['Label'] == pasture]
        fig.add_trace(go.Scatter(
            x=pasture_data['date'],
            y=pasture_data[metric],
            name=pasture,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Fall Recovery Patterns (Aug-Oct)',
        xaxis_title='Date',
        yaxis_title=metric.replace('ndvi_', 'NDVI ').upper()
    )
    st.plotly_chart(fig)
    
    # Calculate recovery potential (max fall NDVI - min fall NDVI)
    recovery_potential = fall_data.groupby('Label').agg({
        metric: ['min', 'max', lambda x: max(x) - min(x)]
    }).round(3)
    recovery_potential.columns = ['Min', 'Max', 'Recovery Range']
    st.write("Fall Recovery Potential:")
    st.dataframe(recovery_potential.sort_values(('Recovery Range'), ascending=False))

else:  # Seasonal Transitions
    # Calculate average NDVI change rates during key transition periods
    # Modified season assignment to avoid duplicate labels
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'
    
    df['season'] = df['date'].dt.month.map(get_season)
    
    seasonal_changes = df[df['Label'].isin(selected_pastures)].groupby(
        ['Label', 'season']
    )['ndvi_change'].mean().unstack()
    
    # Reorder columns to match seasonal progression
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_changes = seasonal_changes[season_order]
    
    fig = px.imshow(seasonal_changes,
                    title='Seasonal NDVI Change Rates',
                    labels=dict(x='Season', y='Pasture', color='Daily NDVI Change'),
                    aspect='auto')
    st.plotly_chart(fig)
    
    # Highlight pastures with best transition characteristics
    st.write("Best Spring Transition (Winter to Spring):")
    st.dataframe(seasonal_changes['Spring'].sort_values(ascending=False).head())
    
    st.write("Best Fall Recovery (Summer to Fall):")
    st.dataframe((seasonal_changes['Fall'] - seasonal_changes['Summer']).sort_values(ascending=False).head())

# Recommendations Section
st.subheader('Management Recommendations')

# Generate recommendations based on analysis
def get_recommendations(pasture_data):
    recommendations = []
    
    # Spring growth check
    spring_growth = pasture_data[pasture_data['date'].dt.month.isin([3,4,5])][metric].mean()
    if spring_growth < 0.4:
        recommendations.append("Consider earlier fertilization for better spring growth")
    
    # Recovery patterns
    fall_recovery = pasture_data[pasture_data['date'].dt.month.isin([8,9,10])][metric].mean()
    if fall_recovery < 0.35:
        recommendations.append("Implement rotational grazing to improve recovery")
    
    # Variability check
    if pasture_data[metric].std() > 0.15:
        recommendations.append("High variability detected - review grazing patterns")
    
    return recommendations

for pasture in selected_pastures:
    pasture_data = df[df['Label'] == pasture]
    recommendations = get_recommendations(pasture_data)
    
    if recommendations:
        st.write(f"**{pasture}**")
        for rec in recommendations:
            st.write(f"- {rec}")

# Download data option
st.sidebar.download_button(
    "Download Selected Data",
    df[df['Label'].isin(selected_pastures)].to_csv(index=False).encode('utf-8'),
    "selected_ndvi_data.csv",
    "text/csv",
    key='download-csv'
)
