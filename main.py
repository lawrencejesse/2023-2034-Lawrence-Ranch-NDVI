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
import geopandas as gpd

st.set_page_config(layout="wide", page_title="NDVI Analysis Dashboard")

# Add welcome message
st.title('Demo NDVI Pasture Analysis')
st.write("""
This demo explores some ways to visualize and work with freely available NDVI satellite imagery (Sentinel-2).

I've pre-loaded cloud free NDVI imagery from 8 quarters of pasture. I've also added field boundaries to split the landbase into individual paddocks.

Open the left sidebar and choose which paddocks you want to examine, and the date range over which you are interested (2023 and 2024). And the main page will be populated with some exporatory visualizations. Feel free to play around with the info. The visualizations are rough, and the analysis are not neccessarily useful in their current state, but the idea is to generate conversation and suggestions from users.
""")

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

def create_map(df, geojson_path, selected_date, map_metric):
    # Get the selected date's data
    date_data = df[df['date'].dt.date == selected_date]
    
    # Read the GeoJSON and handle the CRS
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs('EPSG:4326')
    
    # Merge with selected date's NDVI data
    gdf = gdf.merge(date_data[['Label', map_metric]], on='Label')
    
    # Create choropleth map using plotly
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry.__geo_interface__,
        locations=gdf.index,
        color=map_metric,
        hover_name='Label',
        hover_data=[map_metric, 'ACRES'],
        color_continuous_scale='RdYlGn',
        mapbox_style='carto-positron',
        center={'lat': gdf.geometry.centroid.y.mean(), 
                'lon': gdf.geometry.centroid.x.mean()},
        zoom=14
    )
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=500,
        mapbox=dict(
            pitch=0,
            bearing=0
        )
    )
    
    return fig

df = load_data()

# Sidebar controls
st.sidebar.header('Visualization Controls')

# Pasture selection
all_pastures = sorted(df['Label'].unique())
default_pastures = ['HOME 1/4', 'NE-20', 'LUMNGAIRE']
selected_pastures = st.sidebar.multiselect(
    'Select Pastures to Display',
    all_pastures,
    default=default_pastures
)

# Available dates from the data files
available_dates = [d.strftime('%Y-%m-%d') for d in sorted(df['date'].unique())]
dates_2023 = [d for d in available_dates if d.startswith('2023')]

# Date selection
st.sidebar.header("Select Dates")
selected_dates = st.sidebar.multiselect(
    "Choose dates to analyze",
    available_dates,
    default=dates_2023  # Default to all 2023 dates
)

# Convert selected dates to datetime for filtering
if selected_dates:
    start_date = pd.Timestamp(min(selected_dates))
    end_date = pd.Timestamp(max(selected_dates))
else:
    start_date = df['date'].min()
    end_date = df['date'].max()
    st.sidebar.warning("No dates selected. Showing all dates.")

# Metric selection
metric = st.sidebar.selectbox(
    'Select Metric',
    ['ndvi_mean', 'ndvi_median', 'ndvi_stdev'],
    format_func=lambda x: x.replace('ndvi_', 'NDVI ').upper()
)

# Main content
# st.title('NDVI Analysis Dashboard')

# Time series plot using Plotly
def create_time_series():
    fig = go.Figure()
    
    # Calculate grand mean for the selected date range
    mask_dates = (df['date'] >= pd.Timestamp(start_date)) & \
                 (df['date'] <= pd.Timestamp(end_date))
    grand_mean = df[mask_dates].groupby('date')[metric].mean()
    
    # Add grand mean as a reference line
    fig.add_trace(go.Scatter(
        x=grand_mean.index,
        y=grand_mean.values,
        name='Overall Average',
        mode='lines',
        line=dict(color='rgba(0,0,0,0.5)', dash='dash', width=2),
        hovertemplate='Overall Average: %{y:.3f}<extra></extra>'
    ))
    
    # Add individual pasture lines
    for pasture in selected_pastures:
        mask = (df['Label'] == pasture) & mask_dates
        pasture_data = df[mask]
        
        # Calculate how this pasture compares to grand mean
        avg_diff = pasture_data[metric].mean() - grand_mean.loc[pasture_data['date']].mean()
        performance = "above" if avg_diff > 0 else "below"
        
        fig.add_trace(go.Scatter(
            x=pasture_data['date'],
            y=pasture_data[metric],
            name=f"{pasture} ({performance} avg)",
            mode='lines+markers',
            hovertemplate=f"{pasture}<br>Value: %{{y:.3f}}<br>Date: %{{x}}<extra></extra>"
        ))
    
    fig.update_layout(
        title='NDVI Values Over Time (Dashed Line = Dynamic Overall Average)',
        xaxis_title='Date',
        yaxis_title=metric.replace('ndvi_', 'NDVI ').upper(),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

# Display time series plot
st.plotly_chart(create_time_series(), use_container_width=True)

# Statistics and Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader('Pasture Health Status')
    
    # Calculate current values and overall average for each date
    current_date = df['date'].max()
    current_values = df[df['Label'].isin(selected_pastures) & (df['date'] == current_date)].set_index('Label')[metric]
    
    # Calculate dynamic overall average for comparison
    overall_avg_by_date = df.groupby('date')[metric].mean()
    
    # Calculate how each pasture performs relative to the overall average
    def get_health_status(row):
        pasture_data = df[df['Label'] == row.name]
        avg_diff = pasture_data[metric].mean() - overall_avg_by_date.loc[pasture_data['date']].mean()
        
        if avg_diff > 0.1: return "🟢 Above Average"
        elif avg_diff < -0.1: return "🔴 Below Average"
        else: return "🟡 Average"
    
    # Calculate trending (comparing to overall average)
    def get_trend(row):
        pasture_current = current_values.get(row.name, 0)
        current_overall_avg = overall_avg_by_date.loc[current_date]
        
        diff = pasture_current - current_overall_avg
        if abs(diff) < 0.05:
            return '➡️ At Average'
        return '📈 Above Average' if diff > 0 else '📉 Below Average'
    
    # Create status dataframe
    status_df = pd.DataFrame(index=selected_pastures)
    status_df['Current Value'] = current_values.round(3)
    status_df['Overall Average'] = overall_avg_by_date[current_date].round(3)
    status_df['Difference'] = (current_values - overall_avg_by_date[current_date]).round(3)
    status_df['Status'] = status_df.apply(get_health_status, axis=1)
    status_df['Relative Position'] = status_df.apply(get_trend, axis=1)
    
    st.dataframe(status_df, use_container_width=True)
    
    # Update benchmark comparison to use dynamic average
    st.subheader('Performance vs Overall Average')
    for pasture in selected_pastures:
        pasture_data = df[df['Label'] == pasture]
        pasture_avg = pasture_data[metric].mean()
        matching_overall_avg = overall_avg_by_date.loc[pasture_data['date']].mean()
        performance_ratio = (pasture_avg / matching_overall_avg) * 100
        
        st.metric(
            label=pasture,
            value=f"{pasture_avg:.3f}",
            delta=f"{(performance_ratio - 100):.1f}% vs average",
            delta_color="normal"
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

# Add this code after the title but before the time series plot
st.subheader('Spatial Overview')
col_map, col_info = st.columns([2, 1])

with col_info:
    # Available dates for map view
    map_dates = [d.strftime('%Y-%m-%d') for d in sorted(df['date'].unique())]
    
    # Map date selector
    st.header("Map View")
    map_date = st.selectbox(
        "Select date for map",
        map_dates,
        format_func=lambda x: x,
        key='map_date'
    )
    
    # Convert selected date string to datetime for filtering
    map_date = datetime.strptime(map_date, '%Y-%m-%d')
    
    # Add metric selector for map
    map_metric = st.selectbox(
        'Select Map Metric',
        ['ndvi_mean', 'ndvi_median', 'ndvi_stdev'],
        format_func=lambda x: x.replace('ndvi_', 'NDVI ').upper(),
        key='map_metric'
    )
    
    st.write(f"Showing {map_metric.replace('ndvi_', 'NDVI ').upper()} values")
    st.write(f"Date: {map_date.strftime('%Y-%m-%d')}")
    st.write("Hover over pastures to see details.")

with col_map:
    # Get the GeoJSON file for the selected date
    selected_file = f"NDVI Zonal Stats/_ndvi{map_date.strftime('%Y-%m-%d')}, Total Land, NDVI.data.geojson"
    
    if os.path.exists(selected_file):
        # Get the selected date's data
        date_data = df[df['date'].dt.date == map_date.date()]
        
        # Read the GeoJSON and handle the CRS
        gdf = gpd.read_file(selected_file)
        gdf = gdf.to_crs('EPSG:4326')
        
        # Merge with selected date's NDVI data
        gdf = gdf.merge(date_data[['Label', map_metric]], on='Label')
        
        # Create choropleth map using plotly
        fig = px.choropleth_mapbox(
            gdf,
            geojson=gdf.geometry.__geo_interface__,
            locations=gdf.index,
            color=map_metric,
            hover_name='Label',
            hover_data=[map_metric, 'ACRES'],
            color_continuous_scale='RdYlGn',
            mapbox_style='carto-positron',
            center={'lat': gdf.geometry.centroid.y.mean(), 
                    'lon': gdf.geometry.centroid.x.mean()},
            zoom=14
        )
        
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            height=500,
            mapbox=dict(
                pitch=0,
                bearing=0
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"No data available for {map_date}")
