
# NDVI Analysis Dashboard

This repository contains a Streamlit-based web application that visualizes, analyzes, and provides insights into Normalized Difference Vegetation Index (NDVI) data for various pastures. The dashboard enables users to:

- View time-series NDVI trends across multiple pastures.
- Compare pasture-level NDVI metrics (mean, median, standard deviation) to overall averages.
- Investigate seasonal patterns, monthly cycles, year-over-year comparisons, and growth/recovery characteristics.
- Generate pasture-specific recommendations based on observed NDVI dynamics.
- Download filtered data for further offline analysis.

## Features

1. **Time-Series Plots:**  
   Visualize NDVI metrics over time, with an interactive line chart and an overall average reference line.
   
2. **Comparison and Benchmarking:**  
   - Compare selected pastures against a dynamic average line to determine relative performance.
   - Highlight underperforming or outperforming pastures with clear status indicators.
   
3. **Seasonal & Monthly Analysis:**  
   - Use monthly heatmaps to uncover seasonal patterns.
   - Explore year-over-year comparisons and identify annual trends.
   - Analyze growth rates (spring growth, fall recovery, seasonal transitions) to determine pasture health and management needs.
   
4. **Interactive Controls:**  
   - Select which pastures to view.
   - Filter data by date range.
   - Choose different NDVI metrics (mean, median, standard deviation).
   
5. **Actionable Recommendations:**  
   - Receive context-based suggestions for improving pasture health (e.g., changes in grazing rotations or fertilization timing).

6. **Data Download:**  
   Export filtered datasets for additional offline research or record-keeping.

## Installation & Setup

### Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/) package manager
- (Optional) A virtual environment tool like `venv` or `conda` for dependency management.

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/ndvi-analysis-dashboard.git
   cd ndvi-analysis-dashboard
   ```

2. **Install Dependencies:**
   This application relies on several Python libraries:
   - `streamlit`
   - `pandas`
   - `plotly`
   - `seaborn`
   - `matplotlib`
   
   You can install these dependencies with:
   ```bash
   pip install -r requirements.txt
   ```
   
   *(If you don't have a `requirements.txt` file yet, you can create one with the dependencies listed above.)*

3. **Data Preparation:**
   - Place all your NDVI `.geojson` files containing zonal statistics into a folder named `NDVI Zonal Stats`.
   - Ensure that your GeoJSON files are named according to the expected pattern, e.g. `pastureX_ndviYYYY-MM-DD.geojson`.
   - The date is extracted from the filename, so make sure the naming convention and date format match what’s used in the code. For example: `Pasture1_ndvi2023-06-01.geojson`.

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   
   Replace `app.py` with the actual filename of your Streamlit script if you have renamed it.

   After running the command, the application will start on a local server (usually at `http://localhost:8501`). Open this URL in your web browser to interact with the dashboard.

## Usage

- **Sidebar Filters:**  
  Use the sidebar to select which pastures to analyze, the date range of interest, and the NDVI metric to focus on.
  
- **Interactive Charts:**  
  Hover over charts to see detailed information. Toggle different analysis modes (Year-over-Year, Pasture Size vs NDVI, etc.) to gain more insights.
  
- **Growth and Recovery Analysis:**  
  Dive into specialized views like Spring Growth Rate or Fall Recovery to understand seasonal transitions.

- **Download Data:**  
  Once you've filtered the data to your desired subset, use the "Download Selected Data" button in the sidebar to export a CSV.

## Contributing

- If you find a bug or want to propose a new feature, feel free to open an issue.
- Pull requests are welcome! Please ensure changes are tested and documented.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code as you see fit, without restriction.

## Acknowledgments

- The NDVI metrics and analysis methods are inspired by common remote sensing and vegetation health assessment practices.
- Visualization and analytical tools used here are powered by open-source libraries like Plotly, Pandas, Seaborn, and Matplotlib.
- Streamlit provides the interactive interface that makes data exploration user-friendly.
