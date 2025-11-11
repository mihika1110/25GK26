# 🌟 Advanced Renewable Energy Production Predictor - Streamlit App

A comprehensive web application built with Streamlit for predicting renewable energy production using **ALL** your machine learning models from Jupyter notebooks.

## 🚀 Features

### 📊 Interactive Dashboard
- **Real-time Predictions**: Get instant renewable energy forecasts
- **6 ML Models**: Linear Regression, Random Forest, XGBoost, SVR, CNN+LSTM, and LSTM
- **Dynamic Visualizations**: Interactive charts and graphs using Plotly
- **Model Comparison**: Side-by-side performance analysis with rankings

### 🔧 Input Controls
- **Season Selection**: Choose from Winter, Spring, Summer, Autumn
- **Day of Week**: Select any day of the week
- **Solar Irradiance Parameters**:
  - DHI (Diffuse Horizontal Irradiance)
  - DNI (Direct Normal Irradiance) 
  - GHI (Global Horizontal Irradiance)
- **Weather Conditions**:
  - Wind Speed (m/s)
  - Humidity (%)
  - Temperature (°C)

### 📈 Output Predictions
- **PV Production**: Solar energy generation forecast
- **Wind Production**: Wind energy generation forecast
- **Total Renewable**: Combined renewable energy output

### 📊 Analytics & Insights
- **Dataset Overview**: Complete data statistics
- **Feature Correlations**: Weather vs energy production relationships
- **Model Performance**: MAE and R² metrics comparison
- **Interactive Charts**: Dynamic visualizations for all metrics

## 🛠️ Installation

### Option 1: Quick Setup (Recommended)
1. **Run the setup script**:
   ```bash
   python run_app.py
   ```
   This will automatically check dependencies, install missing packages, and start the app.

### Option 2: Manual Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Data Files** are in the same directory:
   - `Database.csv`
   - `train_multi_output.csv`
   - `test_multi_output.csv`

## 🚀 Running the Application

### Option 1: Complete App (All Models)
```bash
streamlit run streamlit_app_complete.py
```

### Option 2: Basic App (3 Models)
```bash
streamlit run streamlit_app.py
```

The app will automatically open at `http://localhost:8501`

## 📱 How to Use

### 1. **Set Input Parameters**
   - Use the sidebar controls to adjust:
     - Season and day of the week
     - Solar irradiance values (DHI, DNI, GHI)
     - Weather conditions (wind speed, humidity, temperature)

### 2. **Select ML Model**
   - Choose from 6 available models:
     - **Linear Regression**: Fast and interpretable baseline
     - **Random Forest**: Ensemble method for complex patterns
     - **XGBoost**: Advanced gradient boosting with regularization
     - **Support Vector Regression**: Good for non-linear relationships
     - **CNN+LSTM**: Hybrid deep learning for complex temporal patterns
     - **LSTM**: Long Short-Term Memory for sequence modeling

### 3. **View Predictions**
   - See real-time predictions for:
     - PV (Solar) Production
     - Wind Production
     - Total Renewable Energy
   - Compare predictions across different models

### 4. **Analyze Performance**
   - Review model comparison charts
   - Check MAE and R² scores
   - Explore data correlations and insights

## 🎯 Model Performance

The app includes six machine learning models from your Jupyter notebooks:

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **Linear Regression** | Traditional ML | Fast, interpretable, stable | Linear relationships, baseline |
| **Random Forest** | Ensemble | Handles complex patterns, robust | Non-linear relationships, feature interactions |
| **XGBoost** | Gradient Boosting | Advanced regularization, high performance | Complex patterns, competitions |
| **Support Vector Regression** | Kernel Method | Good for high-dimensional data | Non-linear patterns, small datasets |
| **CNN+LSTM** | Deep Learning | Hybrid approach, temporal + spatial | Complex temporal-spatial patterns |
| **LSTM** | Deep Learning | Memory cells, sequence modeling | Time series, sequential data |

## 📊 Data Features

### Input Features:
- **Season** (1-4): Winter, Spring, Summer, Autumn
- **Day of Week** (1-7): Monday through Sunday
- **Solar Irradiance**:
  - DHI: Diffuse Horizontal Irradiance (0-1000)
  - DNI: Direct Normal Irradiance (0-1000)
  - GHI: Global Horizontal Irradiance (0-1000)
- **Weather**:
  - Wind Speed (0-20 m/s)
  - Humidity (0-100%)
  - Temperature (-20 to 50°C)

### Output Variables:
- **PV Production**: Solar energy generation
- **Wind Production**: Wind energy generation

## 🔧 Technical Details

### Built With:
- **Streamlit**: Web app framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Key Features:
- **Caching**: Optimized performance with `@st.cache_data` and `@st.cache_resource`
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Instant predictions as you adjust parameters
- **Professional UI**: Clean, modern interface with custom CSS

## 🎨 Customization

You can easily customize the app by:
- Modifying input ranges in the sidebar sliders
- Adding new ML models to the comparison
- Changing visualization styles and colors
- Adjusting the layout and styling

## 📈 Future Enhancements

Potential improvements:
- Add more ML models (XGBoost, Neural Networks)
- Include time-series forecasting capabilities
- Add data export functionality
- Implement model retraining with new data
- Add weather API integration for real-time data

## 🐛 Troubleshooting

**Common Issues:**

1. **Data files not found**: Ensure all CSV files are in the same directory as `streamlit_app.py`
2. **Dependencies error**: Run `pip install -r requirements.txt`
3. **Port already in use**: Use `streamlit run streamlit_app.py --server.port 8502`

## 📞 Support

For issues or questions:
1. Check that all required data files are present
2. Verify Python environment has all dependencies
3. Ensure sufficient system memory for model training

---

**⚡ Built with ❤️ for Renewable Energy Prediction**
