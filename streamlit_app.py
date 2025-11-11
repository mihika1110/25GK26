import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Renewable Energy Production Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        # Load the main dataset
        df = pd.read_csv('Database.csv')
        
        # Load training and test data
        train_data = pd.read_csv('train_multi_output.csv')
        test_data = pd.read_csv('test_multi_output.csv')
        
        return df, train_data, test_data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None

# Train models
@st.cache_resource
def train_models(train_data):
    if train_data is None:
        return None
    
    # Prepare features and targets
    X_train = train_data[['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                         'Wind_speed', 'Humidity', 'Temperature']]
    y_train = train_data[['PV_production', 'Wind_production']]
    
    models = {}
    scalers = {}
    
    # Random Forest Model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Linear Regression Model
    linear_regression_model_function = LinearRegression()
    linear_regression_model_function.fit(X_train, y_train)
    models['Linear Regression'] = linear_regression_model_function
    
    # Support Vector Regression Model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers['SVR'] = scaler
    
    svr_base = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.2)
    svr_model = MultiOutputRegressor(svr_base)
    svr_model.fit(X_train_scaled, y_train)
    models['Support Vector Regression'] = svr_model
    
    return models, scalers

# Prediction function
def make_prediction(model, scaler, features):
    if model is None:
        return None
    
    # Convert features to numpy array
    feature_array = np.array(features).reshape(1, -1)
    
    # Scale features if needed (for SVR)
    if scaler is not None:
        feature_array = scaler.transform(feature_array)
    
    # Make prediction
    prediction = model.predict(feature_array)
    return prediction[0]

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Renewable Energy Production Predictor</h1>', unsafe_allow_html=True)
    
    # Load data
    df, train_data, test_data = load_data()
    
    if df is None:
        st.error("Unable to load data files. Please ensure Database.csv, train_multi_output.csv, and test_multi_output.csv are in the current directory.")
        return
    
    # Train models
    models, scalers = train_models(train_data)
    
    # Sidebar for input controls
    st.sidebar.header("🔧 Input Parameters")
    
    # Season selection
    season_options = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
    season = st.sidebar.selectbox("Season", options=list(season_options.keys()), 
                                 format_func=lambda x: season_options[x])
    
    # Day of the week
    day_options = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 
                   5: "Friday", 6: "Saturday", 7: "Sunday"}
    day_of_week = st.sidebar.selectbox("Day of the Week", options=list(day_options.keys()),
                                      format_func=lambda x: day_options[x])
    
    # Solar irradiance parameters
    st.sidebar.subheader("☀️ Solar Irradiance")
    dhi = st.sidebar.slider("DHI (Diffuse Horizontal Irradiance)", 
                           min_value=0.0, max_value=1000.0, value=300.0, step=10.0)
    dni = st.sidebar.slider("DNI (Direct Normal Irradiance)", 
                           min_value=0.0, max_value=1000.0, value=400.0, step=10.0)
    ghi = st.sidebar.slider("GHI (Global Horizontal Irradiance)", 
                           min_value=0.0, max_value=1000.0, value=500.0, step=10.0)
    
    # Weather parameters
    st.sidebar.subheader("🌤️ Weather Conditions")
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 
                                  min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    humidity = st.sidebar.slider("Humidity (%)", 
                                min_value=0.0, max_value=100.0, value=60.0, step=5.0)
    temperature = st.sidebar.slider("Temperature (°C)", 
                                   min_value=-20.0, max_value=50.0, value=20.0, step=1.0)
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_choice = st.sidebar.selectbox("Choose ML Model", 
                                       options=list(models.keys()),
                                       index=0)
    
    # Prepare features for prediction
    features = [season, day_of_week, dhi, dni, ghi, wind_speed, humidity, temperature]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Predictions")
        
        # Make prediction
        selected_model = models[model_choice]
        selected_scaler = scalers.get(model_choice, None)
        prediction = make_prediction(selected_model, selected_scaler, features)
        
        if prediction is not None:
            pv_pred = prediction[0]
            wind_pred = prediction[1]
            total_renewable = pv_pred + wind_pred
            
            # Display predictions in a nice format
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("🔮 Energy Production Forecast")
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                st.metric(
                    label="☀️ PV Production",
                    value=f"{pv_pred:.2f}",
                    help="Predicted solar energy production"
                )
            
            with pred_col2:
                st.metric(
                    label="💨 Wind Production", 
                    value=f"{wind_pred:.2f}",
                    help="Predicted wind energy production"
                )
            
            with pred_col3:
                st.metric(
                    label="⚡ Total Renewable",
                    value=f"{total_renewable:.2f}",
                    help="Total predicted renewable energy production"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='PV Production', x=['Solar'], y=[pv_pred], marker_color='orange'),
                go.Bar(name='Wind Production', x=['Wind'], y=[wind_pred], marker_color='blue'),
                go.Bar(name='Total Renewable', x=['Total'], y=[total_renewable], marker_color='green')
            ])
            
            fig.update_layout(
                title="Renewable Energy Production Prediction",
                xaxis_title="Energy Source",
                yaxis_title="Production",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("📈 Model Comparison")
        
        # Compare all models
        comparison_data = []
        for model_name, model in models.items():
            scaler = scalers.get(model_name, None)
            pred = make_prediction(model, scaler, features)
            if pred is not None:
                comparison_data.append({
                    'Model': model_name,
                    'PV Production': pred[0],
                    'Wind Production': pred[1],
                    'Total': pred[0] + pred[1]
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(comparison_df.set_index('Model'), use_container_width=True)
            
            # Model comparison chart
            fig_comparison = px.bar(
                comparison_df, 
                x='Model', 
                y='Total',
                title="Total Renewable Energy by Model",
                color='Total',
                color_continuous_scale='Viridis'
            )
            fig_comparison.update_layout(height=300)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Data insights section
    st.header("📊 Data Insights")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📋 Dataset Overview")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Training Samples:** {len(train_data):,}")
        st.write(f"**Test Samples:** {len(test_data):,}")
        
        # Feature statistics
        st.subheader("📊 Feature Statistics")
        numeric_features = ['DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature']
        stats_df = df[numeric_features].describe()
        st.dataframe(stats_df, use_container_width=True)
    
    with col4:
        st.subheader("🎯 Target Variables")
        
        # Target variable statistics
        target_stats = df[['PV_production', 'Wind_production']].describe()
        st.dataframe(target_stats, use_container_width=True)
        
        # Correlation with weather
        st.subheader("🔗 Weather Correlations")
        weather_corr = df[['Wind_speed', 'Humidity', 'Temperature', 'PV_production', 'Wind_production']].corr()
        
        fig_corr = px.imshow(
            weather_corr,
            text_auto=True,
            aspect="auto",
            title="Weather vs Energy Production Correlations"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Model evaluation section
    st.header("🎯 Model Performance")
    
    if test_data is not None:
        # Prepare test data
        X_test = test_data[['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                           'Wind_speed', 'Humidity', 'Temperature']]
        y_test = test_data[['PV_production', 'Wind_production']]
        
        # Evaluate models on test data
        performance_data = []
        
        for model_name, model in models.items():
            scaler = scalers.get(model_name, None)
            
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae_pv = mean_absolute_error(y_test['PV_production'], y_pred[:, 0])
            mae_wind = mean_absolute_error(y_test['Wind_production'], y_pred[:, 1])
            r2_pv = r2_score(y_test['PV_production'], y_pred[:, 0])
            r2_wind = r2_score(y_test['Wind_production'], y_pred[:, 1])
            
            performance_data.append({
                'Model': model_name,
                'PV MAE': mae_pv,
                'Wind MAE': mae_wind,
                'PV R²': r2_pv,
                'Wind R²': r2_wind
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display performance metrics
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("📊 Mean Absolute Error")
            fig_mae = px.bar(
                performance_df, 
                x='Model', 
                y=['PV MAE', 'Wind MAE'],
                title="Mean Absolute Error by Model",
                barmode='group'
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col6:
            st.subheader("📈 R² Score")
            fig_r2 = px.bar(
                performance_df, 
                x='Model', 
                y=['PV R²', 'Wind R²'],
                title="R² Score by Model",
                barmode='group'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ Renewable Energy Production Prediction System | Built with Streamlit</p>
        <p>Features: Multiple ML Models, Interactive Controls, Real-time Predictions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()