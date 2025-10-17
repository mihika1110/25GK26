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
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Renewable Energy Production Predictor",
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
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .performance-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .badge-excellent { background-color: #d4edda; color: #155724; }
    .badge-good { background-color: #d1ecf1; color: #0c5460; }
    .badge-fair { background-color: #fff3cd; color: #856404; }
    .badge-poor { background-color: #f8d7da; color: #721c24; }
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

# Build CNN+LSTM model
def build_cnn_lstm_model():
    model = Sequential()
    
    # CNN layer
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', 
                     input_shape=(1, 8)))
    model.add(MaxPooling1D(pool_size=1))
    
    # LSTM layer
    model.add(LSTM(units=64, activation='tanh', recurrent_activation='sigmoid', 
                   return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layer
    model.add(Dense(32, activation='relu'))
    
    # Output layer
    model.add(Dense(2, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Build LSTM model
def build_lstm_model():
    model = Sequential()
    
    # LSTM Layer
    model.add(LSTM(units=64, activation='tanh', recurrent_activation='sigmoid',
                   input_shape=(1, 8), return_sequences=False))
    
    # Dropout
    model.add(Dropout(0.2))
    
    # Dense Layer
    model.add(Dense(32, activation='relu'))
    
    # Output Layer
    model.add(Dense(2, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Train all models
@st.cache_resource
def train_all_models(train_data):
    if train_data is None:
        return None, None, None, None
    
    # Prepare features and targets
    X_train = train_data[['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                         'Wind_speed', 'Humidity', 'Temperature']].values
    y_train = train_data[['PV_production', 'Wind_production']].values
    
    models = {}
    scalers = {}
    model_info = {}
    
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models['Linear Regression'] = lr_model
    model_info['Linear Regression'] = {
        'type': 'sklearn',
        'description': 'Simple linear regression for baseline comparison',
        'complexity': 'Low',
        'speed': 'Fast'
    }
    
    # 2. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    model_info['Random Forest'] = {
        'type': 'sklearn',
        'description': 'Ensemble method using multiple decision trees',
        'complexity': 'Medium',
        'speed': 'Medium'
    }
    
    # 3. XGBoost
    xgb_base = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb_model = MultiOutputRegressor(xgb_base)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    model_info['XGBoost'] = {
        'type': 'sklearn',
        'description': 'Gradient boosting with advanced regularization',
        'complexity': 'High',
        'speed': 'Medium'
    }
    
    # 4. Support Vector Regression
    scaler_svr = StandardScaler()
    X_train_scaled = scaler_svr.fit_transform(X_train)
    scalers['SVR'] = scaler_svr
    
    svr_base = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.2)
    svr_model = MultiOutputRegressor(svr_base)
    svr_model.fit(X_train_scaled, y_train)
    models['Support Vector Regression'] = svr_model
    model_info['Support Vector Regression'] = {
        'type': 'sklearn',
        'description': 'Support vector machine for regression',
        'complexity': 'High',
        'speed': 'Slow'
    }
    
    # 5. CNN+LSTM Model
    scaler_cnn_lstm_X = MinMaxScaler()
    scaler_cnn_lstm_y = MinMaxScaler()
    X_train_scaled_cnn = scaler_cnn_lstm_X.fit_transform(X_train)
    y_train_scaled_cnn = scaler_cnn_lstm_y.fit_transform(y_train)
    
    # Reshape for CNN+LSTM
    X_train_reshaped = X_train_scaled_cnn.reshape((X_train_scaled_cnn.shape[0], 1, X_train_scaled_cnn.shape[1]))
    
    cnn_lstm_model = build_cnn_lstm_model()
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    
    cnn_lstm_model.fit(
        X_train_reshaped, y_train_scaled_cnn,
        epochs=25,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    
    models['CNN+LSTM'] = cnn_lstm_model
    scalers['CNN+LSTM_X'] = scaler_cnn_lstm_X
    scalers['CNN+LSTM_y'] = scaler_cnn_lstm_y
    model_info['CNN+LSTM'] = {
        'type': 'tensorflow',
        'description': 'Hybrid CNN+LSTM for complex temporal patterns',
        'complexity': 'Very High',
        'speed': 'Slow'
    }
    
    # 6. LSTM Model
    scaler_lstm_X = MinMaxScaler()
    scaler_lstm_y = MinMaxScaler()
    X_train_scaled_lstm = scaler_lstm_X.fit_transform(X_train)
    y_train_scaled_lstm = scaler_lstm_y.fit_transform(y_train)
    
    # Reshape for LSTM
    X_train_reshaped_lstm = X_train_scaled_lstm.reshape((X_train_scaled_lstm.shape[0], 1, X_train_scaled_lstm.shape[1]))
    
    lstm_model = build_lstm_model()
    
    # Train with early stopping
    early_stop_lstm = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    
    lstm_model.fit(
        X_train_reshaped_lstm, y_train_scaled_lstm,
        epochs=20,
        batch_size=64,
        callbacks=[early_stop_lstm],
        verbose=0
    )
    
    models['LSTM'] = lstm_model
    scalers['LSTM_X'] = scaler_lstm_X
    scalers['LSTM_y'] = scaler_lstm_y
    model_info['LSTM'] = {
        'type': 'tensorflow',
        'description': 'Long Short-Term Memory for sequence modeling',
        'complexity': 'High',
        'speed': 'Medium'
    }
    
    return models, scalers, model_info

# Prediction function
def make_prediction(model, scaler_X, scaler_y, features, model_name):
    if model is None:
        return None
    
    # Convert features to numpy array
    feature_array = np.array(features).reshape(1, -1)
    
    # Handle different model types
    if model_name in ['CNN+LSTM', 'LSTM']:
        # Scale features
        feature_array_scaled = scaler_X.transform(feature_array)
        # Reshape for LSTM/CNN
        feature_array_reshaped = feature_array_scaled.reshape((1, 1, feature_array_scaled.shape[1]))
        # Predict
        prediction_scaled = model.predict(feature_array_reshaped, verbose=0)
        # Inverse transform
        prediction = scaler_y.inverse_transform(prediction_scaled)
        return prediction[0]
    else:
        # For sklearn models
        if scaler_X is not None:
            feature_array = scaler_X.transform(feature_array)
        prediction = model.predict(feature_array)
        return prediction[0]

# Get performance badge
def get_performance_badge(r2_score):
    if r2_score >= 0.9:
        return '<span class="performance-badge badge-excellent">Excellent</span>'
    elif r2_score >= 0.7:
        return '<span class="performance-badge badge-good">Good</span>'
    elif r2_score >= 0.5:
        return '<span class="performance-badge badge-fair">Fair</span>'
    else:
        return '<span class="performance-badge badge-poor">Poor</span>'

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Advanced Renewable Energy Production Predictor</h1>', unsafe_allow_html=True)
    
    # Load data
    df, train_data, test_data = load_data()
    
    if df is None:
        st.error("Unable to load data files. Please ensure Database.csv, train_multi_output.csv, and test_multi_output.csv are in the current directory.")
        return
    
    # Train models
    with st.spinner("Training all models... This may take a moment."):
        models, scalers, model_info = train_all_models(train_data)
    
    if models is None:
        st.error("Failed to train models.")
        return
    
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
    
    # Model information
    if model_choice in model_info:
        info = model_info[model_choice]
        st.sidebar.markdown(f"""
        <div class="model-card">
        <h4>Model Info</h4>
        <p><strong>Type:</strong> {info['type']}</p>
        <p><strong>Description:</strong> {info['description']}</p>
        <p><strong>Complexity:</strong> {info['complexity']}</p>
        <p><strong>Speed:</strong> {info['speed']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prepare features for prediction
    features = [season, day_of_week, dhi, dni, ghi, wind_speed, humidity, temperature]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Predictions")
        
        # Make prediction
        selected_model = models[model_choice]
        selected_scaler_X = scalers.get(model_choice, None)
        selected_scaler_y = None
        
        # Handle different scaler types for deep learning models
        if model_choice == 'CNN+LSTM':
            selected_scaler_X = scalers.get('CNN+LSTM_X', None)
            selected_scaler_y = scalers.get('CNN+LSTM_y', None)
        elif model_choice == 'LSTM':
            selected_scaler_X = scalers.get('LSTM_X', None)
            selected_scaler_y = scalers.get('LSTM_y', None)
        
        prediction = make_prediction(selected_model, selected_scaler_X, selected_scaler_y, features, model_choice)
        
        if prediction is not None:
            pv_pred = prediction[0]
            wind_pred = prediction[1]
            total_renewable = pv_pred + wind_pred
            
            # Display predictions in a nice format
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader(f"🔮 Energy Production Forecast - {model_choice}")
            
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
                title=f"Renewable Energy Production Prediction - {model_choice}",
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
            scaler_X = scalers.get(model_name, None)
            scaler_y = None
            
            # Handle different scaler types
            if model_name == 'CNN+LSTM':
                scaler_X = scalers.get('CNN+LSTM_X', None)
                scaler_y = scalers.get('CNN+LSTM_y', None)
            elif model_name == 'LSTM':
                scaler_X = scalers.get('LSTM_X', None)
                scaler_y = scalers.get('LSTM_y', None)
            
            pred = make_prediction(model, scaler_X, scaler_y, features, model_name)
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
            fig_comparison.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Model Performance Section
    st.header("🎯 Model Performance Analysis")
    
    if test_data is not None:
        # Prepare test data
        X_test = test_data[['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                           'Wind_speed', 'Humidity', 'Temperature']].values
        y_test = test_data[['PV_production', 'Wind_production']].values
        
        # Evaluate models on test data
        performance_data = []
        
        for model_name, model in models.items():
            scaler_X = scalers.get(model_name, None)
            scaler_y = None
            
            # Handle different scaler types
            if model_name == 'CNN+LSTM':
                scaler_X = scalers.get('CNN+LSTM_X', None)
                scaler_y = scalers.get('CNN+LSTM_y', None)
                X_test_scaled = scaler_X.transform(X_test)
                X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                y_pred_scaled = model.predict(X_test_reshaped, verbose=0)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
            elif model_name == 'LSTM':
                scaler_X = scalers.get('LSTM_X', None)
                scaler_y = scalers.get('LSTM_y', None)
                X_test_scaled = scaler_X.transform(X_test)
                X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                y_pred_scaled = model.predict(X_test_reshaped, verbose=0)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
            else:
                if scaler_X is not None:
                    X_test_scaled = scaler_X.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae_pv = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
            mae_wind = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
            r2_pv = r2_score(y_test[:, 0], y_pred[:, 0])
            r2_wind = r2_score(y_test[:, 1], y_pred[:, 1])
            
            performance_data.append({
                'Model': model_name,
                'PV MAE': mae_pv,
                'Wind MAE': mae_wind,
                'PV R²': r2_pv,
                'Wind R²': r2_wind,
                'Avg R²': (r2_pv + r2_wind) / 2
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display performance metrics
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📊 Mean Absolute Error")
            fig_mae = px.bar(
                performance_df, 
                x='Model', 
                y=['PV MAE', 'Wind MAE'],
                title="Mean Absolute Error by Model",
                barmode='group'
            )
            fig_mae.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col4:
            st.subheader("📈 R² Score")
            fig_r2 = px.bar(
                performance_df, 
                x='Model', 
                y=['PV R²', 'Wind R²'],
                title="R² Score by Model",
                barmode='group'
            )
            fig_r2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Performance ranking
        st.subheader("🏆 Model Ranking")
        performance_df_sorted = performance_df.sort_values('Avg R²', ascending=False)
        
        for idx, row in performance_df_sorted.iterrows():
            col5, col6, col7 = st.columns([2, 1, 1])
            with col5:
                st.write(f"**{row['Model']}**")
            with col6:
                st.write(f"Avg R²: {row['Avg R²']:.3f}")
            with col7:
                st.markdown(get_performance_badge(row['Avg R²']), unsafe_allow_html=True)
    
    # Data insights section
    st.header("📊 Dataset Insights")
    
    col8, col9 = st.columns(2)
    
    with col8:
        st.subheader("📋 Dataset Overview")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Training Samples:** {len(train_data):,}")
        st.write(f"**Test Samples:** {len(test_data):,}")
        st.write(f"**Available Models:** {len(models)}")
        
        # Feature statistics
        st.subheader("📊 Feature Statistics")
        numeric_features = ['DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature']
        stats_df = df[numeric_features].describe()
        st.dataframe(stats_df, use_container_width=True)
    
    with col9:
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
            title="Weather vs Energy Production Correlations",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ Advanced Renewable Energy Production Prediction System | Built with Streamlit</p>
        <p>Models: Linear Regression, Random Forest, XGBoost, SVR, CNN+LSTM, LSTM | Interactive Controls | Real-time Predictions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
