

# ⚡ Renewable Energy Production Prediction Using Hybrid ML Models

[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Model Performance](https://img.shields.io/badge/CNN%2BLSTM%20R%C2%B2-0.983%20PV%20%7C%200.965%20Wind-success)](README.md#results-and-best-performance)

## 🌟 Overview

This project presents a systematic and rigorous approach to **renewable energy forecasting** by implementing and comparing six distinct machine learning models, progressing from traditional statistical methods to advanced deep learning architectures. The final **CNN-LSTM hybrid model** achieves superior performance in accurately predicting both **Solar Photovoltaic (PV)** and **Wind energy production**.

---

## 🎯 Project Objectives & Problem Statement

### Problem Statement
Renewable energy sources like solar and wind are inherently **intermittent and weather-dependent**. This instability poses significant challenges for:
* **Grid Stability:** Difficulty in balancing power supply and demand.
* **Energy Planning:** Accurate scheduling of energy storage and backup power.
* **Cost Optimization:** Minimizing reliance on expensive, fast-response fossil fuel plants.

Our project tackles this clear problem by creating models that can match unstable renewable energy supply with the need for a **stable, low-cost power grid**.

### Core Objectives
* Build a **robust prediction system** for future renewable energy production.
* Systematically compare traditional ML and advanced deep learning approaches to establish performance benchmarks.
* Develop and validate the **Hybrid CNN-LSTM architecture** for optimal prediction accuracy.
* Provide actionable forecasting data to support **grid management** and **energy planning decisions**.

  <img width="1108" height="621" alt="image" src="https://github.com/user-attachments/assets/9373ea7d-f354-48d1-9784-bd8276d08399" />


---

## 🛠️ Methodology and Data Processing

### 1. Our Dataset
* **Data Size:** 38,880 samples collected at **5-minute intervals**.
* **Coverage:** Complete annual cycle, capturing all seasonal variations.
* **Features:** **70 total features** after engineering, including:
    * Meteorological Parameters: Solar Irradiance (DHI, DNI, GHI), Wind Speed, Humidity, Temperature.
    * Energy Production Data: Solar (PV) and Wind generation (target variables).
    * Temporal Features: Season, Day of Week, and Lag Features.
* **Dataset Source:** [Renewable Energy & Electricity Demand Time Series Dataset (Mendeley Data)](https://data.mendeley.com/datasets/fdfftr3tc2/1/files/fff037a3-d0e4-496f-92f7-5c5820a734f1)


### 2. Preprocessing Pipeline
The data undergoes a strict chronological pipeline to ensure quality and time-series integrity:
1.  **Load Database:** Ingesting the initial time-series data.
2.  **Feature Engineering:** Time Series Conversion, adding temporal features, and creating 70 lag features.
3.  **Outlier Removal:** Using the **Interquartile Range (IQR) Method** to ensure data consistency.
4.  **Normalization:** Applying **MinMaxScaler (Scale to 0-1)** for deep learning readiness.
5.  **Data Splitting:** **70% Training / 30% Testing**. The split is **Chronological** for time series integrity, and a separate Random Shuffling split was also created for multi-output training.



---

## 🔄 Model Evolution Pipeline

This project follows a progressive modeling approach, where we systematically assess and evolve the architecture based on the limitations observed in previous models.

### Phase 1: Traditional Machine Learning (Baseline)

| Model | Purpose | Key Limitation Addressed | Discussion Insights |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline Performance | None (Establishes low-end benchmark) | Chosen for simplicity, but fundamentally unable to capture **non-linear weather patterns** and temporal dependencies. **Failed for wind prediction ($R^2=0.481$)**. |
| **Support Vector Regression (SVR)** | Non-linear Relationships | Non-linearity | Uses the "kernel trick" (RBF Kernel) to model non-linear data. Proven more robust than Linear Regression but still lacked temporal awareness. |

### Phase 2: Ensemble Methods (Strong Non-Temporal Benchmark)

| Model | Purpose | Key Limitation Addressed | Discussion Insights |
| :--- | :--- | :--- | :--- |
| **Random Forest** (Bagging) | Reduce Variance & Overfitting | High Variance in Single Models | Handled non-linear relationships effectively, achieving drastic improvements. Revealed GHI dominance for solar prediction ($R^2_{PV}=0.983$). |
| **XGBoost** (Boosting) | Reduce Model Bias | High Model Bias | Sequentially corrects errors of previous weak learners. Provided a strong performance benchmark but **still missed crucial temporal sequence patterns**. |

### Phase 3: Deep Learning & Hybrid Architecture

| Model | Purpose | Key Contribution | Discussion Insights |
| :--- | :--- | :--- | :--- |
| **LSTM** (Long Short-Term Memory) | Capture **Temporal Dependencies** | Sequential Nature of Time Series | **Specifically designed for sequential data**. Captured time-based and seasonal patterns effectively, achieving superior wind prediction ($R^2_{Wind}=0.962$) over tree-based models. |
| **CNN + LSTM Hybrid** ⭐ | Combine Spatial & Temporal Modeling | Simultaneous Modeling of Local/Long-Term Patterns | Achieved **peak performance** by capturing both local feature patterns (CNN) and long-term dependencies (LSTM) simultaneously. |

### ⭐ Proposed Hybrid Architecture (CNN-LSTM)

| Component | Function | Detail |
| :--- | :--- | :--- |
| **Conv1D Layer** (64 filters, kernel=3) | **Spatial Feature Extraction** | Automatically identifies important local patterns and interactions within the weather feature sequences. |
| **MaxPooling1D** (pool\_size=2) | **Dimensionality Reduction** | Reduces the size of the extracted features, making the model more robust and efficient. |
| **LSTM Layer** (64 Units) | **Temporal Sequence Learning** | Processes the filtered sequences over time to understand long-term dependencies and temporal dynamics. |
| **Dropout (0.2)** | **Regularization** | Prevents overfitting during the training of the deep architecture. |
| **Dense Layers** (32 Units) | **Feature Consolidation** | Non-linear mapping and consolidation of the combined spatial and temporal features. |
| **Output Layer** (2 Units, Linear) | **Simultaneous Prediction** | Generates simultaneous predictions for both Solar (PV) and Wind power output. |



---

## 🏆 Results and Best Performance

All models were trained using standard deep learning practices (Early Stopping, MSE loss) and evaluated on key metrics including **$R^2$, RMSE, and MAE**.

### Model Performance Summary

| Model | PV $R^2$ | Wind $R^2$ | PV MAE | Wind MAE | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Linear Regression | 0.903 | 0.481 | 784.2 | 692.4 | Poorly handles non-linearity; fails for wind. |
| SVR(RBF) | 0.911 | 0.554 | 652.0 | 615.3 | Better non-linearity, but not competitive. |
| Random Forest | 0.983 | 0.946 | 198.8 | 142.2 | Excellent non-temporal performance; struggles with pure time-series. |
| XGBoost | 0.971 | 0.837 | 318.1 | 348.6 | Strong ensemble; better for PV than Wind. |
| LSTM | 0.980 | 0.962 | 506.8 | 188.2 | Superior for Wind, capturing temporal trends. |
| **CNN+LSTM** | **0.983** | **0.965** | 367.7 | **178.5** | **Peak Overall Performance (Highest combined $R^2$)** |

### Best Configuration Hyperparameters

The optimal model configuration was determined via **Grid Search** across 12 configurations (3 Window Sizes $\times$ 4 Activation Combinations).

| Parameter | Optimal Value |
| :--- | :--- |
| **Window Size** | **48** (70 features $\times$ 48 time steps) |
| **Conv Activation** | **tanh** |
| **LSTM Activation** | **sigmoid / sigmoid** |
| **Dense Activation** | **elu** |

### Final Model Performance (CNN-LSTM)

| Metric | PV Production | Wind Production |
| :--- | :--- | :--- |
| **$R^2$ (Coefficient of Determination)** | **$0.983$** | **$0.965$** |
| **RMSE (Root Mean Squared Error)** | $547$ MW | $222$ MW |

This robust performance demonstrates that the CNN-LSTM hybrid model effectively captures both the **spatial feature interactions** and the **long-term temporal patterns** in renewable energy data.

---

## 👩‍🔬 Team Roles & Contributions

| Name | Roll Number | Primary Contributions |
| :--- | :--- | :--- |
| **Mihika** | 2301CS31 | Built predictive models: Implemented Linear Regression and Random Forest. |
| **Saniya Prakash** | 2301CS49 | **Data Preprocessing Specialist**: Outlier detection, normalization, and feature engineering. Implemented XGBoost. |
| **Shefali Bishnoi** | 2301CS87 | Implemented SVR with multiple kernels and the core LSTM model. Prepared research documentation. |
| **Juhi Sahni** | 2301CS88 | **Project Lead & Architect**: Defined overall roadmap, ensured integration. Implemented and optimized the **CNN-LSTM hybrid model** and oversaw version control. |

---

