def evaluate_model(y_true, y_pred, model_name, target_name, feature_names=None, rf_model=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

    # Sample for visualization if dataset is large 
    if len(y_true) > 10000:
        sample_idx = np.random.choice(len(y_true), 5000, replace=False)
        y_true_viz = y_true[sample_idx] if isinstance(y_true, np.ndarray) else y_true.iloc[sample_idx]
        y_pred_viz = y_pred[sample_idx]
    else:
        y_true_viz = y_true
        y_pred_viz = y_pred

    # Calculate metrics on FULL dataset (not sampled)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = len(feature_names) if feature_names is not None else 1
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else r2
    ev = explained_variance_score(y_true, y_pred)

    print(f"--- {model_name} - {target_name} ---")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}") 
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Adjusted R²: {adj_r2:.3f}")
    print(f"Explained Variance: {ev:.3f}")
    print(f"Sample Size: {len(y_true)}")
    if len(y_true) > 10000:
        print(f"Visualization Sample: 5,000 points")

    # 1. Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_viz, y_pred_viz, alpha=0.6, s=10)
    max_val = max(y_true_viz.max(), y_pred_viz.max())
    min_val = min(y_true_viz.min(), y_pred_viz.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} - {target_name}\nActual vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Residual Plot
    residuals_viz = y_true_viz - y_pred_viz
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_viz, residuals_viz, alpha=0.6, s=10)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"{model_name} - {target_name}\nResidual Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. Error Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals_viz, bins=50, kde=True, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} - {target_name}\nError Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. Time Series Comparison
    plt.figure(figsize=(12, 6))
    sample_size_ts = min(200, len(y_true))
    plt.plot(range(sample_size_ts), y_true[:sample_size_ts], label='Actual', linewidth=2, alpha=0.8)
    plt.plot(range(sample_size_ts), y_pred[:sample_size_ts], label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title(f"{model_name} - {target_name}\nTime Series Comparison (First {sample_size_ts} samples)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Feature Importance (Random Forest only)
    if rf_model is not None and feature_names is not None:
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', alpha=0.8)
        plt.xlabel("Feature Importance Score")
        plt.title(f"{model_name} - {target_name}\nFeature Importance")
        
        # Add value labels
        for i, v in enumerate(importance_df['Importance']):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
            
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        print(f"\nTop Features for {target_name}:")
        print(importance_df.sort_values('Importance', ascending=False).head(10))

    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 
        'adj_r2': adj_r2, 'explained_variance': ev
    }