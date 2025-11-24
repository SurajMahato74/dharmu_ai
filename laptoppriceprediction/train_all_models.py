import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def train_and_save_models():
    """Train all models and save them for prediction"""
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    print("Loading data...")
    # Load the cleaned data
    data = pd.read_csv('../archive/data_cleaned.csv')
    
    # Remove unnamed index column if present
    if 'Unnamed: 0.1' in data.columns:
        data = data.drop('Unnamed: 0.1', axis=1)
    
    print(f"Dataset loaded: {data.shape}")
    
    # Prepare features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Select only numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_columns]
    
    # Handle missing values
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    print(f"Using {len(numeric_columns)} numeric features for modeling")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
    
    # Scale features (for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models_to_train = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and save each model
    model_results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")
        
        # Use scaled data for linear models, unscaled for tree-based
        if model_name in ['linear', 'ridge', 'lasso']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            use_scaled = True
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            use_scaled = False
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_results[model_name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'use_scaled': use_scaled
        }
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        
        # Save the model
        model_filename = f'{model_name}_model.pkl'
        joblib.dump(model, os.path.join(models_dir, model_filename))
        print(f"  Saved: {model_filename}")
    
    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(list(X_numeric.columns), os.path.join(models_dir, 'feature_names.pkl'))
    
    # Save model results
    joblib.dump(model_results, os.path.join(models_dir, 'model_results.pkl'))
    
    print(f"\nAll models saved to {models_dir}/ directory")
    
    # Display best model
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['r2'])
    print(f"\nBest performing model: {best_model}")
    print(f"R² Score: {model_results[best_model]['r2']:.4f}")
    
    return model_results

if __name__ == "__main__":
    train_and_save_models()