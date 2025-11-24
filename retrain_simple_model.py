import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

def create_simple_model():
    """Create a simple model that matches Django preprocessing"""
    
    # Load data
    data = pd.read_csv('archive/data.csv')
    
    # Simple feature engineering matching Django
    def preprocess_data(df):
        # Brand encoding
        brand_mapping = {'HP': 0, 'Dell': 1, 'Lenovo': 2, 'Asus': 3, 'Acer': 4, 'Apple': 5, 'MSI': 6, 'Samsung': 7, 'Infinix': 8, 'Xiaomi': 9}
        df['brand_encoded'] = df['brand'].map(brand_mapping).fillna(0)
        
        # Extract RAM numeric
        df['Ram_numeric'] = df['Ram'].str.extract(r'(\d+)').astype(float)
        
        # Extract ROM numeric  
        df['ROM_numeric'] = df['ROM'].str.extract(r'(\d+)').astype(float)
        
        # Processor encoding (simplified)
        processor_mapping = {
            '11th Gen Intel Core i5': 0, '12th Gen Intel Core i5': 1, '13th Gen Intel Core i5': 2,
            '12th Gen Intel Core i7': 3, '13th Gen Intel Core i7': 4, '5th Gen AMD Ryzen 5': 5,
            '7th Gen AMD Ryzen 7': 6, 'Apple M2': 7
        }
        df['processor_encoded'] = df['processor'].map(processor_mapping).fillna(0)
        df['CPU_encoded'] = df['processor_encoded']  # Same as processor
        
        # GPU encoding
        gpu_mapping = {
            'Intel Integrated': 0, 'Intel Iris Xe': 1, 'NVIDIA GeForce RTX 3050': 2,
            'NVIDIA GeForce RTX 4060': 3, 'AMD Radeon': 4, 'NVIDIA GeForce GTX 1650': 5
        }
        df['GPU_encoded'] = df['GPU'].str.contains('Intel', case=False, na=False).astype(int)
        
        # OS encoding
        os_mapping = {
            'Windows 11 OS': 0, 'Windows 10 OS': 1, 'Mac OS': 2, 'DOS OS': 3, 'Chrome OS': 4
        }
        df['OS_encoded'] = df['OS'].map(os_mapping).fillna(0)
        
        # ROM type encoding
        rom_type_mapping = {'SSD': 0, 'Hard-Disk': 1}
        df['ROM_type_encoded'] = df['ROM_type'].map(rom_type_mapping).fillna(0)
        
        # Simple features
        df['aspect_ratio'] = 1.78  # Default 16:9
        df['screen_area'] = df['display_size'] ** 2 * 0.5
        df['ram_rom_ratio'] = df['Ram_numeric'] / df['ROM_numeric']
        df['performance_score'] = df['Ram_numeric'] * 0.3 + 10 * 0.4 + df['spec_rating'] * 0.3
        df['is_gaming'] = df['GPU'].str.contains('RTX|GTX', case=False, na=False).astype(int)
        df['processor_gen'] = 12  # Default
        
        return df
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Select features that match Django preprocessing
    feature_columns = [
        'brand_encoded', 'spec_rating', 'processor_encoded', 'CPU_encoded',
        'Ram_numeric', 'ROM_numeric', 'ROM_type_encoded',
        'GPU_encoded', 'display_size', 'resolution_width', 'resolution_height',
        'OS_encoded', 'warranty', 'aspect_ratio', 'screen_area', 'ram_rom_ratio',
        'performance_score', 'is_gaming', 'processor_gen'
    ]
    
    # Prepare data
    X = data[feature_columns].fillna(0)
    y = data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: Rs.{mae:,.0f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/simple_model.pkl')
    joblib.dump(scaler, 'models/simple_scaler.pkl')
    
    print("Model saved successfully!")
    
    # Test with sample data
    test_sample = {
        'brand_encoded': 0,  # HP
        'spec_rating': 70,
        'processor_encoded': 1,  # 12th Gen i5
        'CPU_encoded': 1,
        'Ram_numeric': 8,
        'ROM_numeric': 512,
        'ROM_type_encoded': 0,  # SSD
        'GPU_encoded': 0,  # Intel
        'display_size': 15.6,
        'resolution_width': 1920,
        'resolution_height': 1080,
        'OS_encoded': 0,  # Windows 11
        'warranty': 1,
        'aspect_ratio': 1.78,
        'screen_area': 121.68,
        'ram_rom_ratio': 0.015625,
        'performance_score': 25.4,
        'is_gaming': 0,
        'processor_gen': 12
    }
    
    sample_df = pd.DataFrame([test_sample])
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)[0]
    
    print(f"\nTest Prediction: Rs.{prediction:,.0f}")
    
    return model, scaler

if __name__ == "__main__":
    create_simple_model()