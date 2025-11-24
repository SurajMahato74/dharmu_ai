#!/usr/bin/env python3
"""
Test Django Model Loading
"""

import os
import sys
import joblib

def test_model_loading():
    """Test if Django can load the models correctly"""
    
    models_dir = 'models'
    
    # Test files that should exist
    required_files = [
        'best_model.pkl',
        'scaler.pkl',
        'random_forest_model.pkl',
        'gradient_boosting_model.pkl',
        'linear_regression_model.pkl'
    ]
    
    print("🔍 Testing Model Files...")
    print("=" * 40)
    
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            try:
                model = joblib.load(file_path)
                print(f"✅ {file} - OK")
            except Exception as e:
                print(f"❌ {file} - Error loading: {e}")
        else:
            print(f"❌ {file} - File not found")
    
    # Test prediction with best model
    print(f"\n🤖 Testing Prediction...")
    print("-" * 30)
    
    try:
        model_path = os.path.join(models_dir, 'best_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Test with sample data (16 features)
            import numpy as np
            sample_features = np.array([[
                0,      # brand_encoded (HP)
                8,      # ram_gb
                256,    # storage_gb
                1,      # is_ssd
                15.6,   # display_size
                1920,   # resolution_width
                1080,   # resolution_height
                1,      # os_encoded
                1,      # warranty
                1.78,   # aspect_ratio
                195.0,  # screen_area
                0.03,   # ram_rom_ratio
                45.0,   # performance_score
                0,      # is_gaming
                11,     # processor_gen
                70      # spec_rating
            ]])
            
            features_scaled = scaler.transform(sample_features)
            prediction = model.predict(features_scaled)[0]
            
            print(f"✅ Sample prediction: ₹{prediction:,.0f}")
            
            if 20000 <= prediction <= 200000:
                print("✅ Prediction in reasonable range")
            else:
                print("⚠️  Prediction outside expected range")
                
        else:
            print("❌ Required model files not found")
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    test_model_loading()