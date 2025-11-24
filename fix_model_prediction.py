import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ModelFixer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def load_and_fix_model(self, model_path, scaler_path):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
            
    def retrain_if_needed(self, data_path):
        try:
            data = pd.read_csv(data_path)
            X = data.drop('Price', axis=1)
            y = data['Price']
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Retrain scaler and model
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Save fixed models
            joblib.dump(self.model, 'models/best_model.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')
            joblib.dump(self.feature_columns, 'feature_columns.pkl')
            
            print("✓ Model retrained and saved successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error retraining model: {e}")
            return False
            
    def test_prediction(self, sample_data):
        try:
            if isinstance(sample_data, dict):
                sample_df = pd.DataFrame([sample_data])
            else:
                sample_df = sample_data
                
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in sample_df.columns:
                    sample_df[col] = 0
                    
            # Reorder columns to match training data
            sample_df = sample_df[self.feature_columns]
            
            # Scale and predict
            sample_scaled = self.scaler.transform(sample_df)
            prediction = self.model.predict(sample_scaled)[0]
            
            print(f"✓ Test prediction: ₹{prediction:,.0f}")
            return prediction
            
        except Exception as e:
            print(f"✗ Error in prediction: {e}")
            return None

if __name__ == "__main__":
    fixer = ModelFixer()
    
    # Try to load existing model
    if not fixer.load_and_fix_model('models/best_model.pkl', 'models/scaler.pkl'):
        print("Retraining model...")
        fixer.retrain_if_needed('archive/data_cleaned.csv')
    
    # Test with sample data
    sample = {
        'ram_gb': 8,
        'storage_gb': 512,
        'screen_size': 15.6,
        'weight_kg': 2.0,
        'warranty_years': 1,
        'spec_rating': 4.0
    }
    
    fixer.test_prediction(sample)