import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class ImprovedLaptopPriceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def extract_numeric_features(self, df):
        """Extract numeric features from text columns"""
        df = df.copy()
        
        # Extract RAM
        def extract_ram(ram_str):
            if pd.isna(ram_str):
                return 8  # Default
            ram_str = str(ram_str).upper()
            if 'GB' in ram_str:
                return int(ram_str.replace('GB', '').strip())
            return 8
        
        # Extract Storage
        def extract_storage(storage_str):
            if pd.isna(storage_str):
                return 512  # Default
            storage_str = str(storage_str).upper()
            if 'TB' in storage_str:
                return int(float(storage_str.replace('TB', '').strip()) * 1024)
            elif 'GB' in storage_str:
                return int(storage_str.replace('GB', '').strip())
            return 512
        
        # Extract GPU memory
        def extract_gpu_memory(gpu_str):
            if pd.isna(gpu_str):
                return 0
            gpu_str = str(gpu_str).upper()
            if 'GB' in gpu_str and any(x in gpu_str for x in ['NVIDIA', 'AMD', 'RTX', 'GTX']):
                try:
                    # Extract number before GB
                    parts = gpu_str.split('GB')[0].split()
                    for part in reversed(parts):
                        if part.isdigit():
                            return int(part)
                except:
                    pass
            return 0
        
        df['ram_gb'] = df['Ram'].apply(extract_ram)
        df['storage_gb'] = df['ROM'].apply(extract_storage)
        df['gpu_memory'] = df['GPU'].apply(extract_gpu_memory)
        
        return df
    
    def create_engineered_features(self, df):
        """Create engineered features"""
        df = df.copy()
        
        # Gaming laptop indicator
        gaming_keywords = ['gaming', 'rog', 'tuf', 'predator', 'legion', 'omen', 'nitro', 'alienware']
        df['is_gaming'] = df['name'].str.lower().str.contains('|'.join(gaming_keywords), na=False).astype(int)
        
        # Premium brand indicator
        premium_brands = ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus']
        df['is_premium_brand'] = df['brand'].isin(premium_brands).astype(int)
        
        # Performance score
        df['performance_score'] = (
            df['ram_gb'] * 0.3 + 
            df['storage_gb'] * 0.0001 + 
            df['gpu_memory'] * 0.4 + 
            df['spec_rating'] * 0.3
        )
        
        # RAM to storage ratio
        df['ram_storage_ratio'] = df['ram_gb'] / (df['storage_gb'] / 1000)
        
        # Screen area
        df['screen_area'] = df['resolution_width'] * df['resolution_height'] / 1000000
        
        return df
    
    def remove_outliers(self, df):
        """Remove obvious outliers"""
        df = df.copy()
        
        # Remove laptops with unrealistic price-to-spec ratios
        # High RAM but very low price (likely data errors)
        df = df[~((df['ram_gb'] >= 32) & (df['price'] < 100000))]
        
        # Very high prices for basic specs
        df = df[~((df['ram_gb'] <= 8) & (df['gpu_memory'] == 0) & (df['price'] > 200000))]
        
        # Remove extreme outliers using IQR method
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep outliers but cap them
        df.loc[df['price'] < lower_bound, 'price'] = lower_bound
        df.loc[df['price'] > upper_bound, 'price'] = upper_bound
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Don't re-process if already processed
        if 'ram_gb' not in df.columns:
            df = self.extract_numeric_features(df)
        if 'is_gaming' not in df.columns:
            df = self.create_engineered_features(df)
        
        # Select features
        categorical_features = ['brand', 'OS']
        numerical_features = [
            'ram_gb', 'storage_gb', 'gpu_memory', 'spec_rating',
            'display_size', 'resolution_width', 'resolution_height',
            'is_gaming', 'is_premium_brand', 'performance_score',
            'ram_storage_ratio', 'screen_area'
        ]
        
        # Encode categorical features
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature].fillna('Unknown'))
            else:
                df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature].fillna('Unknown'))
        
        # Final feature list
        self.feature_names = numerical_features + [f'{f}_encoded' for f in categorical_features]
        
        return df[self.feature_names]
    
    def train(self, df):
        """Train the improved model"""
        print("Preparing features...")
        # Prepare features and get cleaned dataframe
        df_clean = self.extract_numeric_features(df)
        df_clean = self.create_engineered_features(df_clean)
        df_clean = self.remove_outliers(df_clean)
        
        X = self.prepare_features(df_clean)
        y = df_clean['price']
        
        print(f"Training data shape: {X.shape}")
        print(f"Feature names: {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting model (best performer)
        print("Training Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"\nModel Performance:")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Train RMSE: Rs.{train_rmse:,.0f}")
        print(f"Test RMSE: Rs.{test_rmse:,.0f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def predict(self, laptop_config):
        """Predict price for a laptop configuration"""
        # Create a DataFrame with the configuration
        df = pd.DataFrame([laptop_config])
        
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        return max(10000, prediction)  # Minimum price threshold
    
    def save_model(self, model_path='models/improved_model.pkl', scaler_path='models/improved_scaler.pkl'):
        """Save the trained model"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, 'models/improved_encoders.pkl')
        joblib.dump(self.feature_names, 'models/improved_features.pkl')
        
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('archive/data.csv')
    
    # Train improved model
    model = ImprovedLaptopPriceModel()
    model.train(df)
    
    # Save model
    model.save_model()
    
    # Test predictions
    print("\n=== Testing Predictions ===")
    
    test_configs = [
        {
            'brand': 'HP', 'Ram': '4GB', 'ROM': '256GB', 'GPU': 'Intel UHD Graphics',
            'OS': 'Windows 11 OS', 'spec_rating': 60, 'display_size': 15.6,
            'resolution_width': 1920, 'resolution_height': 1080,
            'name': 'HP Laptop'
        },
        {
            'brand': 'HP', 'Ram': '8GB', 'ROM': '512GB', 'GPU': 'Intel UHD Graphics',
            'OS': 'Windows 11 OS', 'spec_rating': 65, 'display_size': 15.6,
            'resolution_width': 1920, 'resolution_height': 1080,
            'name': 'HP Laptop'
        },
        {
            'brand': 'HP', 'Ram': '16GB', 'ROM': '512GB', 'GPU': '4GB NVIDIA GeForce RTX 3050',
            'OS': 'Windows 11 OS', 'spec_rating': 75, 'display_size': 15.6,
            'resolution_width': 1920, 'resolution_height': 1080,
            'name': 'HP Gaming Laptop'
        },
        {
            'brand': 'HP', 'Ram': '32GB', 'ROM': '1TB', 'GPU': '8GB NVIDIA GeForce RTX 4060',
            'OS': 'Windows 11 OS', 'spec_rating': 85, 'display_size': 15.6,
            'resolution_width': 1920, 'resolution_height': 1080,
            'name': 'HP Gaming Laptop'
        }
    ]
    
    for i, config in enumerate(test_configs):
        price = model.predict(config)
        print(f"Config {i+1}: {config['Ram']} RAM, {config['ROM']} Storage, {config['GPU'][:20]}... -> Rs.{price:,.0f}")