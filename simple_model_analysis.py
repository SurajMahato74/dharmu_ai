import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleModelAnalysis:
    def __init__(self, model_paths, scaler_path, data_path):
        self.data = pd.read_csv(data_path)
        
        # Remove unnamed index column if present
        if 'Unnamed: 0.1' in self.data.columns:
            self.data = self.data.drop('Unnamed: 0.1', axis=1)
        
        print("=== SIMPLE MODEL ANALYSIS ===")
        print(f"Dataset loaded: {self.data.shape}")
        print(f"Available columns: {len(self.data.columns)} features")
        print(f"Target variable (price) range: ${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}")
        
        # Load models and scalers
        self.models = {}
        
        for name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    print(f"[OK] Loaded {name}: {model_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {name}: {e}")
            else:
                print(f"[MISSING] Model file not found: {model_path}")
        
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"[OK] Loaded scaler: {scaler_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load scaler: {e}")
        else:
            print(f"[MISSING] Scaler file not found: {scaler_path}")
            self.scaler = None
        
        print("=" * 50)
    
    def prepare_data(self):
        print(f"\n>>> PREPARING DATA:")
        
        # Use only numeric features
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if 'price' in numeric_columns:
            numeric_columns = numeric_columns.drop('price')
        
        X = self.data[numeric_columns]
        y = self.data['price']
        
        print(f"  >> Using {len(numeric_columns)} numeric features")
        print(f"  >> Target variable: {len(y)} samples")
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"  >> Filling {missing_count} missing values with median")
            X = X.fillna(X.median())
        else:
            print(f"  >> No missing values found")
        
        print(f"  >> Data preparation complete")
        return X, y
    
    def analyze_single_model(self, model_name, model):
        print(f"\n{'='*20} ANALYZING {model_name.upper()} {'='*20}")
        
        X, y = self.prepare_data()
        
        try:
            # Use scaling if available
            if self.scaler is not None:
                print(f"\n>>> APPLYING SCALING:")
                X_scaled = self.scaler.transform(X)
                print(f"  >> Features scaled using pre-fitted scaler")
            else:
                X_scaled = X.values
                print(f"  >> No scaler available, using raw features")
            
            # Split for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Get predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            print(f"\n>>> MODEL EVALUATION:")
            print(f"  >> Training R2 Score: {train_r2:.4f}")
            print(f"  >> Test R2 Score: {test_r2:.4f}")
            print(f"  >> RMSE: ${test_rmse:,.2f}")
            print(f"  >> MAE: ${test_mae:,.2f}")
            
            # Cross-validation
            print(f"\n>>> CROSS-VALIDATION:")
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            print(f"  >> CV R2 Scores: {cv_scores}")
            print(f"  >> Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                print(f"\n>>> FEATURE IMPORTANCE:")
                feature_names = X.columns
                importances = model.feature_importances_
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"  >> Top 10 Most Important Features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"    {i+1:2d}. {row['feature']:20s} ({row['importance']:.4f})")
            
            # Sample predictions
            print(f"\n>>> SAMPLE PREDICTIONS (First 5 test samples):")
            sample_df = pd.DataFrame({
                'Actual': y_test.iloc[:5].values,
                'Predicted': y_pred_test[:5],
                'Error': y_test.iloc[:5].values - y_pred_test[:5],
                'Error %': ((y_test.iloc[:5].values - y_pred_test[:5]) / y_test.iloc[:5].values * 100)
            })
            print(sample_df.round(2))
            
            return {
                'model_name': model_name,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        except Exception as e:
            print(f"  >> ERROR analyzing {model_name}: {str(e)}")
            return None
    
    def run_analysis(self):
        print(f"\n>>> STARTING ANALYSIS OF YOUR 3 MODELS <<<")
        
        results = {}
        
        for name in self.models.keys():
            result = self.analyze_single_model(name, self.models[name])
            if result:
                results[name] = result
        
        # Summary comparison
        if len(results) > 1:
            print(f"\n{'='*60}")
            print(f">>> SUMMARY COMPARISON")
            print(f"{'='*60}")
            
            summary_df = pd.DataFrame({
                name: {
                    'Test R2': result['test_r2'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'CV Mean': result['cv_mean']
                }
                for name, result in results.items()
            }).T
            
            print(summary_df.round(4))
            
            # Find best model
            best_model = max(results.keys(), key=lambda k: results[k]['test_r2'])
            print(f"\n>>> BEST MODEL: {best_model}")
            print(f"   Test R2 Score: {results[best_model]['test_r2']:.4f}")
            print(f"   RMSE: ${results[best_model]['rmse']:,.2f}")
        
        print(f"\n>>> ANALYSIS COMPLETE!")
        return results

if __name__ == "__main__":
    # Your 3 models
    model_files = {
        'Random Forest': 'models/random_forest_enhanced.pkl',
        'Gradient Boosting': 'models/gradient_boosting_enhanced.pkl', 
        'Linear Regression': 'models/linear_regression_enhanced.pkl',
    }
    
    scaler_file = 'models/scaler_enhanced.pkl'
    data_file = 'archive/data_cleaned.csv'
    
    # Run the analysis
    analyzer = SimpleModelAnalysis(model_files, scaler_file, data_file)
    results = analyzer.run_analysis()