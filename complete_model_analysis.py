import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CompleteModelAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Remove unnamed index column if present
        if 'Unnamed: 0.1' in self.data.columns:
            self.data = self.data.drop('Unnamed: 0.1', axis=1)
        
        print(f"Dataset loaded: {self.data.shape}")
        print(f"Target variable (price) statistics:")
        print(self.data['price'].describe())
        
    def prepare_data(self):
        """Prepare features and target variables"""
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        
        # Select only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        print(f"Using {len(numeric_columns)} numeric features for modeling")
        print(f"Missing values handled: {X_numeric.isnull().sum().sum()} NaN values filled with median")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, X_numeric.columns.tolist()
    
    def train_models(self):
        """Train multiple models for comparison"""
        print("\n=== Training Multiple Models ===")
        
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names = self.prepare_data()
        self.feature_names = feature_names
        
        # Define models
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train model (use scaled data for linear models, unscaled for tree-based)
            if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred,
                'y_test': y_test,
                'use_scaled': 'Linear' in name or 'Ridge' in name or 'Lasso' in name
            }
            
            print(f"{name} Results:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
        
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    
    def feature_importance_analysis(self):
        """Comprehensive feature importance analysis"""
        print("\n=== Feature Importance Analysis ===")
        
        # Get the best performing model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_model = self.results[best_model_name]['model']
        
        print(f"Best model: {best_model_name} (R² = {self.results[best_model_name]['r2']:.4f})")
        
        # Get feature names
        feature_names = self.feature_names
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            
            # Create feature importance DataFrame
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_df.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return feature_df
        
        return None
    
    def cross_validation_analysis(self):
        """Cross-validation analysis for all models"""
        print("\n=== Cross-Validation Analysis ===")
        
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, _ = self.prepare_data()
        
        cv_results = {}
        
        for name, result in self.results.items():
            print(f"\nCross-validation for {name}:")
            
            # Use appropriate data
            if result['use_scaled']:
                X_cv = X_train_scaled
            else:
                X_cv = X_train
            
            # Perform cross-validation
            cv_scores = cross_val_score(result['model'], X_cv, y_train, cv=5, scoring='r2')
            
            cv_results[name] = cv_scores
            print(f"  CV R² Scores: {cv_scores}")
            print(f"  Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Plot cross-validation results
        plt.figure(figsize=(10, 6))
        cv_data = []
        cv_labels = []
        
        for name, scores in cv_results.items():
            cv_data.extend(scores)
            cv_labels.extend([name] * len(scores))
        
        cv_df = pd.DataFrame({'R² Score': cv_data, 'Model': cv_labels})
        sns.boxplot(data=cv_df, x='Model', y='R² Score')
        plt.xticks(rotation=45)
        plt.title('Cross-Validation R² Scores by Model')
        plt.tight_layout()
        plt.show()
        
        return cv_results
    
    def model_comparison(self):
        """Comprehensive model comparison"""
        print("\n=== Model Performance Comparison ===")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            name: {
                'R² Score': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae']
            }
            for name, result in self.results.items()
        }).T
        
        print(comparison_df.round(4))
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² Score comparison
        axes[0].bar(comparison_df.index, comparison_df['R² Score'])
        axes[0].set_title('R² Score Comparison')
        axes[0].set_ylabel('R² Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1].bar(comparison_df.index, comparison_df['RMSE'])
        axes[1].set_title('RMSE Comparison')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[2].bar(comparison_df.index, comparison_df['MAE'])
        axes[2].set_title('MAE Comparison')
        axes[2].set_ylabel('MAE')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def residual_analysis(self):
        """Residual analysis for best model"""
        print("\n=== Residual Analysis ===")
        
        # Get best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_result = self.results[best_model_name]
        
        y_test = best_result['y_test']
        y_pred = best_result['y_pred']
        residuals = y_test - y_pred
        
        # Plot residuals
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predicted - {best_model_name}')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.show()
        
        # Residual statistics
        print(f"Residual Statistics for {best_model_name}:")
        print(f"  Mean Residual: {residuals.mean():.2f}")
        print(f"  Std Residual: {residuals.std():.2f}")
        print(f"  Min Residual: {residuals.min():.2f}")
        print(f"  Max Residual: {residuals.max():.2f}")
        
        return residuals
    
    def prediction_examples(self):
        """Show prediction examples"""
        print("\n=== Prediction Examples ===")
        
        # Get best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_result = self.results[best_model_name]
        
        y_test = best_result['y_test']
        y_pred = best_result['y_pred']
        
        # Show first 10 predictions
        examples_df = pd.DataFrame({
            'Actual': y_test.iloc[:10].values,
            'Predicted': y_pred[:10],
            'Error': y_test.iloc[:10].values - y_pred[:10],
            'Error %': ((y_test.iloc[:10].values - y_pred[:10]) / y_test.iloc[:10].values * 100)
        })
        
        print("First 10 Predictions:")
        print(examples_df.round(2))
        
        return examples_df
    
    def full_analysis(self):
        """Run complete model analysis"""
        print("Starting Complete Model Analysis...")
        print("=" * 50)
        
        # Train models
        self.train_models()
        
        # Feature importance
        feature_df = self.feature_importance_analysis()
        
        # Cross-validation
        cv_results = self.cross_validation_analysis()
        
        # Model comparison
        comparison_df = self.model_comparison()
        
        # Residual analysis
        residuals = self.residual_analysis()
        
        # Prediction examples
        examples = self.prediction_examples()
        
        print("\n" + "=" * 50)
        print("Complete Model Analysis Finished!")
        
        return {
            'results': self.results,
            'feature_importance': feature_df,
            'cross_validation': cv_results,
            'comparison': comparison_df,
            'residuals': residuals,
            'examples': examples
        }

if __name__ == "__main__":
    # Run complete analysis
    analyzer = CompleteModelAnalysis('archive/data_cleaned.csv')
    analysis_results = analyzer.full_analysis()