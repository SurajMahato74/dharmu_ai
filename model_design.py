import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class LaptopPricePredictor:
    def __init__(self):
        """Initialize the laptop price predictor"""
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, X, y):
        """Load preprocessed data"""
        self.X = X
        self.y = y
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split completed:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
    def scale_features(self):
        """Scale features for models that require it"""
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers['standard'] = scaler
        print("Features scaled using StandardScaler")
        
    def create_linear_regression_model(self):
        """Create and train Linear Regression model"""
        print("\n" + "="*50)
        print("LINEAR REGRESSION MODEL")
        print("="*50)
        
        # Create model
        lr_model = LinearRegression()
        
        # Train model
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred_train = lr_model.predict(self.X_train_scaled)
        y_pred_test = lr_model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(lr_model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        
        # Store model and performance
        self.models['Linear Regression'] = lr_model
        self.model_performance['Linear Regression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test
        }
        
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Training RMSE: ₹{train_rmse:,.0f}")
        print(f"Testing RMSE: ₹{test_rmse:,.0f}")
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\nWhy Linear Regression?")
        print("- Simple and interpretable model")
        print("- Good baseline for comparison")
        print("- Fast training and prediction")
        print("- Works well when features have linear relationships with target")
        
        return lr_model
    
    def create_random_forest_model(self):
        """Create and train Random Forest model with hyperparameter tuning"""
        print("\n" + "="*50)
        print("RANDOM FOREST MODEL")
        print("="*50)
        
        # Create model with initial parameters
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_pred_train = best_rf.predict(self.X_train)
        y_pred_test = best_rf.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_rf, self.X_train, self.y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model and performance
        self.models['Random Forest'] = best_rf
        self.model_performance['Random Forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'feature_importance': feature_importance
        }
        
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Training RMSE: ₹{train_rmse:,.0f}")
        print(f"Testing RMSE: ₹{test_rmse:,.0f}")
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\nTop 5 Important Features:")
        for i, (feature, importance) in enumerate(feature_importance.head().values):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        print("\nWhy Random Forest?")
        print("- Handles non-linear relationships well")
        print("- Robust to outliers and noise")
        print("- Provides feature importance")
        print("- Reduces overfitting through ensemble learning")
        print("- Works well with mixed data types")
        
        return best_rf
    
    def create_gradient_boosting_model(self):
        """Create and train Gradient Boosting model with hyperparameter tuning"""
        print("\n" + "="*50)
        print("GRADIENT BOOSTING MODEL")
        print("="*50)
        
        # Create model with initial parameters
        gb_model = GradientBoostingRegressor(random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        best_gb = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_pred_train = best_gb.predict(self.X_train)
        y_pred_test = best_gb.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_gb, self.X_train, self.y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': best_gb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model and performance
        self.models['Gradient Boosting'] = best_gb
        self.model_performance['Gradient Boosting'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'feature_importance': feature_importance
        }
        
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Training RMSE: ₹{train_rmse:,.0f}")
        print(f"Testing RMSE: ₹{test_rmse:,.0f}")
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\nTop 5 Important Features:")
        for i, (feature, importance) in enumerate(feature_importance.head().values):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        print("\nWhy Gradient Boosting?")
        print("- Excellent predictive performance")
        print("- Handles complex non-linear patterns")
        print("- Sequential learning improves weak learners")
        print("- Good generalization capability")
        print("- Robust to different data distributions")
        
        return best_gb
    
    def compare_models(self):
        """Compare all models and select the best one"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, performance in self.model_performance.items():
            comparison_data.append({
                'Model': model_name,
                'Train R²': performance['train_r2'],
                'Test R²': performance['test_r2'],
                'Train RMSE': performance['train_rmse'],
                'Test RMSE': performance['test_rmse'],
                'Train MAE': performance['train_mae'],
                'Test MAE': performance['test_mae'],
                'CV R² Mean': performance['cv_mean'],
                'CV R² Std': performance['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Select best model based on test R²
        best_model_name = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Test R²: {self.model_performance[best_model_name]['test_r2']:.4f}")
        print(f"Test RMSE: ₹{self.model_performance[best_model_name]['test_rmse']:,.0f}")
        
        return comparison_df
    
    def visualize_results(self):
        """Create visualizations for model results"""
        print("\n" + "="*50)
        print("MODEL VISUALIZATION")
        print("="*50)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Laptop Price Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # Model performance comparison
        models = list(self.model_performance.keys())
        test_r2_scores = [self.model_performance[model]['test_r2'] for model in models]
        test_rmse_scores = [self.model_performance[model]['test_rmse'] for model in models]
        
        axes[0, 0].bar(models, test_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Test R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models, test_rmse_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Test RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE (₹)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Actual vs Predicted for best model
        best_predictions = self.model_performance[self.best_model_name]['predictions']
        axes[0, 2].scatter(self.y_test, best_predictions, alpha=0.6, color='green')
        axes[0, 2].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 2].set_title(f'Actual vs Predicted - {self.best_model_name}')
        axes[0, 2].set_xlabel('Actual Price (₹)')
        axes[0, 2].set_ylabel('Predicted Price (₹)')
        
        # Residuals plot for best model
        residuals = self.y_test - best_predictions
        axes[1, 0].scatter(best_predictions, residuals, alpha=0.6, color='purple')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title(f'Residuals Plot - {self.best_model_name}')
        axes[1, 0].set_xlabel('Predicted Price (₹)')
        axes[1, 0].set_ylabel('Residuals (₹)')
        
        # Feature importance for best model (if available)
        if 'feature_importance' in self.model_performance[self.best_model_name]:
            feature_imp = self.model_performance[self.best_model_name]['feature_importance'].head(10)
            axes[1, 1].barh(feature_imp['feature'], feature_imp['importance'], color='orange')
            axes[1, 1].set_title(f'Top 10 Feature Importance - {self.best_model_name}')
            axes[1, 1].set_xlabel('Importance')
        
        # Cross-validation scores
        cv_means = [self.model_performance[model]['cv_mean'] for model in models]
        cv_stds = [self.model_performance[model]['cv_std'] for model in models]
        axes[1, 2].bar(models, cv_means, yerr=cv_stds, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 2].set_title('Cross-Validation R² Scores')
        axes[1, 2].set_ylabel('CV R² Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, model_dir="models"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
        
        # Save scaler
        if self.scalers:
            scaler_filename = f"{model_dir}/scaler.pkl"
            joblib.dump(self.scalers['standard'], scaler_filename)
            print(f"Saved scaler to {scaler_filename}")
        
        # Save best model separately
        best_model_filename = f"{model_dir}/best_model.pkl"
        joblib.dump(self.best_model, best_model_filename)
        print(f"Saved best model ({self.best_model_name}) to {best_model_filename}")
    
    def train_all_models(self, X, y):
        """Train all models in the pipeline"""
        print("LAPTOP PRICE PREDICTION - MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Load and split data
        self.load_data(X, y)
        self.split_data()
        self.scale_features()
        
        # Train all models
        self.create_linear_regression_model()
        self.create_random_forest_model()
        self.create_gradient_boosting_model()
        
        # Compare and visualize
        comparison_df = self.compare_models()
        self.visualize_results()
        
        # Save models
        self.save_models()
        
        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return self.best_model, self.best_model_name, comparison_df

# Example usage
if __name__ == "__main__":
    # This would be called after data cleaning
    print("Model Design Module - Ready for training!")
    print("Use this module after running data_cleaning.py")
    print("\nExample usage:")
    print("from data_cleaning import LaptopDataCleaner")
    print("from model_design import LaptopPricePredictor")
    print("")
    print("# Clean data")
    print("cleaner = LaptopDataCleaner('data.csv')")
    print("X, y, cleaned_df = cleaner.run_complete_pipeline()")
    print("")
    print("# Train models")
    print("predictor = LaptopPricePredictor()")
    print("best_model, best_name, comparison = predictor.train_all_models(X, y)")