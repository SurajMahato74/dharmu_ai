import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelAnalysis:
    def __init__(self):
        """Initialize advanced model analysis"""
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
        
    def scale_features(self):
        """Scale features for models that require it"""
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers['standard'] = scaler
        
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calculate comprehensive regression metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Custom accuracy for regression (within 10% tolerance)
        tolerance = 0.1
        accurate_predictions = np.abs((y_true - y_pred) / y_true) <= tolerance
        accuracy = np.mean(accurate_predictions) * 100
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'accuracy_10pct': accuracy
        }
    
    def train_enhanced_models(self):
        """Train models with enhanced parameters"""
        print("Training Enhanced Models with More Epochs...")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        lr_pred_train = lr_model.predict(self.X_train_scaled)
        lr_pred_test = lr_model.predict(self.X_test_scaled)
        
        self.models['Linear Regression'] = lr_model
        self.model_performance['Linear Regression'] = {
            'train_metrics': self.calculate_regression_metrics(self.y_train, lr_pred_train),
            'test_metrics': self.calculate_regression_metrics(self.y_test, lr_pred_test),
            'train_predictions': lr_pred_train,
            'test_predictions': lr_pred_test,
            'cv_scores': cross_val_score(lr_model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        }
        
        # Random Forest with more estimators
        rf_model = RandomForestRegressor(
            n_estimators=500,  # Increased epochs
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        rf_pred_train = rf_model.predict(self.X_train)
        rf_pred_test = rf_model.predict(self.X_test)
        
        self.models['Random Forest'] = rf_model
        self.model_performance['Random Forest'] = {
            'train_metrics': self.calculate_regression_metrics(self.y_train, rf_pred_train),
            'test_metrics': self.calculate_regression_metrics(self.y_test, rf_pred_test),
            'train_predictions': rf_pred_train,
            'test_predictions': rf_pred_test,
            'cv_scores': cross_val_score(rf_model, self.X_train, self.y_train, cv=5, scoring='r2'),
            'feature_importance': pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # Gradient Boosting with more estimators
        gb_model = GradientBoostingRegressor(
            n_estimators=500,  # Increased epochs
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        gb_model.fit(self.X_train, self.y_train)
        gb_pred_train = gb_model.predict(self.X_train)
        gb_pred_test = gb_model.predict(self.X_test)
        
        self.models['Gradient Boosting'] = gb_model
        self.model_performance['Gradient Boosting'] = {
            'train_metrics': self.calculate_regression_metrics(self.y_train, gb_pred_train),
            'test_metrics': self.calculate_regression_metrics(self.y_test, gb_pred_test),
            'train_predictions': gb_pred_train,
            'test_predictions': gb_pred_test,
            'cv_scores': cross_val_score(gb_model, self.X_train, self.y_train, cv=5, scoring='r2'),
            'feature_importance': pd.DataFrame({
                'feature': self.X.columns,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # Select best model
        best_r2 = 0
        for model_name, performance in self.model_performance.items():
            test_r2 = performance['test_metrics']['r2']
            if test_r2 > best_r2:
                best_r2 = test_r2
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
    
    def create_confusion_matrix_regression(self, y_true, y_pred, model_name):
        """Create confusion matrix for regression by binning predictions"""
        # Create price bins
        bins = [0, 30000, 60000, 100000, float('inf')]
        labels = ['Budget', 'Mid-range', 'Premium', 'High-end']
        
        y_true_binned = pd.cut(y_true, bins=bins, labels=labels)
        y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels)
        
        # Create confusion matrix
        confusion_matrix = pd.crosstab(y_true_binned, y_pred_binned, margins=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Price Category Confusion Matrix - {model_name}')
        plt.ylabel('Actual Price Category')
        plt.xlabel('Predicted Price Category')
        plt.tight_layout()
        plt.show()
        
        return confusion_matrix
    
    def plot_model_comparison(self):
        """Create comprehensive model comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
        
        models = list(self.model_performance.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. R² Score Comparison
        r2_scores = [self.model_performance[model]['test_metrics']['r2'] for model in models]
        bars1 = axes[0, 0].bar(models, r2_scores, color=colors)
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
        
        # 2. RMSE Comparison
        rmse_scores = [self.model_performance[model]['test_metrics']['rmse'] for model in models]
        bars2 = axes[0, 1].bar(models, rmse_scores, color=colors)
        axes[0, 1].set_title('RMSE Comparison (Lower is Better)')
        axes[0, 1].set_ylabel('RMSE (₹)')
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1000,
                           f'₹{height:,.0f}', ha='center', va='bottom')
        
        # 3. MAE Comparison
        mae_scores = [self.model_performance[model]['test_metrics']['mae'] for model in models]
        bars3 = axes[0, 2].bar(models, mae_scores, color=colors)
        axes[0, 2].set_title('MAE Comparison (Lower is Better)')
        axes[0, 2].set_ylabel('MAE (₹)')
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 500,
                           f'₹{height:,.0f}', ha='center', va='bottom')
        
        # 4. MAPE Comparison
        mape_scores = [self.model_performance[model]['test_metrics']['mape'] for model in models]
        bars4 = axes[1, 0].bar(models, mape_scores, color=colors)
        axes[1, 0].set_title('MAPE Comparison (Lower is Better)')
        axes[1, 0].set_ylabel('MAPE (%)')
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}%', ha='center', va='bottom')
        
        # 5. Accuracy (10% tolerance) Comparison
        accuracy_scores = [self.model_performance[model]['test_metrics']['accuracy_10pct'] for model in models]
        bars5 = axes[1, 1].bar(models, accuracy_scores, color=colors)
        axes[1, 1].set_title('Accuracy within 10% Tolerance')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_ylim(0, 100)
        for i, bar in enumerate(bars5):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom')
        
        # 6. Cross-Validation Scores
        cv_means = [np.mean(self.model_performance[model]['cv_scores']) for model in models]
        cv_stds = [np.std(self.model_performance[model]['cv_scores']) for model in models]
        bars6 = axes[1, 2].bar(models, cv_means, yerr=cv_stds, capsize=5, color=colors)
        axes[1, 2].set_title('Cross-Validation R² Scores')
        axes[1, 2].set_ylabel('CV R² Score')
        for i, bar in enumerate(bars6):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_individual_model_analysis(self):
        """Create detailed analysis for each model"""
        for model_name in self.models.keys():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{model_name} - Detailed Analysis', fontsize=16, fontweight='bold')
            
            y_pred = self.model_performance[model_name]['test_predictions']
            
            # 1. Actual vs Predicted
            axes[0, 0].scatter(self.y_test, y_pred, alpha=0.6, color='blue')
            axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Price (₹)')
            axes[0, 0].set_ylabel('Predicted Price (₹)')
            axes[0, 0].set_title('Actual vs Predicted Prices')
            
            # Add R² score to the plot
            r2 = self.model_performance[model_name]['test_metrics']['r2']
            axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 2. Residuals Plot
            residuals = self.y_test - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Price (₹)')
            axes[0, 1].set_ylabel('Residuals (₹)')
            axes[0, 1].set_title('Residuals Plot')
            
            # 3. Error Distribution
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('Residuals (₹)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Error Distribution')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            
            # 4. Feature Importance (if available)
            if 'feature_importance' in self.model_performance[model_name]:
                feature_imp = self.model_performance[model_name]['feature_importance'].head(10)
                axes[1, 1].barh(feature_imp['feature'], feature_imp['importance'], color='purple')
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 10 Feature Importance')
            else:
                # For Linear Regression, show coefficients
                if hasattr(self.models[model_name], 'coef_'):
                    coef_df = pd.DataFrame({
                        'feature': self.X.columns,
                        'coefficient': np.abs(self.models[model_name].coef_)
                    }).sort_values('coefficient', ascending=False).head(10)
                    axes[1, 1].barh(coef_df['feature'], coef_df['coefficient'], color='purple')
                    axes[1, 1].set_xlabel('|Coefficient|')
                    axes[1, 1].set_title('Top 10 Feature Coefficients')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Feature Analysis')
            
            plt.tight_layout()
            plt.show()
            
            # Create confusion matrix for this model
            self.create_confusion_matrix_regression(self.y_test, y_pred, model_name)
    
    def plot_learning_curves(self):
        """Plot learning curves for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Learning Curves - Model Performance vs Training Size', fontsize=16)
        
        models_data = [
            ('Linear Regression', self.models['Linear Regression'], self.X_train_scaled),
            ('Random Forest', self.models['Random Forest'], self.X_train),
            ('Gradient Boosting', self.models['Gradient Boosting'], self.X_train)
        ]
        
        for i, (name, model, X_data) in enumerate(models_data):
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_data, self.y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            axes[i].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            axes[i].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            axes[i].set_xlabel('Training Set Size')
            axes[i].set_ylabel('R² Score')
            axes[i].set_title(f'{name}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, performance in self.model_performance.items():
            test_metrics = performance['test_metrics']
            cv_scores = performance['cv_scores']
            
            comparison_data.append({
                'Model': model_name,
                'R² Score': test_metrics['r2'],
                'RMSE (₹)': test_metrics['rmse'],
                'MAE (₹)': test_metrics['mae'],
                'MAPE (%)': test_metrics['mape'],
                'Accuracy (10%)': test_metrics['accuracy_10pct'],
                'CV Mean': np.mean(cv_scores),
                'CV Std': np.std(cv_scores)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(comparison_df.round(4))
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        best_metrics = self.model_performance[self.best_model_name]['test_metrics']
        print(f"   R² Score: {best_metrics['r2']:.4f}")
        print(f"   RMSE: ₹{best_metrics['rmse']:,.0f}")
        print(f"   MAE: ₹{best_metrics['mae']:,.0f}")
        print(f"   MAPE: {best_metrics['mape']:.2f}%")
        print(f"   Accuracy (10% tolerance): {best_metrics['accuracy_10pct']:.1f}%")
        
        return comparison_df
    
    def run_complete_analysis(self, X, y):
        """Run complete advanced analysis"""
        print("ADVANCED MODEL ANALYSIS WITH ENHANCED EPOCHS")
        print("="*60)
        
        # Load and prepare data
        self.load_data(X, y)
        self.split_data()
        self.scale_features()
        
        # Train enhanced models
        self.train_enhanced_models()
        
        # Generate comprehensive visualizations
        print("\nGenerating comprehensive visualizations...")
        
        # 1. Model comparison chart
        self.plot_model_comparison()
        
        # 2. Individual model analysis
        self.plot_individual_model_analysis()
        
        # 3. Learning curves
        self.plot_learning_curves()
        
        # 4. Performance report
        comparison_df = self.generate_performance_report()
        
        # Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("ADVANCED ANALYSIS COMPLETED!")
        print("="*60)
        
        return self.best_model, self.best_model_name, comparison_df
    
    def save_models(self):
        """Save all trained models"""
        import os
        os.makedirs("models", exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"models/{model_name.lower().replace(' ', '_')}_enhanced.pkl"
            joblib.dump(model, filename)
            print(f"Saved enhanced {model_name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scalers['standard'], "models/scaler_enhanced.pkl")
        
        # Save best model
        joblib.dump(self.best_model, "models/best_model_enhanced.pkl")
        print(f"Saved best enhanced model ({self.best_model_name})")

# Example usage
if __name__ == "__main__":
    print("Advanced Model Analysis Module Ready!")
    print("Use this after data cleaning:")
    print("analyzer = AdvancedModelAnalysis()")
    print("analyzer.run_complete_analysis(X, y)")