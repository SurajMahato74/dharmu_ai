import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelAnalysis:
    def __init__(self, model_path, scaler_path, data_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.data = pd.read_csv(data_path)
        
    def analyze_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.data.drop('Price', axis=1).columns
            importances = self.model.feature_importances_
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance Analysis')
            plt.tight_layout()
            plt.show()
            
            return dict(zip(feature_names, importances))
        return None
    
    def cross_validation_analysis(self, cv=5):
        X = self.data.drop('Price', axis=1)
        y = self.data['Price']
        X_scaled = self.scaler.transform(X)
        
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
        
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def learning_curve_analysis(self):
        X = self.data.drop('Price', axis=1)
        y = self.data['Price']
        X_scaled = self.scaler.transform(X)
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title('Learning Curve Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return train_sizes, train_scores, val_scores

if __name__ == "__main__":
    analyzer = AdvancedModelAnalysis(
        'models/best_model.pkl',
        'models/scaler.pkl', 
        'archive/data_cleaned.csv'
    )
    
    print("=== Advanced Model Analysis ===")
    analyzer.analyze_feature_importance()
    analyzer.cross_validation_analysis()
    analyzer.learning_curve_analysis()