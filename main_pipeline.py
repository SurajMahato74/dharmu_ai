"""
LAPTOP PRICE PREDICTION - COMPLETE MACHINE LEARNING PIPELINE
============================================================

This script runs the complete pipeline for laptop price prediction:
1. Data Cleaning and Preprocessing
2. Model Training with Multiple Algorithms
3. Model Comparison and Selection
4. Prediction Interface Setup

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_cleaning import LaptopDataCleaner
from model_design import LaptopPricePredictor
from prediction_interface import LaptopPricePredictionInterface

class LaptopPricePredictionPipeline:
    def __init__(self, data_path):
        """Initialize the complete pipeline"""
        self.data_path = data_path
        self.cleaner = None
        self.predictor = None
        self.interface = None
        self.results = {}
        
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "="*80)
        print(f"{title:^80}")
        print("="*80)
    
    def step1_data_cleaning(self):
        """Step 1: Clean and preprocess the data"""
        self.print_header("STEP 1: DATA CLEANING AND PREPROCESSING")
        
        print("🔄 Starting data cleaning process...")
        
        # Initialize data cleaner
        self.cleaner = LaptopDataCleaner(self.data_path)
        
        # Run complete cleaning pipeline
        X, y, cleaned_df = self.cleaner.run_complete_pipeline()
        
        # Store results
        self.results['cleaned_data'] = {
            'X': X,
            'y': y,
            'cleaned_df': cleaned_df,
            'original_shape': self.cleaner.df.shape,
            'cleaned_shape': cleaned_df.shape,
            'features_count': X.shape[1]
        }
        
        print(f"✅ Data cleaning completed successfully!")
        print(f"   Original dataset: {self.cleaner.df.shape}")
        print(f"   Cleaned dataset: {cleaned_df.shape}")
        print(f"   Features for ML: {X.shape[1]}")
        
        return X, y, cleaned_df
    
    def step2_model_training(self, X, y):
        """Step 2: Train multiple models and compare performance"""
        self.print_header("STEP 2: MODEL TRAINING AND COMPARISON")
        
        print("🤖 Starting model training process...")
        print("   Training 3 different algorithms:")
        print("   1. Linear Regression (Baseline)")
        print("   2. Random Forest (Ensemble)")
        print("   3. Gradient Boosting (Advanced Ensemble)")
        
        # Initialize predictor
        self.predictor = LaptopPricePredictor()
        
        # Train all models
        best_model, best_model_name, comparison_df = self.predictor.train_all_models(X, y)
        
        # Store results
        self.results['model_training'] = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'comparison_df': comparison_df,
            'all_models': self.predictor.models,
            'model_performance': self.predictor.model_performance
        }
        
        print(f"✅ Model training completed successfully!")
        print(f"   Best model: {best_model_name}")
        print(f"   Best R² score: {self.predictor.model_performance[best_model_name]['test_r2']:.4f}")
        print(f"   Best RMSE: ₹{self.predictor.model_performance[best_model_name]['test_rmse']:,.0f}")
        
        return best_model, best_model_name, comparison_df
    
    def step3_model_analysis(self):
        """Step 3: Analyze model performance and provide insights"""
        self.print_header("STEP 3: MODEL PERFORMANCE ANALYSIS")
        
        print("📊 Analyzing model performance...")
        
        # Get performance data
        performance = self.results['model_training']['model_performance']
        comparison_df = self.results['model_training']['comparison_df']
        
        print("\n🏆 MODEL RANKING:")
        print("-" * 50)
        
        # Sort models by test R² score
        sorted_models = sorted(performance.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        for i, (model_name, perf) in enumerate(sorted_models, 1):
            print(f"{i}. {model_name}")
            print(f"   Test R²: {perf['test_r2']:.4f}")
            print(f"   Test RMSE: ₹{perf['test_rmse']:,.0f}")
            print(f"   Cross-validation: {perf['cv_mean']:.4f} ± {perf['cv_std']:.4f}")
            print()
        
        # Model insights
        best_model_name = self.results['model_training']['best_model_name']
        best_perf = performance[best_model_name]
        
        print("🎯 KEY INSIGHTS:")
        print("-" * 50)
        
        if best_perf['test_r2'] > 0.85:
            print("✅ Excellent model performance! The model explains >85% of price variance.")
        elif best_perf['test_r2'] > 0.75:
            print("✅ Good model performance! The model explains >75% of price variance.")
        else:
            print("⚠️  Model performance could be improved. Consider feature engineering.")
        
        # Check for overfitting
        train_test_diff = best_perf['train_r2'] - best_perf['test_r2']
        if train_test_diff > 0.1:
            print("⚠️  Potential overfitting detected (train-test R² difference > 0.1)")
        else:
            print("✅ No significant overfitting detected.")
        
        # RMSE interpretation
        avg_price = self.results['cleaned_data']['y'].mean()
        rmse_percentage = (best_perf['test_rmse'] / avg_price) * 100
        print(f"📈 Average prediction error: ₹{best_perf['test_rmse']:,.0f} ({rmse_percentage:.1f}% of average price)")
        
        return comparison_df
    
    def step4_feature_importance_analysis(self):
        """Step 4: Analyze feature importance"""
        self.print_header("STEP 4: FEATURE IMPORTANCE ANALYSIS")
        
        print("🔍 Analyzing feature importance...")
        
        best_model_name = self.results['model_training']['best_model_name']
        performance = self.results['model_training']['model_performance']
        
        if 'feature_importance' in performance[best_model_name]:
            feature_imp = performance[best_model_name]['feature_importance']
            
            print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES ({best_model_name}):")
            print("-" * 60)
            
            for i, (feature, importance) in enumerate(feature_imp.head(10).values, 1):
                print(f"{i:2d}. {feature:<25} {importance:.4f}")
            
            # Feature insights
            print("\n💡 FEATURE INSIGHTS:")
            print("-" * 50)
            
            top_feature = feature_imp.iloc[0]['feature']
            top_importance = feature_imp.iloc[0]['importance']
            
            print(f"🥇 Most important feature: {top_feature} ({top_importance:.4f})")
            
            # Check if performance score is important
            if 'performance_score' in feature_imp['feature'].values:
                perf_rank = feature_imp[feature_imp['feature'] == 'performance_score'].index[0] + 1
                print(f"📊 Performance score ranking: #{perf_rank}")
            
            # Check gaming laptop indicator
            if 'is_gaming' in feature_imp['feature'].values:
                gaming_rank = feature_imp[feature_imp['feature'] == 'is_gaming'].index[0] + 1
                print(f"🎮 Gaming indicator ranking: #{gaming_rank}")
        
        else:
            print("ℹ️  Feature importance not available for this model type.")
    
    def step5_setup_prediction_interface(self):
        """Step 5: Setup prediction interface"""
        self.print_header("STEP 5: PREDICTION INTERFACE SETUP")
        
        print("🖥️  Setting up prediction interface...")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Check if models are saved
        model_files = [
            "models/best_model.pkl",
            "models/scaler.pkl",
            "models/linear_regression_model.pkl",
            "models/random_forest_model.pkl",
            "models/gradient_boosting_model.pkl"
        ]
        
        saved_files = [f for f in model_files if os.path.exists(f)]
        
        print(f"✅ Model files saved: {len(saved_files)}/{len(model_files)}")
        for file in saved_files:
            print(f"   📁 {file}")
        
        # Initialize interface
        self.interface = LaptopPricePredictionInterface()
        
        print("\n🚀 PREDICTION INTERFACE READY!")
        print("-" * 50)
        print("To run the interactive interface, execute:")
        print("   streamlit run prediction_interface.py")
        print("\nOr use the interface programmatically:")
        print("   interface = LaptopPricePredictionInterface()")
        print("   interface.main_interface()")
    
    def generate_report(self):
        """Generate comprehensive report"""
        self.print_header("PIPELINE EXECUTION REPORT")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"📅 Report Generated: {timestamp}")
        print(f"📊 Dataset: {os.path.basename(self.data_path)}")
        
        # Data summary
        print("\n📈 DATA SUMMARY:")
        print("-" * 40)
        original_shape = self.results['cleaned_data']['original_shape']
        cleaned_shape = self.results['cleaned_data']['cleaned_shape']
        features_count = self.results['cleaned_data']['features_count']
        
        print(f"Original dataset: {original_shape[0]} samples, {original_shape[1]} columns")
        print(f"Cleaned dataset: {cleaned_shape[0]} samples, {cleaned_shape[1]} columns")
        print(f"ML features: {features_count}")
        
        # Model performance summary
        print("\n🤖 MODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        performance = self.results['model_training']['model_performance']
        best_model_name = self.results['model_training']['best_model_name']
        
        for model_name, perf in performance.items():
            status = "🏆 BEST" if model_name == best_model_name else "  "
            print(f"{status} {model_name}:")
            print(f"     R²: {perf['test_r2']:.4f} | RMSE: ₹{perf['test_rmse']:,.0f}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        best_r2 = performance[best_model_name]['test_r2']
        
        if best_r2 > 0.9:
            print("✅ Excellent model! Ready for production deployment.")
        elif best_r2 > 0.8:
            print("✅ Good model performance. Consider minor improvements.")
            print("   • Collect more data for edge cases")
            print("   • Fine-tune hyperparameters further")
        else:
            print("⚠️  Model needs improvement before deployment:")
            print("   • Collect more training data")
            print("   • Engineer additional features")
            print("   • Try advanced algorithms (XGBoost, Neural Networks)")
        
        print("\n🎯 NEXT STEPS:")
        print("-" * 40)
        print("1. Test the prediction interface with various configurations")
        print("2. Validate predictions with real market data")
        print("3. Deploy the model to production environment")
        print("4. Set up monitoring for model performance")
        print("5. Plan for regular model retraining")
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        start_time = datetime.now()
        
        self.print_header("LAPTOP PRICE PREDICTION - COMPLETE ML PIPELINE")
        print(f"🚀 Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Data source: {self.data_path}")
        
        try:
            # Step 1: Data Cleaning
            X, y, cleaned_df = self.step1_data_cleaning()
            
            # Step 2: Model Training
            best_model, best_model_name, comparison_df = self.step2_model_training(X, y)
            
            # Step 3: Model Analysis
            self.step3_model_analysis()
            
            # Step 4: Feature Importance
            self.step4_feature_importance_analysis()
            
            # Step 5: Setup Interface
            self.step5_setup_prediction_interface()
            
            # Generate Report
            self.generate_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.print_header("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            print(f"⏱️  Total execution time: {duration}")
            print(f"🏆 Best model: {best_model_name}")
            print(f"📊 Best R² score: {self.results['model_training']['model_performance'][best_model_name]['test_r2']:.4f}")
            print("\n🚀 Ready to make predictions! Run: streamlit run prediction_interface.py")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            print("Please check the error details and try again.")
            return False

def main():
    """Main execution function"""
    print("LAPTOP PRICE PREDICTION SYSTEM")
    print("=" * 50)
    
    # Data path
    data_path = "c:/Users/suraj/OneDrive/Desktop/assignmen/dharmu_ai/archive/data.csv"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists and try again.")
        return
    
    # Initialize and run pipeline
    pipeline = LaptopPricePredictionPipeline(data_path)
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎯 WHAT'S NEXT?")
        print("-" * 30)
        print("1. Run the prediction interface:")
        print("   streamlit run prediction_interface.py")
        print("\n2. Test different laptop configurations")
        print("3. Analyze price predictions and recommendations")
        print("4. Use the model for market analysis")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("-" * 30)
        print("1. Check data file path and format")
        print("2. Ensure all required packages are installed")
        print("3. Check for sufficient memory and disk space")

if __name__ == "__main__":
    main()