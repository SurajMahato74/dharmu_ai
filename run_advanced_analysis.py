"""
ADVANCED LAPTOP PRICE PREDICTION ANALYSIS
=========================================

This script runs advanced model analysis with:
- Enhanced models with more epochs
- Detailed visualizations for each model
- Confusion matrices for regression
- Learning curves
- Comprehensive performance comparison
- Individual model analysis pages

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

# Import our modules
from data_cleaning import LaptopDataCleaner
from advanced_model_analysis import AdvancedModelAnalysis

def main():
    """Run advanced model analysis"""
    print("ADVANCED LAPTOP PRICE PREDICTION ANALYSIS")
    print("="*60)
    
    # Data path
    data_path = "c:/Users/suraj/OneDrive/Desktop/assignmen/dharmu_ai/archive/data.csv"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    start_time = datetime.now()
    
    try:
        # Step 1: Clean data (if not already done)
        print("\n🔄 Step 1: Data Cleaning...")
        cleaner = LaptopDataCleaner(data_path)
        X, y, cleaned_df = cleaner.run_complete_pipeline()
        
        # Step 2: Advanced Model Analysis
        print("\n🤖 Step 2: Advanced Model Analysis...")
        analyzer = AdvancedModelAnalysis()
        best_model, best_name, comparison_df = analyzer.run_complete_analysis(X, y)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("🎉 ADVANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {duration}")
        print(f"🏆 Best model: {best_name}")
        
        print("\n📊 ANALYSIS GENERATED:")
        print("- Model comparison charts")
        print("- Individual model analysis (2x2 plots for each model)")
        print("- Confusion matrices for price categories")
        print("- Learning curves")
        print("- Comprehensive performance metrics")
        
        print("\n🎯 WHAT YOU GOT:")
        print("1. Enhanced models with more epochs (500 estimators)")
        print("2. Detailed visualizations for each model")
        print("3. Regression confusion matrices (price categories)")
        print("4. Learning curves showing training progress")
        print("5. Comprehensive metric comparison")
        print("6. Feature importance analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        print("Please check the error details and try again.")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Review all the generated visualizations")
        print("2. Analyze model performance differences")
        print("3. Use insights for model selection")
        print("4. Run prediction interface: streamlit run prediction_interface.py")
    else:
        print("\n🔧 Check the error and try again")