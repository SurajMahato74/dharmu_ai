#!/usr/bin/env python3
"""
Advanced Analysis Runner
Executes comprehensive analysis of the laptop price prediction models
"""

import sys
import os
from advanced_model_analysis import AdvancedModelAnalysis
from analyze_data import DataAnalyzer

def run_complete_analysis():
    print("🚀 Starting Advanced Model Analysis Pipeline")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'models/best_model.pkl',
        'models/scaler.pkl',
        'archive/data_cleaned.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run the main pipeline first: python main_pipeline.py")
        return False
    
    try:
        # 1. Data Analysis
        print("\n📊 Phase 1: Data Analysis")
        print("-" * 30)
        data_analyzer = DataAnalyzer('archive/data_cleaned.csv')
        data_analyzer.basic_stats()
        data_analyzer.price_distribution()
        data_analyzer.brand_analysis()
        data_analyzer.correlation_analysis()
        
        # 2. Model Analysis
        print("\n🤖 Phase 2: Model Analysis")
        print("-" * 30)
        model_analyzer = AdvancedModelAnalysis(
            'models/best_model.pkl',
            'models/scaler.pkl',
            'archive/data_cleaned.csv'
        )
        
        # Feature importance
        print("\n🔍 Analyzing Feature Importance...")
        importance_scores = model_analyzer.analyze_feature_importance()
        if importance_scores:
            print("Top 5 Most Important Features:")
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features[:5]:
                print(f"  {feature}: {score:.4f}")
        
        # Cross-validation
        print("\n📈 Running Cross-Validation Analysis...")
        cv_scores = model_analyzer.cross_validation_analysis()
        
        # Learning curves
        print("\n📚 Generating Learning Curves...")
        model_analyzer.learning_curve_analysis()
        
        print("\n✅ Advanced analysis completed successfully!")
        print("Check the generated plots for detailed insights.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Advanced Analysis Runner")
        print("Usage: python run_advanced_analysis.py")
        print("This script runs comprehensive analysis on the trained models.")
        return
    
    success = run_complete_analysis()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()