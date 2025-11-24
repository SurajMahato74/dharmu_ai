import pandas as pd
import numpy as np
import joblib
from prediction_interface import LaptopPricePredictor
import warnings
warnings.filterwarnings('ignore')

class BudgetTester:
    def __init__(self):
        self.predictor = LaptopPricePredictor()
        
    def test_budget_ranges(self):
        """Test predictions across different budget ranges"""
        
        budget_scenarios = [
            {
                'name': 'Budget Laptop (₹30,000)',
                'config': {
                    'brand': 'HP',
                    'processor': 'Intel Core i3',
                    'ram': '8GB',
                    'storage': '256GB SSD',
                    'gpu': 'Intel Integrated',
                    'display_size': '15.6"',
                    'os': 'Windows 11'
                },
                'expected_range': (25000, 40000)
            },
            {
                'name': 'Mid-Range Laptop (₹60,000)',
                'config': {
                    'brand': 'Dell',
                    'processor': 'Intel Core i5',
                    'ram': '16GB',
                    'storage': '512GB SSD',
                    'gpu': 'Intel Integrated',
                    'display_size': '15.6"',
                    'os': 'Windows 11'
                },
                'expected_range': (50000, 75000)
            },
            {
                'name': 'Gaming Laptop (₹90,000)',
                'config': {
                    'brand': 'Asus',
                    'processor': 'Intel Core i7',
                    'ram': '16GB',
                    'storage': '1TB SSD',
                    'gpu': 'NVIDIA RTX 3060',
                    'display_size': '15.6"',
                    'os': 'Windows 11'
                },
                'expected_range': (80000, 120000)
            },
            {
                'name': 'Premium Laptop (₹150,000)',
                'config': {
                    'brand': 'Apple',
                    'processor': 'Apple M2',
                    'ram': '16GB',
                    'storage': '512GB SSD',
                    'gpu': 'Apple Integrated',
                    'display_size': '13.6"',
                    'os': 'macOS'
                },
                'expected_range': (120000, 180000)
            }
        ]
        
        print("🧪 Testing Budget Predictions")
        print("=" * 50)
        
        results = []
        
        for scenario in budget_scenarios:
            print(f"\n📱 {scenario['name']}")
            print("-" * 30)
            
            try:
                prediction = self.predictor.predict_price(scenario['config'])
                predicted_price = prediction['predicted_price']
                min_expected, max_expected = scenario['expected_range']
                
                # Check if prediction is within expected range
                is_reasonable = min_expected <= predicted_price <= max_expected
                
                print(f"Configuration: {scenario['config']}")
                print(f"Predicted Price: ₹{predicted_price:,}")
                print(f"Expected Range: ₹{min_expected:,} - ₹{max_expected:,}")
                print(f"Status: {'✅ PASS' if is_reasonable else '❌ FAIL'}")
                
                results.append({
                    'scenario': scenario['name'],
                    'predicted': predicted_price,
                    'expected_min': min_expected,
                    'expected_max': max_expected,
                    'reasonable': is_reasonable
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'scenario': scenario['name'],
                    'predicted': None,
                    'expected_min': min_expected,
                    'expected_max': max_expected,
                    'reasonable': False
                })
        
        # Summary
        print(f"\n📊 Test Summary")
        print("=" * 30)
        passed = sum(1 for r in results if r['reasonable'])
        total = len(results)
        print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        return results
    
    def test_feature_sensitivity(self):
        """Test how sensitive predictions are to feature changes"""
        
        base_config = {
            'brand': 'Dell',
            'processor': 'Intel Core i5',
            'ram': '8GB',
            'storage': '256GB SSD',
            'gpu': 'Intel Integrated',
            'display_size': '15.6"',
            'os': 'Windows 11'
        }
        
        print(f"\n🔬 Feature Sensitivity Analysis")
        print("=" * 40)
        
        base_prediction = self.predictor.predict_price(base_config)
        base_price = base_prediction['predicted_price']
        print(f"Base Configuration Price: ₹{base_price:,}")
        
        # Test RAM changes
        print(f"\n💾 RAM Impact:")
        for ram in ['4GB', '8GB', '16GB', '32GB']:
            config = base_config.copy()
            config['ram'] = ram
            try:
                pred = self.predictor.predict_price(config)
                price_diff = pred['predicted_price'] - base_price
                print(f"  {ram}: ₹{pred['predicted_price']:,} ({price_diff:+,})")
            except:
                print(f"  {ram}: Error")
        
        # Test Storage changes
        print(f"\n💿 Storage Impact:")
        for storage in ['256GB SSD', '512GB SSD', '1TB SSD', '2TB SSD']:
            config = base_config.copy()
            config['storage'] = storage
            try:
                pred = self.predictor.predict_price(config)
                price_diff = pred['predicted_price'] - base_price
                print(f"  {storage}: ₹{pred['predicted_price']:,} ({price_diff:+,})")
            except:
                print(f"  {storage}: Error")

if __name__ == "__main__":
    tester = BudgetTester()
    
    # Run budget range tests
    results = tester.test_budget_ranges()
    
    # Run feature sensitivity tests
    tester.test_feature_sensitivity()
    
    print(f"\n🎯 Testing completed!")