import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        
    def basic_stats(self):
        print("=== Dataset Overview ===")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print("\n=== Price Statistics ===")
        print(self.data['Price'].describe())
        
    def price_distribution(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.data['Price'], bins=50, alpha=0.7)
        plt.title('Price Distribution')
        plt.xlabel('Price (₹)')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(self.data['Price'])
        plt.title('Price Box Plot')
        plt.ylabel('Price (₹)')
        
        plt.tight_layout()
        plt.show()
        
    def brand_analysis(self):
        if 'brand' in self.data.columns:
            plt.figure(figsize=(12, 6))
            brand_prices = self.data.groupby('brand')['Price'].agg(['mean', 'count'])
            brand_prices = brand_prices.sort_values('mean', ascending=False)
            
            plt.subplot(1, 2, 1)
            brand_prices['mean'].plot(kind='bar')
            plt.title('Average Price by Brand')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            brand_prices['count'].plot(kind='bar')
            plt.title('Count by Brand')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
    def correlation_analysis(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    analyzer = DataAnalyzer('archive/data_cleaned.csv')
    analyzer.basic_stats()
    analyzer.price_distribution()
    analyzer.brand_analysis()
    analyzer.correlation_analysis()