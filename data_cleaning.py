import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LaptopDataCleaner:
    def __init__(self, data_path):
        """Initialize the data cleaner with dataset path"""
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Explore the dataset structure and basic statistics"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print("\nPrice Statistics:")
        print(f"Min Price: ₹{self.df['price'].min():,.0f}")
        print(f"Max Price: ₹{self.df['price'].max():,.0f}")
        print(f"Mean Price: ₹{self.df['price'].mean():,.0f}")
        print(f"Median Price: ₹{self.df['price'].median():,.0f}")
        
        return self.df.info()
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Create a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # Remove unnecessary columns
        columns_to_drop = ['Unnamed: 0', 'name']
        self.cleaned_df = self.cleaned_df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
        
        # Handle missing values
        print("\nHandling missing values...")
        self.cleaned_df['spec_rating'].fillna(self.cleaned_df['spec_rating'].median(), inplace=True)
        self.cleaned_df['resolution_width'].fillna(self.cleaned_df['resolution_width'].median(), inplace=True)
        self.cleaned_df['resolution_height'].fillna(self.cleaned_df['resolution_height'].median(), inplace=True)
        
        # Clean RAM column - extract numeric value
        print("Cleaning RAM column...")
        self.cleaned_df['Ram_numeric'] = self.cleaned_df['Ram'].str.extract('(\d+)').astype(float)
        
        # Clean ROM column - extract numeric value
        print("Cleaning ROM column...")
        self.cleaned_df['ROM_numeric'] = self.cleaned_df['ROM'].str.extract('(\d+)').astype(float)
        
        # Clean display size
        print("Cleaning display size...")
        self.cleaned_df['display_size_numeric'] = pd.to_numeric(self.cleaned_df['display_size'], errors='coerce')
        
        # Create price categories
        print("Creating price categories...")
        self.cleaned_df['price_category'] = pd.cut(self.cleaned_df['price'], 
                                                  bins=[0, 30000, 60000, 100000, float('inf')],
                                                  labels=['Budget', 'Mid-range', 'Premium', 'High-end'])
        
        # Extract processor generation
        print("Extracting processor information...")
        self.cleaned_df['processor_gen'] = self.cleaned_df['processor'].str.extract('(\d+)th Gen|(\d+)nd Gen|(\d+)rd Gen|(\d+)st Gen').fillna(method='ffill', axis=1).iloc[:, 0]
        self.cleaned_df['processor_gen'] = pd.to_numeric(self.cleaned_df['processor_gen'], errors='coerce')
        
        # Clean brand names
        print("Cleaning brand names...")
        self.cleaned_df['brand'] = self.cleaned_df['brand'].str.strip().str.title()
        
        print(f"Cleaned dataset shape: {self.cleaned_df.shape}")
        return self.cleaned_df
    
    def encode_categorical_features(self):
        """Encode categorical features for machine learning"""
        print("\n" + "="*50)
        print("FEATURE ENCODING")
        print("="*50)
        
        # Initialize label encoders
        label_encoders = {}
        categorical_columns = ['brand', 'processor', 'CPU', 'Ram_type', 'ROM_type', 'GPU', 'OS']
        
        for col in categorical_columns:
            if col in self.cleaned_df.columns:
                le = LabelEncoder()
                self.cleaned_df[f'{col}_encoded'] = le.fit_transform(self.cleaned_df[col].astype(str))
                label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return label_encoders
    
    def create_features(self):
        """Create additional features for better prediction"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # Screen resolution ratio
        self.cleaned_df['aspect_ratio'] = self.cleaned_df['resolution_width'] / self.cleaned_df['resolution_height']
        
        # Screen area (approximate)
        self.cleaned_df['screen_area'] = (self.cleaned_df['display_size_numeric'] ** 2) * 0.5
        
        # RAM to ROM ratio
        self.cleaned_df['ram_rom_ratio'] = self.cleaned_df['Ram_numeric'] / self.cleaned_df['ROM_numeric']
        
        # Performance score (combination of RAM, processor gen, and spec rating)
        self.cleaned_df['performance_score'] = (
            self.cleaned_df['Ram_numeric'] * 0.3 + 
            self.cleaned_df['processor_gen'].fillna(10) * 0.4 + 
            self.cleaned_df['spec_rating'] * 0.3
        )
        
        # Gaming laptop indicator (based on GPU)
        gaming_keywords = ['RTX', 'GTX', 'Radeon RX', 'GeForce']
        self.cleaned_df['is_gaming'] = self.cleaned_df['GPU'].str.contains('|'.join(gaming_keywords), case=False, na=False).astype(int)
        
        print("Created new features:")
        print("- aspect_ratio")
        print("- screen_area") 
        print("- ram_rom_ratio")
        print("- performance_score")
        print("- is_gaming")
        
        return self.cleaned_df
    
    def visualize_data(self):
        """Create visualizations for data analysis"""
        print("\n" + "="*50)
        print("DATA VISUALIZATION")
        print("="*50)
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Laptop Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Price distribution
        axes[0, 0].hist(self.cleaned_df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price (₹)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Brand distribution
        brand_counts = self.cleaned_df['brand'].value_counts().head(10)
        axes[0, 1].bar(brand_counts.index, brand_counts.values, color='lightcoral')
        axes[0, 1].set_title('Top 10 Brands')
        axes[0, 1].set_xlabel('Brand')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # RAM vs Price
        axes[0, 2].scatter(self.cleaned_df['Ram_numeric'], self.cleaned_df['price'], alpha=0.6, color='green')
        axes[0, 2].set_title('RAM vs Price')
        axes[0, 2].set_xlabel('RAM (GB)')
        axes[0, 2].set_ylabel('Price (₹)')
        
        # Price by category
        self.cleaned_df.boxplot(column='price', by='price_category', ax=axes[1, 0])
        axes[1, 0].set_title('Price by Category')
        axes[1, 0].set_xlabel('Price Category')
        axes[1, 0].set_ylabel('Price (₹)')
        
        # Spec rating distribution
        axes[1, 1].hist(self.cleaned_df['spec_rating'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Spec Rating Distribution')
        axes[1, 1].set_xlabel('Spec Rating')
        axes[1, 1].set_ylabel('Frequency')
        
        # Gaming vs Non-gaming price comparison
        gaming_data = [
            self.cleaned_df[self.cleaned_df['is_gaming'] == 1]['price'],
            self.cleaned_df[self.cleaned_df['is_gaming'] == 0]['price']
        ]
        axes[1, 2].boxplot(gaming_data, labels=['Gaming', 'Non-Gaming'])
        axes[1, 2].set_title('Gaming vs Non-Gaming Laptops Price')
        axes[1, 2].set_ylabel('Price (₹)')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        numeric_cols = ['price', 'spec_rating', 'Ram_numeric', 'ROM_numeric', 'display_size_numeric', 
                       'resolution_width', 'resolution_height', 'warranty', 'performance_score']
        correlation_matrix = self.cleaned_df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def prepare_ml_data(self):
        """Prepare final dataset for machine learning"""
        print("\n" + "="*50)
        print("PREPARING ML DATASET")
        print("="*50)
        
        # Select features for ML
        ml_features = [
            'brand_encoded', 'spec_rating', 'processor_encoded', 'CPU_encoded',
            'Ram_numeric', 'Ram_type_encoded', 'ROM_numeric', 'ROM_type_encoded',
            'GPU_encoded', 'display_size_numeric', 'resolution_width', 'resolution_height',
            'OS_encoded', 'warranty', 'aspect_ratio', 'screen_area', 'ram_rom_ratio',
            'performance_score', 'is_gaming', 'processor_gen'
        ]
        
        # Filter available features
        available_features = [col for col in ml_features if col in self.cleaned_df.columns]
        
        # Create ML dataset
        X = self.cleaned_df[available_features].copy()
        y = self.cleaned_df['price'].copy()
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        print(f"ML Dataset prepared:")
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target variable: price")
        
        return X, y
    
    def save_cleaned_data(self, output_path):
        """Save cleaned dataset"""
        self.cleaned_df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to: {output_path}")
    
    def run_complete_pipeline(self):
        """Run the complete data cleaning pipeline"""
        print("LAPTOP PRICE PREDICTION - DATA CLEANING PIPELINE")
        print("="*60)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Clean and process data
        self.clean_data()
        self.encode_categorical_features()
        self.create_features()
        
        # Visualize data
        self.visualize_data()
        
        # Prepare ML data
        X, y = self.prepare_ml_data()
        
        # Save cleaned data
        output_path = self.data_path.replace('.csv', '_cleaned.csv')
        self.save_cleaned_data(output_path)
        
        print("\n" + "="*60)
        print("DATA CLEANING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return X, y, self.cleaned_df

# Example usage
if __name__ == "__main__":
    # Initialize data cleaner
    data_path = "c:/Users/suraj/OneDrive/Desktop/assignmen/dharmu_ai/archive/data.csv"
    cleaner = LaptopDataCleaner(data_path)
    
    # Run complete pipeline
    X, y, cleaned_df = cleaner.run_complete_pipeline()
    
    print(f"\nFinal dataset ready for machine learning:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")