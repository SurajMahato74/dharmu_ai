#!/usr/bin/env python3
"""
Comprehensive Laptop Dataset Cleaning Script
This script performs thorough data cleaning and preprocessing for the laptop price prediction dataset.
Author: Kilo Code Assistant
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
import warnings
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning_debug.log'),
        logging.StreamHandler()
    ]
)

class LaptopDataCleaner:
    def __init__(self, data_path: str):
        """
        Initialize the data cleaner with the dataset path.
        
        Args:
            data_path (str): Path to the CSV file containing laptop data
        """
        self.data_path = Path(data_path)
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_report = {
            'total_rows': 0,
            'total_columns': 0,
            'issues_found': [],
            'actions_taken': [],
            'missing_values_before': {},
            'missing_values_after': {},
            'duplicates_found': 0,
            'outliers_detected': {},
            'data_type_conversions': [],
            'cleaned_rows': 0
        }
        warnings.filterwarnings('ignore')
        
    def load_data(self) -> bool:
        """
        Load the dataset from CSV file.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            logging.info(f"Loading data from: {self.data_path}")
            self.original_data = pd.read_csv(self.data_path)
            self.cleaned_data = self.original_data.copy()
            self.cleaning_report['total_rows'], self.cleaning_report['total_columns'] = self.original_data.shape
            logging.info(f"Data loaded successfully. Shape: {self.original_data.shape}")
            return True
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            return False
    
    def analyze_data_structure(self) -> Dict[str, Any]:
        """
        Analyze the basic structure of the dataset.
        
        Returns:
            Dict containing data structure information
        """
        logging.info("Analyzing data structure...")
        
        analysis = {
            'columns': list(self.cleaned_data.columns),
            'data_types': self.cleaned_data.dtypes.to_dict(),
            'memory_usage': self.cleaned_data.memory_usage(deep=True).sum(),
            'shape': self.cleaned_data.shape,
            'null_counts': self.cleaned_data.isnull().sum().to_dict(),
            'unique_values_per_column': {col: self.cleaned_data[col].nunique() for col in self.cleaned_data.columns}
        }
        
        # Store missing values for report
        self.cleaning_report['missing_values_before'] = analysis['null_counts']
        
        logging.info("Data structure analysis completed.")
        return analysis
    
    def detect_and_report_issues(self) -> None:
        """
        Detect and report various data quality issues.
        """
        logging.info("Detecting data quality issues...")
        
        issues = []
        
        # Check for missing values
        missing_cols = [col for col, count in self.cleaning_report['missing_values_before'].items() if count > 0]
        if missing_cols:
            issues.append(f"Missing values found in columns: {missing_cols}")
        
        # Check for duplicates
        duplicates = self.cleaned_data.duplicated().sum()
        self.cleaning_report['duplicates_found'] = duplicates
        if duplicates > 0:
            issues.append(f"Duplicate rows found: {duplicates}")
        
        # Check for inconsistent data formats
        text_columns = self.cleaned_data.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col in self.cleaned_data.columns:
                # Check for extra spaces
                extra_spaces = self.cleaned_data[col].str.strip().ne(self.cleaned_data[col]).sum()
                if extra_spaces > 0:
                    issues.append(f"Extra spaces found in column '{col}': {extra_spaces} rows")
                
                # Check for inconsistent case
                if self.cleaned_data[col].dtype == 'object':
                    sample_values = self.cleaned_data[col].dropna().head(10)
                    if len(sample_values.unique()) != len(sample_values.unique().str.lower().unique()):
                        issues.append(f"Inconsistent text case found in column '{col}'")
        
        # Check for numerical outliers in price
        if 'price' in self.cleaned_data.columns:
            Q1 = self.cleaned_data['price'].quantile(0.25)
            Q3 = self.cleaned_data['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.cleaned_data['price'] < lower_bound) | (self.cleaned_data['price'] > upper_bound)).sum()
            self.cleaning_report['outliers_detected']['price'] = outliers
            if outliers > 0:
                issues.append(f"Price outliers detected: {outliers} rows")
        
        self.cleaning_report['issues_found'] = issues
        logging.info(f"Detected {len(issues)} data quality issues")
        
    def clean_column_names(self) -> None:
        """
        Clean and standardize column names.
        """
        logging.info("Cleaning column names...")
        
        original_columns = self.cleaned_data.columns.tolist()
        
        # Clean column names
        self.cleaned_data.columns = [
            col.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '')
            for col in self.cleaned_data.columns
        ]
        
        # Handle special cases
        column_mapping = {
            'unnamed:_0': 'index',
            'spec_rating': 'spec_rating',
            'cpu': 'processor_name',
            'ram': 'ram_amount',
            'ram_type': 'ram_type',
            'rom': 'storage_amount',
            'rom_type': 'storage_type',
            'gpu': 'gpu_name',
            'display_size': 'screen_size',
            'resolution_width': 'resolution_width',
            'resolution_height': 'resolution_height',
            'os': 'operating_system'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.cleaned_data.columns:
                self.cleaned_data.rename(columns={old_name: new_name}, inplace=True)
        
        new_columns = self.cleaned_data.columns.tolist()
        self.cleaning_report['actions_taken'].append(f"Column names cleaned: {original_columns} -> {new_columns}")
        
        logging.info("Column names cleaned successfully.")
    
    def clean_text_data(self) -> None:
        """
        Clean and normalize text data in string columns.
        """
        logging.info("Cleaning text data...")
        
        text_columns = self.cleaned_data.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if col in self.cleaned_data.columns:
                # Remove extra spaces and normalize
                self.cleaned_data[col] = self.cleaned_data[col].astype(str).str.strip()
                
                # Replace multiple spaces with single space
                self.cleaned_data[col] = self.cleaned_data[col].str.replace(r'\s+', ' ', regex=True)
                
                # Standardize common values
                if col == 'brand':
                    # Standardize brand names
                    brand_mapping = {
                        'hp': 'HP',
                        'Hp': 'HP',
                        'hP': 'HP',
                        'dell': 'Dell',
                        'Dell': 'Dell',
                        'lenovo': 'Lenovo',
                        'Lenovo': 'Lenovo',
                        'acer': 'Acer',
                        'Acer': 'Acer',
                        'asus': 'ASUS',
                        'Asus': 'ASUS',
                        'MSI': 'MSI',
                        'msi': 'MSI',
                        'apple': 'Apple',
                        'Apple': 'Apple'
                    }
                    self.cleaned_data[col] = self.cleaned_data[col].map(brand_mapping).fillna(self.cleaned_data[col])
                
                if col == 'ram_type':
                    # Standardize RAM types
                    self.cleaned_data[col] = self.cleaned_data[col].str.upper()
                    self.cleaned_data[col] = self.cleaned_data[col].str.replace('-', '')
                
                if col == 'storage_type':
                    # Standardize storage types
                    self.cleaned_data[col] = self.cleaned_data[col].str.upper()
                    self.cleaned_data[col] = self.cleaned_data[col].replace('HARD-DISK', 'HDD')
                
                if col == 'operating_system':
                    # Standardize OS names
                    self.cleaned_data[col] = self.cleaned_data[col].str.title()
                    os_mapping = {
                        'Windows 11 Os': 'Windows 11',
                        'Windows 10 Os': 'Windows 10',
                        'Windows 10  Os': 'Windows 10',
                        'Windows Os': 'Windows',
                        'Mac Os': 'macOS',
                        'Chrome Os': 'Chrome OS'
                    }
                    self.cleaned_data[col] = self.cleaned_data[col].replace(os_mapping)
        
        self.cleaning_report['actions_taken'].append("Text data cleaned and normalized")
        logging.info("Text data cleaning completed.")
    
    def clean_numerical_data(self) -> None:
        """
        Clean and convert numerical data.
        """
        logging.info("Cleaning numerical data...")
        
        # Clean price column
        if 'price' in self.cleaned_data.columns:
            # Remove currency symbols and commas, convert to numeric
            self.cleaned_data['price'] = self.cleaned_data['price'].astype(str).str.replace(r'[₹,\s]', '', regex=True)
            self.cleaned_data['price'] = pd.to_numeric(self.cleaned_data['price'], errors='coerce')
            
            # Remove price outliers using IQR method
            Q1 = self.cleaned_data['price'].quantile(0.25)
            Q3 = self.cleaned_data['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.cleaned_data['price'] < lower_bound) | (self.cleaned_data['price'] > upper_bound)).sum()
            self.cleaned_data = self.cleaned_data[
                (self.cleaned_data['price'] >= lower_bound) & 
                (self.cleaned_data['price'] <= upper_bound)
            ]
            
            self.cleaning_report['actions_taken'].append(f"Price outliers removed: {outlier_count} rows")
        
        # Clean spec_rating column
        if 'spec_rating' in self.cleaned_data.columns:
            self.cleaned_data['spec_rating'] = pd.to_numeric(self.cleaned_data['spec_rating'], errors='coerce')
            # Remove impossible spec ratings (should be 0-100)
            invalid_ratings = (~self.cleaned_data['spec_rating'].between(0, 100)).sum()
            self.cleaned_data = self.cleaned_data[self.cleaned_data['spec_rating'].between(0, 100)]
            self.cleaning_report['actions_taken'].append(f"Invalid spec ratings removed: {invalid_ratings} rows")
        
        # Clean display_size column
        if 'screen_size' in self.cleaned_data.columns:
            self.cleaned_data['screen_size'] = pd.to_numeric(self.cleaned_data['screen_size'], errors='coerce')
            # Remove impossible screen sizes (should be reasonable laptop sizes)
            invalid_sizes = (~self.cleaned_data['screen_size'].between(10, 20)).sum()
            self.cleaned_data = self.cleaned_data[self.cleaned_data['screen_size'].between(10, 20)]
            self.cleaning_report['actions_taken'].append(f"Invalid screen sizes removed: {invalid_sizes} rows")
        
        # Clean RAM and storage columns
        if 'ram_amount' in self.cleaned_data.columns:
            # Extract numeric values from RAM strings
            self.cleaned_data['ram_amount'] = self.cleaned_data['ram_amount'].astype(str).str.extract(r'(\d+)').astype(float)
            # Remove impossible RAM values
            invalid_ram = (~self.cleaned_data['ram_amount'].between(2, 128)).sum()
            self.cleaned_data = self.cleaned_data[self.cleaned_data['ram_amount'].between(2, 128)]
            self.cleaning_report['actions_taken'].append(f"Invalid RAM values removed: {invalid_ram} rows")
        
        if 'storage_amount' in self.cleaned_data.columns:
            # Extract numeric values from storage strings
            self.cleaned_data['storage_amount'] = self.cleaned_data['storage_amount'].astype(str).str.extract(r'(\d+)').astype(float)
            # Convert GB to TB if needed
            self.cleaned_data.loc[self.cleaned_data['storage_amount'] > 1000, 'storage_amount'] /= 1000
            self.cleaning_report['actions_taken'].append("Storage amounts normalized")
        
        # Clean resolution columns
        resolution_cols = ['resolution_width', 'resolution_height']
        for col in resolution_cols:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                # Remove impossible resolution values
                invalid_res = (~self.cleaned_data[col].between(800, 4000)).sum()
                self.cleaned_data = self.cleaned_data[self.cleaned_data[col].between(800, 4000)]
                self.cleaning_report['actions_taken'].append(f"Invalid resolution values removed from {col}: {invalid_res} rows")
        
        # Clean warranty column
        if 'warranty' in self.cleaned_data.columns:
            self.cleaned_data['warranty'] = pd.to_numeric(self.cleaned_data['warranty'], errors='coerce')
            # Remove impossible warranty values
            invalid_warranty = (~self.cleaned_data['warranty'].between(0, 10)).sum()
            self.cleaned_data = self.cleaned_data[self.cleaned_data['warranty'].between(0, 10)]
            self.cleaning_report['actions_taken'].append(f"Invalid warranty values removed: {invalid_warranty} rows")
        
        logging.info("Numerical data cleaning completed.")
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values in the dataset.
        """
        logging.info("Handling missing values...")
        
        missing_before = self.cleaned_data.isnull().sum().sum()
        
        # Fill missing values based on column type and context
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].isnull().sum() > 0:
                if col in ['ram_amount', 'storage_amount']:
                    # Fill with median for numerical columns
                    median_val = self.cleaned_data[col].median()
                    self.cleaned_data[col].fillna(median_val, inplace=True)
                
                elif col == 'warranty':
                    # Fill warranty with 1 year (most common)
                    self.cleaned_data[col].fillna(1, inplace=True)
                
                elif col in ['ram_type', 'storage_type', 'operating_system']:
                    # Fill categorical with mode
                    mode_val = self.cleaned_data[col].mode().iloc[0] if not self.cleaned_data[col].mode().empty else 'Unknown'
                    self.cleaned_data[col].fillna(mode_val, inplace=True)
                
                elif col in ['gpu_name', 'processor_name']:
                    # For specific components, use a generic value
                    self.cleaned_data[col].fillna('Unknown', inplace=True)
        
        missing_after = self.cleaned_data.isnull().sum().sum()
        self.cleaning_report['missing_values_after'] = self.cleaned_data.isnull().sum().to_dict()
        self.cleaning_report['actions_taken'].append(f"Missing values filled: {missing_before} -> {missing_after}")
        
        logging.info("Missing values handling completed.")
    
    def remove_duplicates(self) -> None:
        """
        Remove duplicate rows from the dataset.
        """
        logging.info("Removing duplicate rows...")
        
        initial_rows = len(self.cleaned_data)
        self.cleaned_data.drop_duplicates(inplace=True)
        final_rows = len(self.cleaned_data)
        duplicates_removed = initial_rows - final_rows
        
        self.cleaning_report['actions_taken'].append(f"Duplicates removed: {duplicates_removed} rows")
        logging.info(f"Removed {duplicates_removed} duplicate rows.")
    
    def create_engineered_features(self) -> None:
        """
        Create new features from existing data.
        """
        logging.info("Creating engineered features...")
        
        # Screen resolution feature
        if all(col in self.cleaned_data.columns for col in ['resolution_width', 'resolution_height']):
            self.cleaned_data['total_pixels'] = (
                self.cleaned_data['resolution_width'] * self.cleaned_data['resolution_height']
            )
            self.cleaning_report['actions_taken'].append("Created total_pixels feature")
        
        # Storage per price ratio
        if 'storage_amount' in self.cleaned_data.columns and 'price' in self.cleaned_data.columns:
            self.cleaned_data['storage_per_rupee'] = self.cleaned_data['storage_amount'] / self.cleaned_data['price']
            self.cleaning_report['actions_taken'].append("Created storage_per_rupee feature")
        
        # RAM per price ratio
        if 'ram_amount' in self.cleaned_data.columns and 'price' in self.cleaned_data.columns:
            self.cleaned_data['ram_per_rupee'] = self.cleaned_data['ram_amount'] / self.cleaned_data['price']
            self.cleaning_report['actions_taken'].append("Created ram_per_rupee feature")
        
        # Value score (combination of spec_rating and price)
        if 'spec_rating' in self.cleaned_data.columns and 'price' in self.cleaned_data.columns:
            self.cleaned_data['value_score'] = self.cleaned_data['spec_rating'] / (self.cleaned_data['price'] / 1000)
            self.cleaning_report['actions_taken'].append("Created value_score feature")
        
        # Gaming laptop indicator
        if 'gpu_name' in self.cleaned_data.columns:
            gaming_gpu_keywords = ['gtx', 'rtx', 'radeon', 'geforce']
            self.cleaned_data['is_gaming'] = self.cleaned_data['gpu_name'].str.lower().str.contains('|'.join(gaming_gpu_keywords), na=False)
            self.cleaning_report['actions_taken'].append("Created is_gaming feature")
        
        # Premium laptop indicator
        if 'price' in self.cleaned_data.columns:
            premium_threshold = self.cleaned_data['price'].quantile(0.75)
            self.cleaned_data['is_premium'] = self.cleaned_data['price'] > premium_threshold
            self.cleaning_report['actions_taken'].append("Created is_premium feature")
        
        logging.info("Feature engineering completed.")
    
    def generate_data_insights(self) -> Dict[str, Any]:
        """
        Generate insights and statistics about the cleaned data.
        
        Returns:
            Dict containing data insights
        """
        logging.info("Generating data insights...")
        
        insights = {
            'shape_after_cleaning': self.cleaned_data.shape,
            'memory_usage_mb': round(self.cleaned_data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'price_statistics': {},
            'brand_distribution': {},
            'spec_rating_distribution': {},
            'popular_specifications': {}
        }
        
        if 'price' in self.cleaned_data.columns:
            price_stats = self.cleaned_data['price'].describe()
            insights['price_statistics'] = {
                'mean': round(price_stats['mean'], 2),
                'median': round(price_stats['50%'], 2),
                'min': round(price_stats['min'], 2),
                'max': round(price_stats['max'], 2),
                'std': round(price_stats['std'], 2)
            }
        
        if 'brand' in self.cleaned_data.columns:
            insights['brand_distribution'] = self.cleaned_data['brand'].value_counts().head(10).to_dict()
        
        if 'spec_rating' in self.cleaned_data.columns:
            insights['spec_rating_distribution'] = {
                'mean': round(self.cleaned_data['spec_rating'].mean(), 2),
                'median': round(self.cleaned_data['spec_rating'].median(), 2),
                'min': round(self.cleaned_data['spec_rating'].min(), 2),
                'max': round(self.cleaned_data['spec_rating'].max(), 2)
            }
        
        if 'ram_amount' in self.cleaned_data.columns:
            insights['popular_specifications']['ram_distribution'] = self.cleaned_data['ram_amount'].value_counts().head(5).to_dict()
        
        if 'storage_amount' in self.cleaned_data.columns:
            insights['popular_specifications']['storage_distribution'] = self.cleaned_data['storage_amount'].value_counts().head(5).to_dict()
        
        logging.info("Data insights generated successfully.")
        return insights
    
    def generate_visualizations(self, output_dir: str = "visualizations") -> None:
        """
        Generate visualization plots for the cleaned data.
        
        Args:
            output_dir (str): Directory to save visualization plots
        """
        logging.info("Generating visualizations...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Price distribution
        if 'price' in self.cleaned_data.columns:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(self.cleaned_data['price'], bins=30, alpha=0.7, color='skyblue')
            plt.title('Price Distribution')
            plt.xlabel('Price (₹)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(self.cleaned_data['price'])
            plt.title('Price Box Plot')
            plt.ylabel('Price (₹)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/price_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Brand distribution
        if 'brand' in self.cleaned_data.columns:
            plt.figure(figsize=(14, 8))
            brand_counts = self.cleaned_data['brand'].value_counts().head(15)
            plt.bar(range(len(brand_counts)), brand_counts.values)
            plt.title('Top 15 Laptop Brands by Count')
            plt.xlabel('Brand')
            plt.ylabel('Count')
            plt.xticks(range(len(brand_counts)), brand_counts.index, rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/brand_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Spec Rating vs Price scatter plot
        if 'spec_rating' in self.cleaned_data.columns and 'price' in self.cleaned_data.columns:
            plt.figure(figsize=(12, 8))
            plt.scatter(self.cleaned_data['spec_rating'], self.cleaned_data['price'], alpha=0.6)
            plt.title('Spec Rating vs Price')
            plt.xlabel('Spec Rating')
            plt.ylabel('Price (₹)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/spec_rating_vs_price.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Correlation heatmap
        numerical_columns = self.cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.cleaned_data[numerical_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # RAM and Storage distribution
        if 'ram_amount' in self.cleaned_data.columns:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            ram_counts = self.cleaned_data['ram_amount'].value_counts().sort_index()
            plt.bar(ram_counts.index, ram_counts.values)
            plt.title('RAM Distribution')
            plt.xlabel('RAM (GB)')
            plt.ylabel('Count')
            
            if 'storage_amount' in self.cleaned_data.columns:
                plt.subplot(1, 2, 2)
                storage_counts = self.cleaned_data['storage_amount'].value_counts().sort_index()
                plt.bar(storage_counts.index, storage_counts.values)
                plt.title('Storage Distribution')
                plt.xlabel('Storage (GB)')
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/hardware_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logging.info(f"Visualizations saved to {output_dir}/")
    
    def save_cleaned_data(self, output_path: str) -> bool:
        """
        Save the cleaned dataset to file.
        
        Args:
            output_path (str): Path to save the cleaned dataset
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            self.cleaned_data.to_csv(output_path, index=False)
            logging.info(f"Cleaned data saved to: {output_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save cleaned data: {str(e)}")
            return False
    
    def generate_debug_report(self, output_path: str = "data_cleaning_report.txt") -> None:
        """
        Generate a comprehensive debug report of the cleaning process.
        
        Args:
            output_path (str): Path to save the debug report
        """
        logging.info("Generating debug report...")
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE DATA CLEANING DEBUG REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n\n")
            
            # Basic Information
            f.write("BASIC DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Original shape: {self.cleaning_report['total_rows']} rows × {self.cleaning_report['total_columns']} columns\n")
            f.write(f"Final shape: {self.cleaned_data.shape[0]} rows × {self.cleaned_data.shape[1]} columns\n")
            f.write(f"Rows removed: {self.cleaning_report['total_rows'] - self.cleaned_data.shape[0]}\n")
            f.write(f"Columns: {', '.join(self.cleaned_data.columns)}\n\n")
            
            # Issues Found
            f.write("ISSUES DETECTED\n")
            f.write("-" * 40 + "\n")
            if self.cleaning_report['issues_found']:
                for issue in self.cleaning_report['issues_found']:
                    f.write(f"• {issue}\n")
            else:
                f.write("No major issues detected.\n")
            f.write("\n")
            
            # Actions Taken
            f.write("CLEANING ACTIONS PERFORMED\n")
            f.write("-" * 40 + "\n")
            for action in self.cleaning_report['actions_taken']:
                f.write(f"• {action}\n")
            f.write("\n")
            
            # Missing Values Comparison
            f.write("MISSING VALUES COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write("Column\t\t\tBefore\tAfter\n")
            f.write("-" * 60 + "\n")
            for col in self.cleaning_report['missing_values_before'].keys():
                before = self.cleaning_report['missing_values_before'][col]
                after = self.cleaning_report['missing_values_after'].get(col, 0)
                f.write(f"{col[:25]:25}\t{before}\t{after}\n")
            f.write("\n")
            
            # Duplicates
            f.write("DUPLICATE ROWS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Duplicates found: {self.cleaning_report['duplicates_found']}\n\n")
            
            # Outliers
            f.write("OUTLIERS DETECTED\n")
            f.write("-" * 40 + "\n")
            if self.cleaning_report['outliers_detected']:
                for col, count in self.cleaning_report['outliers_detected'].items():
                    f.write(f"{col}: {count} outliers\n")
            else:
                f.write("No outliers detected.\n")
            f.write("\n")
            
            # Data Types
            f.write("FINAL DATA TYPES\n")
            f.write("-" * 40 + "\n")
            for col, dtype in self.cleaned_data.dtypes.items():
                f.write(f"{col}: {dtype}\n")
            f.write("\n")
            
            # Sample Data
            f.write("SAMPLE OF CLEANED DATA\n")
            f.write("-" * 40 + "\n")
            f.write(self.cleaned_data.head(10).to_string())
            f.write("\n\n")
            
            # Data Quality Summary
            f.write("DATA QUALITY SUMMARY\n")
            f.write("-" * 40 + "\n")
            missing_data = self.cleaned_data.isnull().sum().sum()
            f.write(f"Total missing values: {missing_data}\n")
            f.write(f"Data completeness: {((self.cleaned_data.size - missing_data) / self.cleaned_data.size * 100):.2f}%\n")
            f.write(f"Duplicate rows: {self.cleaned_data.duplicated().sum()}\n")
            f.write(f"Unique values per column:\n")
            for col in self.cleaned_data.columns:
                f.write(f"  {col}: {self.cleaned_data[col].nunique()} unique values\n")
        
        logging.info(f"Debug report saved to: {output_path}")
    
    def run_full_cleaning_pipeline(self, save_output: bool = True) -> Dict[str, Any]:
        """
        Run the complete data cleaning pipeline.
        
        Args:
            save_output (bool): Whether to save the cleaned data and report
            
        Returns:
            Dict containing cleaning results and insights
        """
        logging.info("Starting comprehensive data cleaning pipeline...")
        
        if not self.load_data():
            return {"success": False, "error": "Failed to load data"}
        
        # Step 1: Analyze data structure
        data_structure = self.analyze_data_structure()
        
        # Step 2: Detect issues
        self.detect_and_report_issues()
        
        # Step 3: Clean column names
        self.clean_column_names()
        
        # Step 4: Clean text data
        self.clean_text_data()
        
        # Step 5: Clean numerical data
        self.clean_numerical_data()
        
        # Step 6: Handle missing values
        self.handle_missing_values()
        
        # Step 7: Remove duplicates
        self.remove_duplicates()
        
        # Step 8: Create engineered features
        self.create_engineered_features()
        
        # Step 9: Generate insights
        insights = self.generate_data_insights()
        
        # Step 10: Generate visualizations
        self.generate_visualizations()
        
        # Step 11: Generate debug report
        self.generate_debug_report()
        
        # Step 12: Save cleaned data
        if save_output:
            self.save_cleaned_data("laptop_data_cleaned.csv")
        
        self.cleaning_report['cleaned_rows'] = self.cleaned_data.shape[0]
        
        logging.info("Data cleaning pipeline completed successfully!")
        
        return {
            "success": True,
            "cleaning_report": self.cleaning_report,
            "insights": insights,
            "data_structure": data_structure
        }

def main():
    """
    Main function to run the data cleaning script.
    """
    print("=" * 80)
    print("LAPTOP DATASET CLEANING SCRIPT")
    print("=" * 80)
    print("This script will perform comprehensive data cleaning on the laptop dataset.")
    print("It will generate a detailed debug report and visualizations.")
    print("=" * 80)
    
    # Initialize cleaner
    data_path = "../archive/data.csv"  # Adjust path as needed
    cleaner = LaptopDataCleaner(data_path)
    
    # Run cleaning pipeline
    results = cleaner.run_full_cleaning_pipeline(save_output=True)
    
    if results["success"]:
        print("\n" + "=" * 80)
        print("CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Original dataset: {results['cleaning_report']['total_rows']} rows")
        print(f"Cleaned dataset: {results['cleaning_report']['cleaned_rows']} rows")
        print(f"Rows removed: {results['cleaning_report']['total_rows'] - results['cleaning_report']['cleaned_rows']}")
        print("\nGenerated files:")
        print("• laptop_data_cleaned.csv - Cleaned dataset")
        print("• data_cleaning_debug.log - Detailed cleaning log")
        print("• data_cleaning_report.txt - Comprehensive debug report")
        print("• visualizations/ - Data visualization plots")
        print("\nCheck the generated files for detailed results!")
    else:
        print("Data cleaning failed!")
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()