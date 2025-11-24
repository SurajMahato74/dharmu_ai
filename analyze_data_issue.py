import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('archive/data.csv')

print("=== Dataset Analysis ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Extract RAM values
def extract_ram(ram_str):
    if pd.isna(ram_str):
        return 0
    ram_str = str(ram_str).upper()
    if 'GB' in ram_str:
        return int(ram_str.replace('GB', '').strip())
    return 0

df['ram_numeric'] = df['Ram'].apply(extract_ram)

print("\n=== RAM Distribution ===")
print(df['ram_numeric'].value_counts().sort_index())

print("\n=== Price vs RAM Analysis ===")
ram_price_analysis = df.groupby('ram_numeric')['price'].agg(['count', 'mean', 'median', 'std']).round(2)
print(ram_price_analysis)

# Check specific examples
print("\n=== High RAM Examples ===")
high_ram = df[df['ram_numeric'] >= 32].sort_values('price')
print(high_ram[['brand', 'name', 'price', 'Ram', 'processor', 'GPU']].head(10))

print("\n=== Low RAM Examples ===")
low_ram = df[df['ram_numeric'] <= 8].sort_values('price', ascending=False)
print(low_ram[['brand', 'name', 'price', 'Ram', 'processor', 'GPU']].head(10))

# Correlation analysis
print("\n=== Correlation Analysis ===")
correlation_cols = ['price', 'ram_numeric', 'spec_rating']
if 'ROM' in df.columns:
    # Extract storage
    def extract_storage(storage_str):
        if pd.isna(storage_str):
            return 0
        storage_str = str(storage_str).upper()
        if 'TB' in storage_str:
            return int(float(storage_str.replace('TB', '').strip()) * 1024)
        elif 'GB' in storage_str:
            return int(storage_str.replace('GB', '').strip())
        return 0
    
    df['storage_numeric'] = df['ROM'].apply(extract_storage)
    correlation_cols.append('storage_numeric')

corr_matrix = df[correlation_cols].corr()
print(corr_matrix)

# Check for outliers
print("\n=== Outlier Analysis ===")
print("32GB RAM laptops with very low prices:")
outliers = df[(df['ram_numeric'] == 32) & (df['price'] < 100000)]
print(outliers[['brand', 'name', 'price', 'Ram', 'processor', 'GPU', 'spec_rating']])

print("\n=== Brand Analysis for High RAM ===")
high_ram_brands = df[df['ram_numeric'] >= 16].groupby('brand')['price'].agg(['count', 'mean']).sort_values('mean')
print(high_ram_brands)

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
df.boxplot(column='price', by='ram_numeric', ax=plt.gca())
plt.title('Price Distribution by RAM')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
plt.scatter(df['ram_numeric'], df['price'], alpha=0.6)
plt.xlabel('RAM (GB)')
plt.ylabel('Price (₹)')
plt.title('RAM vs Price Scatter Plot')

plt.subplot(2, 3, 3)
ram_counts = df['ram_numeric'].value_counts().sort_index()
plt.bar(ram_counts.index, ram_counts.values)
plt.xlabel('RAM (GB)')
plt.ylabel('Count')
plt.title('RAM Distribution')

plt.subplot(2, 3, 4)
# Price vs Spec Rating
plt.scatter(df['spec_rating'], df['price'], alpha=0.6)
plt.xlabel('Spec Rating')
plt.ylabel('Price (₹)')
plt.title('Spec Rating vs Price')

plt.subplot(2, 3, 5)
# Check gaming laptops
gaming_keywords = ['gaming', 'rog', 'tuf', 'predator', 'legion', 'omen', 'nitro']
df['is_gaming'] = df['name'].str.lower().str.contains('|'.join(gaming_keywords), na=False)
gaming_analysis = df.groupby(['ram_numeric', 'is_gaming'])['price'].mean().unstack()
gaming_analysis.plot(kind='bar', ax=plt.gca())
plt.title('Average Price by RAM and Gaming Type')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Model Training Issue Analysis ===")
print("The issue might be:")
print("1. Data quality - some high-end laptops might have incorrect prices")
print("2. Feature engineering - RAM alone doesn't determine price")
print("3. Model overfitting to outliers")
print("4. Need to consider brand, processor, GPU together")

# Check specific problematic cases
print("\n=== Problematic Cases ===")
print("High RAM but low price (potential data errors):")
problematic = df[(df['ram_numeric'] >= 32) & (df['price'] < 150000)]
for idx, row in problematic.iterrows():
    print(f"- {row['brand']} {row['name']}: {row['ram_numeric']}GB RAM, Rs.{row['price']}")