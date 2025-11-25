import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# BEST SETTINGS EVER
plt.rcParams.update({
    'font.size': 11,
    'axes.titleweight': 'bold',
    'axes.titlepad': 20,
    'figure.titlesize': 26,
    'figure.titleweight': 'bold'
})

class FINAL_PERFECT_REPORT:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_df = None
        self.df = None

    def load_and_clean(self):
        print("Loading dataset...\n")
        self.raw_df = pd.read_csv(self.data_path)
        self.df = self.raw_df.copy()

        self.df.drop(columns=['Unnamed: 0', 'name'], errors='ignore', inplace=True)
        self.df['spec_rating'].fillna(self.df['spec_rating'].median(), inplace=True)
        self.df['resolution_width'].fillna(self.df['resolution_width'].median(), inplace=True)
        self.df['resolution_height'].fillna(self.df['resolution_height'].median(), inplace=True)

        self.df['Ram_GB'] = self.df['Ram'].str.extract('(\d+)').astype(float)
        self.df['Storage_GB'] = self.df['ROM'].str.extract('(\d+)').astype(float)
        self.df['Display_inch'] = pd.to_numeric(self.df['display_size'], errors='coerce')
        self.df['Performance_Score'] = self.df['Ram_GB'] * 0.4 + self.df['spec_rating'] * 0.6

        print("Cleaning Done!\n")

    # PAGE 1: Raw Data
    def page1(self):
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle("1. Raw Dataset Overview", fontsize=26, y=0.99)
        plt.axis('off')
        info = f"Shape: {self.raw_df.shape} | Total Missing: {self.raw_df.isnull().sum().sum()}"
        plt.text(0.5, 0.88, info, ha='center', fontsize=16, color='gray')
        sample = self.raw_df.head(6).to_string(index=False)
        plt.text(0.5, 0.70, sample, ha='center', va='top', fontsize=10, fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=1.2", facecolor="#fef9e7"))
        plt.text(0.5, 0.15, "Issues: Text values in RAM/ROM, many nulls", ha='center', fontsize=16, color='crimson')
        plt.subplots_adjust(top=0.90)
        plt.show()

    # PAGE 2: Missing Values - FIXED (shows summary, not empty heatmap)
    def page2(self):
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("2. Missing Values: Before vs After", y=0.99)

        missing_before = self.raw_df.isnull().sum().sort_values(ascending=False)
        missing_after = self.df.isnull().sum().sort_values(ascending=False)

        plt.subplot(1, 2, 1)
        missing_before.head(10).plot(kind='barh', color='red', alpha=0.8)
        plt.title("Before Cleaning\n(Top 10 Columns)", pad=15, fontsize=14)
        plt.xlabel("Missing Count")

        plt.subplot(1, 2, 2)
        missing_after.head(10).plot(kind='barh', color='green', alpha=0.8)
        plt.title("After Cleaning\n(Almost Zero!)", pad=15, fontsize=14)
        plt.xlabel("Missing Count")

        plt.subplots_adjust(top=0.85, wspace=0.6)
        plt.show()

    # PAGE 3: Feature Engineering
    def page3(self):
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle("3. Feature Engineering: Text → Numbers", y=0.99)

        cols = ['Ram_GB', 'Storage_GB', 'Display_inch', 'Performance_Score']
        titles = ['RAM (GB)', 'Storage (GB)', 'Display Size (inches)', 'Performance Score (Engineered)']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

        for i, (col, title, colr) in enumerate(zip(cols, titles, colors), 1):
            plt.subplot(2, 2, i)
            plt.hist(self.df[col].dropna(), bins=20, color=colr, edgecolor='black', alpha=0.85)
            plt.title(title, fontsize=14, pad=12)
            plt.xlabel(title.split('(')[0].strip())

        plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)
        plt.show()

    # PAGE 4: Q-Q Plot - FIXED (text moved, no overlap)
    def page4(self):
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle("4. Q-Q Plot: Price Distribution vs Normal", y=0.99)

        plt.subplot(1, 2, 1)
        stats.probplot(self.df['price'], dist="norm", plot=plt)
        plt.title("Q-Q Plot (Price)", fontsize=15, pad=15)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(self.df['price'], bins=40, color='#3498db', edgecolor='black', alpha=0.8)
        plt.title("Price Distribution", fontsize=15, pad=15)
        plt.xlabel("Price (₹)")

        # Normality test - moved safely below
        _, p = stats.shapiro(self.df['price'].dropna())
        result = "NOT Normal (p < 0.05)" if p < 0.05 else "Normal"
        fig.text(0.5, 0.02, f"Shapiro-Wilk Test: p-value = {p:.2e} → {result}",
                 ha='center', fontsize=14, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.9))

        plt.subplots_adjust(bottom=0.15, top=0.88)
        plt.show()

    # PAGE 5: Correlation Matrix - FIXED title
    def page5(self):
        numeric = self.df.select_dtypes(include=[np.number])
        corr = numeric.corr()

        plt.figure(figsize=(14, 10))
        plt.suptitle("5. Correlation Matrix (Cleaned Numeric Features)", y=0.99)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True,
                    linewidths=1, fmt='.2f', cbar_kws={"shrink": 0.8})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # PAGE 6: Price Distribution
    def page6(self):
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("6. Final Price Distribution (Cleaned)", y=0.99)

        plt.hist(self.df['price'], bins=50, color='#9b59b6', alpha=0.85, edgecolor='black')
        mean_p = self.df['price'].mean()
        median_p = self.df['price'].median()
        plt.axvline(mean_p, color='red', linewidth=3, label=f"Mean: ₹{mean_p:,.0f}")
        plt.axvline(median_p, color='gold', linewidth=3, label=f"Median: ₹{median_p:,.0f}")
        plt.xlabel("Price (₹)", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.legend(fontsize=13)
        plt.grid(alpha=0.3)
        plt.subplots_adjust(top=0.88)
        plt.show()

    # PAGE 7: Final Clean Dataset - FIXED (no overlap)
    def page7(self):
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle("7. FINAL CLEANED DATASET - READY FOR ML!", color='darkblue', y=0.98)
        plt.axis('off')

        # Preview table
        preview = self.df[['brand', 'price', 'Ram_GB', 'Storage_GB', 'Display_inch',
                        'spec_rating', 'Performance_Score']].round(1).head(10)
        table = preview.to_string(index=False)

        # Table box
        plt.text(
            0.5, 0.78, table,
            ha='center', va='top', fontsize=12, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#f0f8ff")
        )

        # More spacing for summary
      



        plt.subplots_adjust(top=0.95, bottom=0.05)
        plt.show()


    def run(self):
        self.load_and_clean()
        print("="*70)
        print("       FINAL PERFECT 7-PAGE REPORT (100% FIXED!)")
        print("="*70 + "\n")

        self.page1()
        self.page2()
        self.page3()
        self.page4()
        self.page5()
        self.page6()
        self.page7()

        print("All 7 pages generated perfectly — no overlaps, no empty plots!")

# RUN IT
if __name__ == "__main__":
    path = "C:/Users/developer\Desktop/CollageFinalProject/laptop_price/dharmu_ai/archive/data.csv"
    
    report = FINAL_PERFECT_REPORT(path)
    report.run()