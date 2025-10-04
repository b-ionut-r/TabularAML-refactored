"""
Distribution Shift Analysis Between Train and Test Datasets
===========================================================

This script performs comprehensive distribution shift analysis between train.csv and test.csv,
including statistical tests and visualizations for all features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class DistributionShiftAnalyzer:
    """
    Comprehensive distribution shift analyzer for tabular data.
    
    This class provides methods to analyze and visualize distribution shifts
    between training and test datasets, including statistical significance tests.
    """
    
    def __init__(self, train_path, test_path):
        """Initialize with paths to train and test CSV files."""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # Get common features (excluding target if present)
        train_features = set(self.train_df.columns) - {'id', 'pollution_value'}
        test_features = set(self.test_df.columns) - {'id'}
        self.common_features = list(train_features & test_features)
        
        # Categorize features
        self.numeric_features = []
        self.categorical_features = []
        
        for feature in self.common_features:
            if self.train_df[feature].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (like day_of_week, month, hour)
                if feature in ['day_of_week', 'month', 'hour'] or len(self.train_df[feature].unique()) <= 12:
                    self.categorical_features.append(feature)
                else:
                    self.numeric_features.append(feature)
            else:
                self.categorical_features.append(feature)
        
        print(f"Dataset loaded successfully!")
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"Common features: {len(self.common_features)}")
        print(f"Numeric features: {self.numeric_features}")
        print(f"Categorical features: {self.categorical_features}")
        
    def plot_numeric_distributions(self, save_plots=True):
        """
        Create histograms and density plots for numeric features comparing train vs test.
        
        Uses KDE overlays and statistical tests to highlight distribution differences.
        """
        if not self.numeric_features:
            print("No numeric features found for distribution plotting.")
            return {}
        
        n_features = len(self.numeric_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        ks_results = {}
        
        for idx, feature in enumerate(self.numeric_features):
            ax = axes[idx]
            
            # Get data
            train_data = self.train_df[feature].dropna()
            test_data = self.test_df[feature].dropna()
            
            # Create histograms with density
            ax.hist(train_data, bins=30, alpha=0.6, density=True, 
                   label=f'Train (n={len(train_data)})', color='skyblue')
            ax.hist(test_data, bins=30, alpha=0.6, density=True, 
                   label=f'Test (n={len(test_data)})', color='lightcoral')
            
            # Add KDE curves
            try:
                train_kde = stats.gaussian_kde(train_data)
                test_kde = stats.gaussian_kde(test_data)
                
                x_range = np.linspace(min(train_data.min(), test_data.min()),
                                    max(train_data.max(), test_data.max()), 200)
                ax.plot(x_range, train_kde(x_range), 'b-', linewidth=2, alpha=0.8)
                ax.plot(x_range, test_kde(x_range), 'r-', linewidth=2, alpha=0.8)
            except:
                pass  # Skip KDE if data is problematic
            
            # Perform KS test
            ks_stat, p_value = ks_2samp(train_data, test_data)
            ks_results[feature] = {'ks_statistic': ks_stat, 'p_value': p_value}
            
            # Set labels and title
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'{feature.replace("_", " ").title()}\nKS test p-value: {p_value:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Highlight significant shifts
            if p_value < 0.05:
                ax.patch.set_facecolor('mistyrose')
        
        # Remove empty subplots
        if n_features < n_rows * n_cols:
            for idx in range(n_features, n_rows * n_cols):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.suptitle('Numeric Feature Distributions: Train vs Test', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_plots:
            plt.savefig('numeric_distributions.png', dpi=300, bbox_inches='tight')
            print("Saved: numeric_distributions.png")
        
        plt.show()
        
        return ks_results
    
    def plot_categorical_distributions(self, save_plots=True):
        """
        Create bar charts for categorical features comparing proportions between train and test.
        
        Includes chi-square tests for independence testing.
        """
        if not self.categorical_features:
            print("No categorical features found for distribution plotting.")
            return {}
        
        n_features = len(self.categorical_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        chi2_results = {}
        
        for idx, feature in enumerate(self.categorical_features):
            ax = axes[idx]
            
            # Get value counts and proportions
            train_counts = self.train_df[feature].value_counts().sort_index()
            test_counts = self.test_df[feature].value_counts().sort_index()
            
            # Align indices
            all_values = sorted(set(train_counts.index) | set(test_counts.index))
            train_props = train_counts.reindex(all_values, fill_value=0) / len(self.train_df)
            test_props = test_counts.reindex(all_values, fill_value=0) / len(self.test_df)
            
            # Create bar plot
            x = np.arange(len(all_values))
            width = 0.35
            
            ax.bar(x - width/2, train_props, width, label='Train', alpha=0.8, color='skyblue')
            ax.bar(x + width/2, test_props, width, label='Test', alpha=0.8, color='lightcoral')
            
            # Perform chi-square test
            try:
                # Create contingency table
                contingency_table = pd.DataFrame({
                    'train': train_counts.reindex(all_values, fill_value=0),
                    'test': test_counts.reindex(all_values, fill_value=0)
                }).T
                
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                chi2_results[feature] = {'chi2_statistic': chi2_stat, 'p_value': p_value}
            except:
                chi2_results[feature] = {'chi2_statistic': np.nan, 'p_value': np.nan}
                p_value = np.nan
            
            # Set labels and formatting
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Proportion')
            ax.set_title(f'{feature.replace("_", " ").title()}\nChi-square p-value: {p_value:.4f}' if not np.isnan(p_value) else f'{feature.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(all_values, rotation=45 if len(all_values) > 5 else 0)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight significant shifts
            if not np.isnan(p_value) and p_value < 0.05:
                ax.patch.set_facecolor('mistyrose')
        
        # Remove empty subplots
        if n_features < n_rows * n_cols:
            for idx in range(n_features, n_rows * n_cols):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.suptitle('Categorical Feature Distributions: Train vs Test', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_plots:
            plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
            print("Saved: categorical_distributions.png")
        
        plt.show()
        
        return chi2_results
    
    def create_summary_dashboard(self, ks_results, chi2_results, save_plots=True):
        """
        Create a summary dashboard showing the most significant distribution shifts.
        
        Combines statistical test results and provides visual ranking of shift significance.
        """
        # Combine all test results
        all_results = []
        
        # Add KS test results
        for feature, result in ks_results.items():
            all_results.append({
                'feature': feature,
                'test_type': 'KS Test',
                'statistic': result['ks_statistic'],
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })
        
        # Add Chi-square test results
        for feature, result in chi2_results.items():
            if not np.isnan(result['p_value']):
                all_results.append({
                    'feature': feature,
                    'test_type': 'Chi-square',
                    'statistic': result['chi2_statistic'],
                    'p_value': result['p_value'],
                    'significant': result['p_value'] < 0.05
                })
        
        if not all_results:
            print("No test results available for summary dashboard.")
            return
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('p_value')
        
        # Create summary dashboard
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: P-values ranking
        ax1 = plt.subplot(2, 2, 1)
        colors = ['red' if sig else 'gray' for sig in results_df['significant']]
        bars = ax1.barh(range(len(results_df)), -np.log10(results_df['p_value']), color=colors, alpha=0.7)
        ax1.set_yticks(range(len(results_df)))
        ax1.set_yticklabels([f"{row['feature']} ({row['test_type']})" for _, row in results_df.iterrows()])
        ax1.set_xlabel('-log10(p-value)')
        ax1.set_title('Statistical Significance of Distribution Shifts\n(Red = Significant, Gray = Not Significant)')
        ax1.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05 threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Test statistics
        ax2 = plt.subplot(2, 2, 2)
        scatter_colors = ['red' if sig else 'gray' for sig in results_df['significant']]
        ax2.scatter(results_df['statistic'], -np.log10(results_df['p_value']), 
                   c=scatter_colors, alpha=0.7, s=100)
        ax2.set_xlabel('Test Statistic')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title('Test Statistic vs Significance')
        ax2.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add feature labels to points
        for i, row in results_df.iterrows():
            ax2.annotate(row['feature'], (row['statistic'], -np.log10(row['p_value'])), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        # Plot 3: Summary statistics table
        ax3 = plt.subplot(2, 1, 2)
        ax3.axis('tight')
        ax3.axis('off')
        
        # Create summary table
        summary_data = []
        summary_data.append(['Total Features Analyzed', str(len(results_df))])
        summary_data.append(['Features with Significant Shifts (p < 0.05)', str(sum(results_df['significant']))])
        summary_data.append(['Features with No Significant Shifts', str(sum(~results_df['significant']))])
        summary_data.append(['Most Significant Feature', 
                           f"{results_df.iloc[0]['feature']} (p={results_df.iloc[0]['p_value']:.4f})"])
        
        if sum(results_df['significant']) > 0:
            significant_features = results_df[results_df['significant']]['feature'].tolist()
            summary_data.append(['All Significant Features', ', '.join(significant_features)])
        
        table = ax3.table(cellText=summary_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
        
        ax3.set_title('Distribution Shift Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('distribution_shift_summary.png', dpi=300, bbox_inches='tight')
            print("Saved: distribution_shift_summary.png")
        
        plt.show()
        
        return results_df
    
    def generate_report(self, ks_results, chi2_results, results_df):
        """
        Generate a comprehensive text report of distribution shift findings.
        
        Provides actionable insights and recommendations based on the analysis.
        """
        report = []
        report.append("="*80)
        report.append("DISTRIBUTION SHIFT ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append(f"• Training set: {self.train_df.shape[0]:,} samples, {self.train_df.shape[1]} features")
        report.append(f"• Test set: {self.test_df.shape[0]:,} samples, {self.test_df.shape[1]} features")
        report.append(f"• Common features analyzed: {len(self.common_features)}")
        report.append(f"• Numeric features: {len(self.numeric_features)}")
        report.append(f"• Categorical features: {len(self.categorical_features)}")
        report.append("")
        
        # Significant shifts
        significant_shifts = results_df[results_df['significant']]
        report.append("SIGNIFICANT DISTRIBUTION SHIFTS (p < 0.05):")
        if len(significant_shifts) == 0:
            report.append("• No statistically significant distribution shifts detected.")
        else:
            for _, row in significant_shifts.iterrows():
                report.append(f"• {row['feature']} ({row['test_type']}): p-value = {row['p_value']:.4f}")
        report.append("")
        
        # Feature-specific insights
        if len(significant_shifts) > 0:
            report.append("DETAILED FEATURE ANALYSIS:")
            
            for feature in significant_shifts['feature'].unique():
                report.append(f"\n{feature.upper()}:")
                
                if feature in self.numeric_features:
                    train_data = self.train_df[feature].dropna()
                    test_data = self.test_df[feature].dropna()
                    
                    report.append(f"  • Type: Numeric")
                    report.append(f"  • Train: mean={train_data.mean():.3f}, std={train_data.std():.3f}, range=[{train_data.min():.3f}, {train_data.max():.3f}]")
                    report.append(f"  • Test:  mean={test_data.mean():.3f}, std={test_data.std():.3f}, range=[{test_data.min():.3f}, {test_data.max():.3f}]")
                    
                    # Interpretation
                    mean_diff = abs(train_data.mean() - test_data.mean())
                    std_diff = abs(train_data.std() - test_data.std())
                    
                    if mean_diff > 0.1 * train_data.std():
                        report.append(f"  • ALERT: Substantial difference in central tendency detected")
                    if std_diff > 0.1 * train_data.std():
                        report.append(f"  • ALERT: Substantial difference in variability detected")
                
                elif feature in self.categorical_features:
                    train_counts = self.train_df[feature].value_counts(normalize=True)
                    test_counts = self.test_df[feature].value_counts(normalize=True)
                    
                    report.append(f"  • Type: Categorical")
                    report.append(f"  • Unique values: {len(set(train_counts.index) | set(test_counts.index))}")
                    
                    # Find biggest proportion differences
                    all_values = set(train_counts.index) | set(test_counts.index)
                    max_diff = 0
                    max_diff_value = None
                    
                    for value in all_values:
                        train_prop = train_counts.get(value, 0)
                        test_prop = test_counts.get(value, 0)
                        diff = abs(train_prop - test_prop)
                        if diff > max_diff:
                            max_diff = diff
                            max_diff_value = value
                    
                    if max_diff > 0.05:
                        report.append(f"  • ALERT: Largest proportion difference at value '{max_diff_value}': {max_diff:.3f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        if len(significant_shifts) == 0:
            report.append("• Distributions appear stable between train and test sets.")
            report.append("• Standard model validation approaches should be reliable.")
        else:
            report.append("• Significant distribution shifts detected - consider the following:")
            report.append("  - Use domain adaptation techniques")
            report.append("  - Apply importance weighting during training")
            report.append("  - Monitor model performance more carefully on test set")
            report.append("  - Consider collecting more representative training data")
            
            if 'latitude' in significant_shifts['feature'].values or 'longitude' in significant_shifts['feature'].values:
                report.append("  - Geographic distribution shift detected - consider spatial validation strategies")
            
            temporal_features = ['day_of_year', 'day_of_week', 'hour', 'month']
            if any(f in significant_shifts['feature'].values for f in temporal_features):
                report.append("  - Temporal distribution shift detected - consider time-based validation splits")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        with open('distribution_shift_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Report saved: distribution_shift_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """Main execution function."""
    print("Starting Distribution Shift Analysis...")
    print("="*50)
    
    # Initialize analyzer
    analyzer = DistributionShiftAnalyzer('train.csv', 'test.csv')
    
    # Perform analysis
    print("\n1. Analyzing numeric feature distributions...")
    ks_results = analyzer.plot_numeric_distributions()
    
    print("\n2. Analyzing categorical feature distributions...")
    chi2_results = analyzer.plot_categorical_distributions()
    
    print("\n3. Creating summary dashboard...")
    results_df = analyzer.create_summary_dashboard(ks_results, chi2_results)
    
    print("\n4. Generating comprehensive report...")
    analyzer.generate_report(ks_results, chi2_results, results_df)
    
    print("\nAnalysis complete! Check the following files:")
    print("• numeric_distributions.png - Histograms and density plots for numeric features")
    print("• categorical_distributions.png - Bar charts for categorical features")  
    print("• distribution_shift_summary.png - Summary dashboard with significance rankings")
    print("• distribution_shift_report.txt - Detailed text report with recommendations")

if __name__ == "__main__":
    main()