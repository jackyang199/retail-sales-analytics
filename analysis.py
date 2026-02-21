"""
Retail Sales Data Analysis
==========================

A comprehensive data analysis project demonstrating end-to-end retail analytics.
This script performs data generation, cleaning, analysis, and visualization.

Author: Yanlei Yang
Location: Perth, Australia
Role: Data Analyst
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def generate_sample_data(n_orders: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic retail sales data for Australian market.
    
    Args:
        n_orders: Number of orders to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sales data
    """
    np.random.seed(seed)
    
    # Generate dates throughout 2024
    dates = pd.date_range('2024-01-01', '2024-12-31', periods=n_orders)
    
    # Define Australian cities with realistic distribution
    cities = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']
    city_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    category_weights = [0.25, 0.30, 0.20, 0.15, 0.10]
    
    # Products per category
    products = {
        'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smart Watch'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes'],
        'Home & Garden': ['Furniture', 'Kitchenware', 'Bedding', 'Decor', 'Tools'],
        'Sports': ['Bike', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Running Shoes'],
        'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Cookbook', 'Biography']
    }
    
    # Customer segments
    segments = ['Premium', 'Regular', 'New', 'VIP']
    segment_weights = [0.20, 0.45, 0.20, 0.15]
    
    # Payment methods
    payments = ['Credit Card', 'PayPal', 'Bank Transfer', 'Cash']
    payment_weights = [0.40, 0.30, 0.20, 0.10]
    
    # Build data dictionary
    data = {
        'Order_ID': [f'ORD-{i:05d}' for i in range(1, n_orders + 1)],
        'Order_Date': dates,
        'Customer_ID': np.random.randint(1001, 1100, n_orders),
        'Product_Category': np.random.choice(categories, n_orders, p=category_weights),
        'Product': [],
        'Quantity': np.random.randint(1, 10, n_orders),
        'Unit_Price': np.round(np.random.uniform(10, 500, n_orders), 2),
        'Region': np.random.choice(cities, n_orders, p=city_weights),
        'Customer_Segment': np.random.choice(segments, n_orders, p=segment_weights),
        'Payment_Method': np.random.choice(payments, n_orders, p=payment_weights)
    }
    
    # Assign products based on category
    for cat in categories:
        cat_mask = np.array(data['Product_Category']) == cat
        n_cat = sum(cat_mask)
        data['Product'].extend(np.random.choice(products[cat], n_cat))
    
    # Shuffle products to match order
    indices = np.random.permutation(n_orders)
    data['Product'] = [data['Product'][i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate derived fields
    df['Total_Sales'] = df['Quantity'] * df['Unit_Price']
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Month_Name'] = df['Order_Date'].dt.month_name()
    df['Quarter'] = df['Order_Date'].dt.quarter
    df['DayOfWeek'] = df['Order_Date'].dt.day_name()
    
    return df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    print("=" * 60)
    print("STEP 1: DATA CLEANING")
    print("=" * 60)
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0] if missing.sum() > 0 else 'No missing values found'}")
    
    # Check data types
    print(f"\nData types:\n{df.dtypes}")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\nDuplicates removed: {before - after}")
    
    # Ensure positive values
    df = df[df['Quantity'] > 0]
    df = df[df['Unit_Price'] > 0]
    df = df[df['Total_Sales'] > 0]
    
    print(f"\nRecords after cleaning: {len(df)}")
    
    return df


def descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Calculate descriptive statistics.
    
    Args:
        df: Cleaned DataFrame
    """
    print("\n" + "=" * 60)
    print("STEP 2: DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Key metrics
    total_sales = df['Total_Sales'].sum()
    avg_order = df['Total_Sales'].mean()
    median_order = df['Total_Sales'].median()
    unique_customers = df['Customer_ID'].nunique()
    total_orders = len(df)
    
    print(f"\n{'KEY METRICS':-^40}")
    print(f"Total Sales:         ${total_sales:,.2f}")
    print(f"Average Order:      ${avg_order:,.2f}")
    print(f"Median Order:        ${median_order:,.2f}")
    print(f"Unique Customers:   {unique_customers}")
    print(f"Total Orders:        {total_orders}")
    
    # Product category analysis
    print(f"\n{'SALES BY CATEGORY':-^40}")
    category_sales = df.groupby('Product_Category')['Total_Sales'].agg(['sum', 'mean', 'count'])
    category_sales = category_sales.sort_values('sum', ascending=False)
    print(category_sales.to_string())
    
    # Regional analysis
    print(f"\n{'SALES BY REGION':-^40}")
    region_sales = df.groupby('Region')['Total_Sales'].agg(['sum', 'mean', 'count'])
    region_sales = region_sales.sort_values('sum', ascending=False)
    print(region_sales.to_string())


def key_insights(df: pd.DataFrame) -> None:
    """
    Extract key business insights.
    
    Args:
        df: DataFrame
    """
    print("\n" + "=" * 60)
    print("STEP 3: KEY INSIGHTS")
    print("=" * 60)
    
    # Best performing categories
    best_category = df.groupby('Product_Category')['Total_Sales'].sum().idxmax()
    best_category_sales = df.groupby('Product_Category')['Total_Sales'].sum().max()
    
    # Top region
    top_region = df.groupby('Region')['Total_Sales'].sum().idxmax()
    top_region_sales = df.groupby('Region')['Total_Sales'].sum().max()
    
    # Top product
    top_product = df.groupby('Product')['Total_Sales'].sum().idxmax()
    top_product_sales = df.groupby('Product')['Total_Sales'].sum().max()
    
    # Customer segments
    top_segment = df.groupby('Customer_Segment')['Total_Sales'].sum().idxmax()
    
    print(f"\nBest Performing Category: {best_category} (${best_category_sales:,.2f})")
    print(f"Top Regional Market:      {top_region} (${top_region_sales:,.2f})")
    print(f"Top Product:             {top_product} (${top_product_sales:,.2f})")
    print(f"Leading Segment:        {top_segment}")
    
    # Quarterly performance
    print(f"\n{'QUARTERLY PERFORMANCE':-^40}")
    quarterly = df.groupby('Quarter')['Total_Sales'].sum()
    for q, sales in quarterly.items():
        print(f"Q{q}: ${sales:,.2f}")


def create_visualizations(df: pd.DataFrame, output_dir: str = 'charts') -> None:
    """
    Create visualizations for the analysis.
    
    Args:
        df: DataFrame
        output_dir: Directory to save charts
    """
    print("\n" + "=" * 60)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Retail Sales Analytics Dashboard', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#95C623', '#6B2D5C']
    
    # 1. Monthly Sales Trend
    monthly = df.groupby('Month')['Total_Sales'].sum()
    axes[0, 0].plot(monthly.index, monthly.values, marker='o', color='#2E86AB', linewidth=2)
    axes[0, 0].set_title('Monthly Sales Trend', fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Total Sales ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sales by Category (Pie Chart)
    category_sales = df.groupby('Product_Category')['Total_Sales'].sum()
    axes[0, 1].pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', 
                    startangle=90, colors=colors)
    axes[0, 1].set_title('Sales by Category', fontweight='bold')
    
    # 3. Regional Sales (Horizontal Bar)
    region_sales = df.groupby('Region')['Total_Sales'].sum().sort_values()
    axes[0, 2].barh(region_sales.index, region_sales.values, color='#A23B72')
    axes[0, 2].set_title('Sales by Region', fontweight='bold')
    axes[0, 2].set_xlabel('Total Sales ($)')
    
    # 4. Customer Segments
    segment_sales = df.groupby('Customer_Segment')['Total_Sales'].sum()
    axes[1, 0].bar(segment_sales.index, segment_sales.values, color=colors[:4])
    axes[1, 0].set_title('Sales by Customer Segment', fontweight='bold')
    axes[1, 0].set_ylabel('Total Sales ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Top Products
    top_products = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False).head(7)
    axes[1, 1].barh(top_products.index[::-1], top_products.values[::-1], color='#6B2D5C')
    axes[1, 1].set_title('Top 7 Products', fontweight='bold')
    axes[1, 1].set_xlabel('Total Sales ($)')
    
    # 6. Payment Methods
    payment_sales = df.groupby('Payment_Method')['Total_Sales'].sum()
    axes[1, 2].pie(payment_sales, labels=payment_sales.index, autopct='%1.1f%%',
                   startangle=45, colors=['#2E86AB', '#A23B72', '#F18F01', '#95C623'])
    axes[1, 2].set_title('Payment Methods', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'dashboard_charts.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nDashboard saved to: {output_path}")
    
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("RETAIL SALES DATA ANALYSIS")
    print("Australian Market Analytics")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: Yanlei Yang - Data Analyst")
    
    # Generate sample data
    print("\n[1/5] Generating sample retail data...")
    df = generate_sample_data(n_orders=1000)
    print(f"Generated {len(df)} records")
    
    # Clean data
    print("\n[2/5] Cleaning data...")
    df = data_cleaning(df)
    
    # Descriptive statistics
    print("\n[3/5] Calculating statistics...")
    descriptive_statistics(df)
    
    # Key insights
    print("\n[4/5] Extracting insights...")
    key_insights(df)
    
    # Visualizations
    print("\n[5/5] Creating visualizations...")
    create_visualizations(df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - charts/dashboard_charts.png")
    print("\nKey takeaways ready for executive presentation.")


if __name__ == "__main__":
    main()
