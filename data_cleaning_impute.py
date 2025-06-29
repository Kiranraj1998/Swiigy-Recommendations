import pandas as pd
import numpy as np

def load_and_clean_data():
    """Load and clean the restaurant data."""
    # Load raw data
    df = pd.read_csv('swiggy.csv')
    
    # Data understanding - print basic info
    print("=== Original Data Info ===")
    print(df.info())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Duplicate Rows ===")
    print(f"Number of duplicates: {df.duplicated().sum()}")
    
    # Data cleaning
    # 1. Remove duplicate rows
    df = df.drop_duplicates()
    
    # 2. Handle missing values
    # Handle cost column - remove currency symbol and convert to float
    df['cost'] = df['cost'].str.replace('[₹,â‚¹]', '', regex=True).str.strip()
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    
    # Handle rating column - convert to numeric, treating '--' as NaN
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Handle rating_count - extract numbers from strings like "50+ ratings"
    df['rating_count'] = df['rating_count'].str.extract(r'(\d+)').astype(float)
    
    # Fill numerical columns with median
    numerical_cols = ['rating', 'rating_count', 'cost']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns - fill with 'Unknown'
    categorical_cols = ['name', 'city', 'cuisine', 'lic_no', 'address']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Verify cleaning
    print("\n=== After Cleaning ===")
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    print(f"\nNumber of duplicates after cleaning: {df.duplicated().sum()}")
    print("\nData types after cleaning:")
    print(df.dtypes)
    print("\nSample cleaned data:")
    print(df.head())
    
    # Save cleaned data
    df.to_csv('cleaned_data.csv', index=False)
    print(f"\nCleaned data saved to cleaned_data.csv")
    
    return df

if __name__ == '__main__':
    cleaned_data = load_and_clean_data()