import pandas as pd

# Load data
df = pd.read_csv('swiggy.csv')

# Convert '--' in 'rating' to NaN (so dropna() can detect it)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Remove '₹' from 'cost' and convert to float (then drop NaN)
df['cost'] = df['cost'].str.replace('₹', '').str.strip().astype(float)

# Extract numbers from 'rating_count' (e.g., "50+ ratings" → 50)
df['rating_count'] = df['rating_count'].str.extract(r'(\d+)').astype(float)

# DROP ROWS WITH MISSING VALUES (instead of imputing)
df_cleaned = df.dropna(subset=['rating', 'cost', 'rating_count', 'cuisine'])

# Save cleaned data
df_cleaned.to_csv('cleaned_data_dropped_missing.csv', index=False)

# Verify
print(f"Original rows: {len(df)}")
print(f"Rows after dropping missing values: {len(df_cleaned)}")
print("\nSample cleaned data:")
print(df_cleaned.head())