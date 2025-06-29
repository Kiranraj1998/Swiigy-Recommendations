import pandas as pd

# Load datasets
cleaned_df = pd.read_csv("cleaned_data.csv", low_memory=False)
encoded_df = pd.read_csv("encoded8_data.csv", low_memory=False)

# Extract the 'id' column from both
cleaned_ids = cleaned_df['id'].reset_index(drop=True)
encoded_ids = encoded_df['id'].reset_index(drop=True)

# Check if they match exactly
if cleaned_ids.equals(encoded_ids):
    print("✅ The 'id' columns match perfectly. Row order is preserved.")
else:
    print("❌ The 'id' columns do NOT match. Something has gone out of sync.")

# (Optional) Print the first few mismatches
mismatches = cleaned_ids[cleaned_ids != encoded_ids]
if not mismatches.empty:
    print("\n🔍 Sample mismatches:")
    print(pd.DataFrame({
        'cleaned_id': cleaned_ids[mismatches.index],
        'encoded_id': encoded_ids[mismatches.index]
    }).head())
