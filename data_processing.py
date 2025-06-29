import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

def optimized_encoding(input_file='cleaned_data.csv', 
                     parquet_file='encoded_data.parquet',
                     csv_file='encoded_data.csv'):
    """
    1. Performs one-hot encoding efficiently
    2. Saves to Parquet (small file size)
    3. Optional conversion to CSV
    """
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    categorical_cols = ['city', 'cuisine']
    
    # Fit encoder
    print("Fitting encoder...")
    encoder = OneHotEncoder(dtype='uint8', handle_unknown='ignore', sparse_output=False)
    encoder.fit(df[categorical_cols])
    
    # Save encoder
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    # Transform data
    print("Encoding data...")
    encoded_data = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, 
                            columns=encoder.get_feature_names_out(categorical_cols),
                            dtype='uint8')
    
    # Combine with numerical data
    final_df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    
    # Save as Parquet (highly compressed)
    print(f"Saving as Parquet ({parquet_file})...")
    final_df.to_parquet(parquet_file, engine='pyarrow')
    
    # Convert to CSV if needed
    if csv_file:
        print(f"Converting to CSV ({csv_file})...")
        pd.read_parquet(parquet_file).to_csv(csv_file, index=False)
    
    # Report sizes
    orig_size = os.path.getsize(input_file) / (1024*1024)
    parquet_size = os.path.getsize(parquet_file) / (1024*1024)
    
    if csv_file:
        csv_size = os.path.getsize(csv_file) / (1024*1024)
        print(f"\nFile sizes:\nOriginal: {orig_size:.1f} MB\nParquet: {parquet_size:.1f} MB\nCSV: {csv_size:.1f} MB")
    else:
        print(f"\nFile sizes:\nOriginal: {orig_size:.1f} MB\nParquet: {parquet_size:.1f} MB")

if __name__ == '__main__':
    # Run with CSV conversion (set csv_file=None to skip)
    optimized_encoding(csv_file='encoded_data.csv')