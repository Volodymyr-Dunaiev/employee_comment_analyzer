# Excel I/O utilities for reading and writing Excel files.
# Provides chunked reading for memory-efficient processing of large files.

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def read_excel_in_chunks(file_path, chunk_size=1000):
    # Read an Excel file in chunks for memory efficiency.
    #
    # Args:
    #     file_path: Path to the Excel file
    #     chunk_size: Number of rows to read per chunk (default: 1000)
    #
    # Yields:
    #     DataFrame: Chunks of the Excel file
    
    df = pd.read_excel(file_path)
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def write_excel(df, output_path):
    """Write a DataFrame to an Excel file with expanded category columns.
    
    If the DataFrame contains a 'Predicted_Categories' column with lists,
    this function will expand it into separate columns with category names as values.
    
    Args:
        df: DataFrame to write
        output_path: Path where the Excel file will be saved
    """
    
    # Check if we need to expand the Predicted_Categories column
    if 'Predicted_Categories' in df.columns:
        # Find the maximum number of categories any row has
        max_categories = max(
            len(pred_list) if isinstance(pred_list, list) else 0
            for pred_list in df['Predicted_Categories']
        )
        
        # Create columns: Category 1, Category 2, etc.
        for i in range(max_categories):
            col_name = f'Category {i + 1}'
            df[col_name] = df['Predicted_Categories'].apply(
                lambda x: x[i] if isinstance(x, list) and len(x) > i else ''
            )
        
        # Drop the original list column
        df = df.drop(columns=['Predicted_Categories'])
    
    df.to_excel(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
