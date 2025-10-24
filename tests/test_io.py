# Tests for Excel I/O functionality.

import pandas as pd
from src.io.excel_io import write_excel, read_excel_in_chunks

def test_excel_io(tmp_path):
    # Test that we can write and read Excel files correctly.
    df = pd.DataFrame({'text': ['test']})
    file_path = tmp_path / 'test.xlsx'
    write_excel(df, file_path)
    chunks = list(read_excel_in_chunks(file_path))
    assert len(chunks[0]) == 1
