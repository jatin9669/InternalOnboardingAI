import pandas as pd
from datetime import datetime
import os

def write_to_excel(input_tokens, output_tokens, total_tokens, total_cost, type):

    # Prepare data for Excel
    data = {
        'Timestamp': [datetime.now()],
        'Type': [type],
        'Input Tokens': [input_tokens],
        'Output Tokens': [output_tokens],
        'Total Tokens': [total_tokens],
        'Cost ($)': [round(total_cost, 8)]
    }

    # Create or update Excel file
    excel_file = 'token_usage_billing.xlsx'

    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        new_df = pd.DataFrame(data)

        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = pd.DataFrame(data)

    # Write to Excel
    updated_df.to_excel(excel_file, index=False)