# Import the function that loads a local Parquet dataset split
from app.data_loader import load_split_local

# Load the validation split from a Parquet file into a list of dictionaries
rows = load_split_local("/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet")

#  total number of rows loaded
#  the dictionary keys of the first row (e.g., 'question', 'choices', 'answer')
#  the correct answer for the first row
print(len(rows), rows[0].keys(), rows[3]["answer"])

# Run: python3 -m tests_scripts.test_data_loader
# row 3 should be 2