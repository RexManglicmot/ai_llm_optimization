# Import type hints for better code clarity and static checking
from typing import List, Dict

import pandas as pd

# Create a function
def load_split_local(parquet_path: str) -> List[Dict]:
    """
    Load a local Parquet dataset split (dev / validation / test) 
    and normalize it into a consistent list of dictionaries.

    Args:
        parquet_path (str): Path to the Parquet file.

    Returns:
        List[Dict]: A list where each dict has the format:
            {
                "question": str,          # The question text
                "choices": [A, B, C, D],  # List of 4 possible answers
                "answer": "A"|"B"|"C"|"D" # Correct choice letter
            }
    """
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_path)

    # Transform each row into a standardized dictionary format
    # This is a zip combined with a list comprehension

    # First evaluate the iterable (the zip(...)) because before the loop even starts, Python has to know what it’s going to loop over.
    # That means it first runs: zip(df["question"], df["choices"], df["answer"]) --> creates a zip object (iterator)
    # Then goes into for loop
    # Again, the for appears first when you read it, but Python must evaluate zip(...) first because it needs that iterable to actually run the for loop.

    rows = [
        {
            "question": q,         # The question text
            "choices": list(c),    # Convert choices to a list
            "answer": a            # The correct choice (A/B/C/D)
        }
        for q, c, a in zip(df["question"], df["choices"], df["answer"])
    ]

    # not breaking Python’s rules — you just have two layers of evaluation happening:
    # One at the statement/control level (outer → inner)
    # One at the expression level (inner → outer)
    
    # Control flow: outer to inner
        # assert runs
        # Calls all(...)
        # Iterates generator
    # Expression evaluation inside each iteration: inner to outer
        # r["choices"]
        # len(...)
        # Compare == 4

    # Ensure every question has exactly 4 choices
    assert all(len(r["choices"]) == 4 for r in rows), \
        "Each row must have 4 choices."

    # Ensure every answer is one of the allowed labels
    # answers should be ints 0..3 (A,B,C,D)
    assert all(isinstance(r["answer"], int) and 0 <= r["answer"] <= 3 for r in rows), \
       "Answer index must be an int in {0,1,2,3} (maps to A,B,C,D)."
    # In original code, it is 0, 1, 2, 3 which correspond to A, B, C, D

    # Return the processed list of dictionaries
    return rows

# TO DO: convert model's letter to index before comparing
# LETTER2IDX = {"A":0, "B":1, "C":2, "D":3}
# pred_letter = parse_letter(first_line)           # e.g., "B"
# pred_idx = LETTER2IDX.get(pred_letter, -1)       # -1 if parsing failed
# correct = (pred_idx == r["answer"])              # r["answer"] is 0..3