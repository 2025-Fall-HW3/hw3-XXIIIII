import pandas as pd
import numpy as np
from Markowitz import RiskParityPortfolio

# Load the expected answer
answer_df = pd.read_pickle("./Answer/rp.pkl")

# Get your current implementation's output
your_df = RiskParityPortfolio("SPY").get_results()[0]

print("=" * 80)
print("SHAPE COMPARISON")
print("=" * 80)
print(f"Answer shape: {answer_df.shape}")
print(f"Your shape:   {your_df.shape}")
print()

print("=" * 80)
print("INDEX COMPARISON")
print("=" * 80)
print(f"Indices equal: {answer_df.index.equals(your_df.index)}")
if not answer_df.index.equals(your_df.index):
    print(f"First answer index: {answer_df.index[0]}")
    print(f"First your index:   {your_df.index[0]}")
print()

print("=" * 80)
print("COLUMNS COMPARISON")
print("=" * 80)
print(f"Columns equal: {answer_df.columns.equals(your_df.columns)}")
print()

print("=" * 80)
print("FIRST NON-ZERO WEIGHTS COMPARISON")
print("=" * 80)
# Find first row with non-zero weights in answer
answer_nonzero_idx = (answer_df != 0).any(axis=1).idxmax()
your_nonzero_idx = (your_df != 0).any(axis=1).idxmax()

print(f"Answer first nonzero at index: {answer_nonzero_idx}")
print(f"Your first nonzero at index:   {your_nonzero_idx}")
print()

print("Answer weights at first nonzero:")
print(answer_df.loc[answer_nonzero_idx])
print()
print("Your weights at first nonzero:")
print(your_df.loc[your_nonzero_idx])
print()

print("=" * 80)
print("VALUE DIFFERENCES (first 10 rows with differences)")
print("=" * 80)
diff_mask = ~np.isclose(answer_df, your_df, atol=0.01)
rows_with_diff = diff_mask.any(axis=1)
if rows_with_diff.any():
    diff_rows = answer_df[rows_with_diff].head(10)
    for idx in diff_rows.index:
        print(f"\nDate: {idx}")
        for col in answer_df.columns:
            ans_val = answer_df.loc[idx, col]
            your_val = your_df.loc[idx, col]
            if not np.isclose(ans_val, your_val, atol=0.01):
                print(f"  {col}: Answer={ans_val:.6f}, Yours={your_val:.6f}, Diff={abs(ans_val-your_val):.6f}")
else:
    print("No differences found!")