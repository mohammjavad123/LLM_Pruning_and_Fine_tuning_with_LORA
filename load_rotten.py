from datasets import load_dataset
import pandas as pd

# Download a sample from the validation split (version 3.0.0)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")

# Get a small sample for fast testing
sample_size = 2000
subset = dataset.shuffle(seed=42).select(range(sample_size))

# Make a DataFrame
df = pd.DataFrame({
    "document": subset["article"],
    "summary": subset["highlights"]
})
df.to_csv("cnn_dailymail_sample.csv", index=False)
print(df.head(3))
print(f"✅ Saved {len(df)} examples to cnn_dailymail_sample.csv")
