import pandas as pd
from datasets import Dataset, DatasetDict


def prepare_dataset(data_path, split_ratio=0.8):
    """
    Load a CSV dataset and split it into train and test subsets.
    :param data_path: Path to the dataset CSV file.
    :param split_ratio: Proportion of data to use for training.
    :return: A DatasetDict with 'train' and 'test' splits.
    """
    # Load dataset using pandas
    df = pd.read_csv(data_path)

    # Check required columns
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Dataset must contain 'input' and 'output' columns. Found: {df.columns}")

    # Drop rows with missing values
    if df.isnull().any().any():
        print("Found missing values. Cleaning dataset...")
        df = df.dropna(subset=["input", "output"])
        print(f"Cleaned dataset now has {len(df)} rows.")

    # Split dataset
    train_size = int(len(df) * split_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return DatasetDict({"train": train_dataset, "test": test_dataset})
