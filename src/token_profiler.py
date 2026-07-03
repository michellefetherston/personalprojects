
import pandas as pd
import re
from collections import Counter


def tokenize_descriptions(df: pd.DataFrame, column="description") -> pd.Series:
    """
    Tokenize text column into lists of words.
    """
    cleaned = (
        df[column]
        .str.upper()
        .str.replace(r"[^A-Z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return cleaned.str.split()


def compute_token_frequencies(tokens_series: pd.Series) -> pd.DataFrame:
    """
    Compute frequency of all tokens.
    """
    all_tokens = [token for row in tokens_series if isinstance(row, list) for token in row]
    counts = Counter(all_tokens)

    return pd.DataFrame(counts.items(), columns=["token", "count"]).sort_values(
        by="count", ascending=False
    )


def suggest_noise_tokens(
    df: pd.DataFrame,
    column="description",
    max_token_length=4,
    min_frequency=10,
    top_n=100
) -> pd.DataFrame:
    """
    Identify candidate noise tokens based on length and frequency.

    Parameters:
        df (DataFrame): input data
        column (str): text column to analyze
        max_token_length (int): max token length to consider noise
        min_frequency (int): minimum frequency threshold
        top_n (int): limit output size

    Returns:
        DataFrame: candidate noise tokens
    """
    tokens = tokenize_descriptions(df, column)
    token_freq = compute_token_frequencies(tokens)

    candidates = token_freq[
        (token_freq["token"].str.len() <= max_token_length) &
        (token_freq["count"] >= min_frequency)
    ]

    return candidates.head(top_n)


def get_token_examples(
    df: pd.DataFrame,
    token: str,
    column="description",
    n=5
) -> pd.Series:
    """
    Return example rows containing a given token.
    """
    return df[df[column].str.contains(token, case=False, na=False)][column].head(n)


def build_noise_token_list(
    df: pd.DataFrame,
    column="description",
    max_token_length=4,
    min_frequency=10
) -> set:
    """
    Generate a set of suggested noise tokens.
    """
    candidates = suggest_noise_tokens(
        df,
        column=column,
        max_token_length=max_token_length,
        min_frequency=min_frequency
    )

    return set(candidates["token"].tolist())