# src/fuzzy_grouping.py

from rapidfuzz import process, fuzz
import pandas as pd


def build_merchant_mapping(descriptions, threshold=85, scorer=fuzz.token_set_ratio):
    """
    Build a mapping of cleaned descriptions to canonical merchant names using fuzzy matching.

    Parameters:
        descriptions (iterable): Unique cleaned descriptions
        threshold (int): Similarity threshold for grouping
        scorer (function): RapidFuzz scoring function

    Returns:
        dict: Mapping {original_description: canonical_merchant}
    """
    merchants = []
    mapping = {}

    for desc in descriptions:
        if not desc:
            mapping[desc] = desc
            continue

        # First merchant becomes initial cluster
        if not merchants:
            merchants.append(desc)
            mapping[desc] = desc
            continue

        # Find best existing match
        match, score, _ = process.extractOne(
            desc,
            merchants,
            scorer=scorer
        )

        if score >= threshold:
            mapping[desc] = match
        else:
            merchants.append(desc)
            mapping[desc] = desc

    return mapping


def apply_fuzzy_grouping(df: pd.DataFrame, threshold=85) -> pd.DataFrame:
    """
    Apply fuzzy grouping to a dataframe with a 'desc_clean' column.

    Parameters:
        df (pd.DataFrame): Input dataframe with cleaned descriptions
        threshold (int): Similarity threshold

    Returns:
        pd.DataFrame: Dataframe with added 'merchant' column
    """
    if "desc_clean" not in df.columns:
        raise ValueError("DataFrame must contain 'desc_clean' column.")

    unique_descriptions = df["desc_clean"].dropna().unique()

    merchant_mapping = build_merchant_mapping(
        unique_descriptions,
        threshold=threshold
    )

    df["merchant"] = df["desc_clean"].map(merchant_mapping)

    return df


def simplify_merchant_names(df: pd.DataFrame, max_words=2) -> pd.DataFrame:
    """
    Create a simplified merchant label for reporting.

    Parameters:
        df (pd.DataFrame): Input dataframe with 'merchant' column
        max_words (int): Number of words to retain

    Returns:
        pd.DataFrame: Dataframe with 'merchant_clean' column
    """
    if "merchant" not in df.columns:
        raise ValueError("DataFrame must contain 'merchant' column.")

    def simplify(name):
        if not isinstance(name, str):
            return name
        words = name.split()
        return " ".join(words[:max_words])

    df["merchant_clean"] = df["merchant"].apply(simplify)

    return df


def group_merchants(df: pd.DataFrame, threshold=85, max_words=2) -> pd.DataFrame:
    """
    Full pipeline to group and simplify merchants.

    Parameters:
        df (pd.DataFrame): Input dataframe
        threshold (int): Fuzzy matching threshold
        max_words (int): Words retained in simplified label

    Returns:
        pd.DataFrame: Dataframe with 'merchant' and 'merchant_clean'
    """
    df = apply_fuzzy_grouping(df, threshold=threshold)
    df = simplify_merchant_names(df, max_words=max_words)

    return df