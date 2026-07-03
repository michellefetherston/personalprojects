# src/cleaning.py

import pandas as pd
import re


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load transaction data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to datetime.
    """
    if "trans_date" in df.columns:
        df["trans_date"] = pd.to_datetime(df["trans_date"], errors="coerce")

    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

    return df


def clean_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove currency symbols and convert amount to float.
    """
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .replace(r"[\$,]", "", regex=True)
            .astype(float)
        )

    return df


def clean_description(text: str) -> str:
    """
    Normalize transaction description text.
    """
    if pd.isna(text):
        return ""

    text = text.upper()
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = re.sub(r'(WI|IL|MN|MI|USA|SQ|TST|LLC)', '', text)  # remove select noise tokens
    text = re.sub(r"[^A-Z\s]", "", text)                  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()              # normalize whitespace

    return text


def apply_description_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply description cleaning to dataframe.
    """
    if "description" in df.columns:
        df["desc_clean"] = df["description"].apply(clean_description)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for analysis.
    """
    if "trans_date" in df.columns:
        df["month"] = df["trans_date"].dt.to_period("M")
        df["day_of_week"] = df["trans_date"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    return df


def clean_transactions(filepath: str) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    """
    df = load_data(filepath)
    df = standardize_column_names(df)
    df = convert_dates(df)
    df = clean_amount(df)
    df = apply_description_cleaning(df)
    df = add_time_features(df)

    return df