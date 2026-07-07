
import pandas as pd
import re


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load transaction data.
    """
    return pd.read_csv(filepath)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def remove_payments_and_credits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove payments/credits from spending analysis.
    """

    if "category" not in df.columns:
        return df

    excluded_categories = {
        "Payments and Credits"
    }

    return (
        df.loc[
            ~df["category"].isin(excluded_categories)
        ]
        .copy()
    )


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns.
    """

    if "trans_date" in df.columns:
        df["trans_date"] = pd.to_datetime(
            df["trans_date"],
            errors="coerce"
        )

    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(
            df["post_date"],
            errors="coerce"
        )

    return df


def clean_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert amounts to numeric.

    Handles:
        $123.45
        -123.45
        (123.45)
    """

    amount = (
        df["amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )

    df["amount"] = pd.to_numeric(
        amount,
        errors="coerce"
    )

    return df


def clean_description(text: str) -> str:
    """
    Light cleaning only.

    Preserve words.
    Preserve state abbreviations.
    Preserve merchant signals.
    """

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def apply_description_cleaning(
    df: pd.DataFrame
) -> pd.DataFrame:

    df["desc_clean"] = (
        df["description"]
        .astype(str)
        .apply(clean_description)
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month and date features.
    """

    df["month"] = (
        df["trans_date"]
        .dt.to_period("M")
    )

    df["day_of_week"] = (
        df["trans_date"]
        .dt.day_name()
    )

    df["is_weekend"] = (
        df["day_of_week"]
        .isin(["Saturday", "Sunday"])
    )

    return df


def clean_transactions(filepath: str) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    """

    df = load_data(filepath)

    df = standardize_column_names(df)

    df = remove_payments_and_credits(df)

    df = convert_dates(df)

    df = clean_amount(df)

    df = apply_description_cleaning(df)

    df = add_time_features(df)

    return df