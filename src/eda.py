import pandas as pd


def spending_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total spend by category.
    """
    return (
        df.groupby("category")["amount"]
        .sum()
        .reset_index(name="total_spend")
        .sort_values("total_spend", ascending=False)
    )


def monthly_category_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly spending by category.
    """
    return (
        df.groupby(["month", "category"])["amount"]
        .sum()
        .reset_index()
    )


def average_monthly_spending_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average monthly spend by category.
    """
    monthly = monthly_category_spending(df)

    return (
        monthly.groupby("category")["amount"]
        .mean()
        .reset_index(name="avg_monthly_spend")
        .sort_values("avg_monthly_spend", ascending=False)
    )


def spending_by_merchant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total spend by merchant.
    """
    return (
        df.groupby("merchant_clean")["amount"]
        .sum()
        .reset_index(name="total_spend")
        .sort_values("total_spend", ascending=False)
    )


def monthly_merchant_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly merchant spending.
    """
    return (
        df.groupby(["month", "merchant_clean"])["amount"]
        .sum()
        .reset_index()
    )


def average_monthly_spending_by_merchant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average monthly spend by merchant.
    """
    monthly = monthly_merchant_spending(df)

    return (
        monthly.groupby("merchant_clean")["amount"]
        .mean()
        .reset_index(name="avg_monthly_spend")
        .sort_values("avg_monthly_spend", ascending=False)
    )


def monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total spending by calendar month.
    """
    return (
        df.groupby("month")["amount"]
        .sum()
        .reset_index(name="monthly_spend")
    )


def average_monthly_spending(df: pd.DataFrame) -> float:
    """
    Average spending per month.
    """
    monthly = monthly_spending(df)

    return monthly["monthly_spend"].mean()