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


def average_monthly_spending_by_category(df):
    """
    Average monthly spend by category including months
    with zero spending.
    """

    monthly = (
        df.groupby(["month", "category"])["amount"]
        .sum()
        .reset_index()
    )

    months = sorted(df["month"].unique())
    categories = sorted(df["category"].unique())

    full_index = pd.MultiIndex.from_product(
        [months, categories],
        names=["month", "category"]
    )

    monthly_complete = (
        monthly
        .set_index(["month", "category"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return (
        monthly_complete
        .groupby("category")["amount"]
        .mean()
        .reset_index(name="avg_monthly_spend")
        .sort_values(
            "avg_monthly_spend",
            ascending=False
        )
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


def average_monthly_spending_by_merchant(df):
    """
    Average monthly spend by merchant across ALL months
    in the analysis period, including months with $0 spend.
    """

    monthly = (
        df.groupby(["month", "merchant_clean"])["amount"]
        .sum()
        .reset_index()
    )

    months = sorted(df["month"].unique())
    merchants = sorted(df["merchant_clean"].unique())

    # Create complete Month × Merchant grid
    full_index = pd.MultiIndex.from_product(
        [months, merchants],
        names=["month", "merchant_clean"]
    )

    monthly_complete = (
        monthly
        .set_index(["month", "merchant_clean"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return (
        monthly_complete
        .groupby("merchant_clean")["amount"]
        .mean()
        .reset_index(name="avg_monthly_spend")
        .sort_values(
            "avg_monthly_spend",
            ascending=False
        )
    )


def monthly_spending(df):

    monthly = (
        df.groupby("month")["amount"]
        .sum()
        .reset_index(name="monthly_spend")
    )

    monthly["month"] = (
        monthly["month"]
        .dt.to_timestamp()
    )

    return monthly

def average_monthly_spending(df: pd.DataFrame) -> float:
    """
    Average spending per month.
    """
    monthly = monthly_spending(df)

    return monthly["monthly_spend"].mean()