import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")


# ======================================================
# Average Monthly Spend by Merchant
# ======================================================

def plot_avg_monthly_spend_by_merchant(
    merchant_summary,
    top_n=20,
    figsize=(12, 8)
):

    plot_df = (
        merchant_summary
        .sort_values(
            "avg_monthly_spend",
            ascending=False
        )
        .head(top_n)
    )

    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        y="merchant_clean",
        x="avg_monthly_spend",
        palette="Blues_r"
    )

    plt.title(
        f"Top {top_n} Merchants by Average Monthly Spend"
    )

    plt.xlabel("Average Monthly Spend ($)")
    plt.ylabel("")

    plt.tight_layout()
    plt.show()


# ======================================================
# Monthly Spending Trend
# ======================================================

def plot_monthly_spending(
    monthly_totals,
    figsize=(12, 6)
):

    plot_df = monthly_totals.copy()

    if str(plot_df["month"].dtype).startswith("period"):

        plot_df["month"] = (
            plot_df["month"]
            .dt.to_timestamp()
        )

    plt.figure(figsize=figsize)

    sns.lineplot(
        data=plot_df,
        x="month",
        y="monthly_spend",
        marker="o"
    )

    plt.title(
        "Monthly Spending Trend"
    )

    plt.xlabel("Month")
    plt.ylabel("Spend ($)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ======================================================
# Average Spend by Month Of Year
# ======================================================

def plot_average_spend_by_month(
    analysis_df,
    figsize=(12, 6)
):

    plot_df = analysis_df.copy()

    plot_df["month_name"] = (
        plot_df["trans_date"]
        .dt.month_name()
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    monthly_avg = (
        plot_df
        .groupby(
            [
                plot_df["trans_date"].dt.month,
                "month_name"
            ]
        )["amount"]
        .sum()
        .reset_index()
        .groupby("month_name")["amount"]
        .mean()
        .reindex(month_order)
        .reset_index()
    )

    plt.figure(figsize=figsize)

    sns.barplot(
        data=monthly_avg,
        x="month_name",
        y="amount",
        palette="Greens_r"
    )

    plt.title(
        "Average Spend by Month of Year"
    )

    plt.xlabel("")
    plt.ylabel("Average Spend ($)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ======================================================
# Top Merchants - Most Recent 12 Months
# ======================================================

def plot_top_merchants_last_12_months(
    analysis_df,
    top_n=20,
    figsize=(12, 8)
):

    latest_month = (
        analysis_df["trans_date"]
        .max()
    )

    cutoff = (
        latest_month
        - pd.DateOffset(months=12)
    )

    recent_df = (
        analysis_df[
            analysis_df["trans_date"] >= cutoff
        ]
    )

    merchant_spend = (
        recent_df
        .groupby("merchant_clean")["amount"]
        .sum()
        .reset_index(
            name="total_spend"
        )
        .sort_values(
            "total_spend",
            ascending=False
        )
        .head(top_n)
    )

    plt.figure(figsize=figsize)

    sns.barplot(
        data=merchant_spend,
        y="merchant_clean",
        x="total_spend",
        palette="Oranges_r"
    )

    plt.title(
        f"Top {top_n} Merchants by Spend (Most Recent 12 Months)"
    )

    plt.xlabel("Total Spend ($)")
    plt.ylabel("")

    plt.tight_layout()
    plt.show()


# ======================================================
# Top Merchants by Category
# ======================================================

def plot_top_merchants_by_category(
    analysis_df,
    top_n=20,
    figsize=(12, 8)
):

    categories = sorted(
        analysis_df["category"]
        .dropna()
        .unique()
    )

    for category in categories:

        plot_df = (
            analysis_df[
                analysis_df["category"] == category
            ]
            .groupby("merchant_clean")["amount"]
            .sum()
            .reset_index(
                name="total_spend"
            )
            .sort_values(
                "total_spend",
                ascending=False
            )
            .head(top_n)
        )

        plt.figure(figsize=figsize)

        sns.barplot(
                data=plot_df,
            y="merchant_clean",
            x="total_spend",
            palette="Purples_r"
        )

        plt.title(
            f"{category}: Top {top_n} Merchants"
        )

        plt.xlabel("Total Spend ($)")
        plt.ylabel("")

        plt.tight_layout()

# ======================================================
# Merchant Frequency vs Spend
# ======================================================

def plot_merchant_frequency_vs_spend(
    merchant_summary,
    min_monthly_spend=0,
    top_n_labels=20,
    figsize=(12, 8)
):
    """
    X = monthly_frequency
    Y = avg_monthly_spend
    Bubble size = transaction_count

    Useful for identifying:
      - subscriptions
      - groceries
      - dining
      - one-time purchases
      - recurring household expenses
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = (
        merchant_summary[
            merchant_summary["avg_monthly_spend"]
            >= min_monthly_spend
        ]
        .copy()
    )

    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=plot_df,
        x="monthly_frequency",
        y="avg_monthly_spend",
        size="transaction_count",
        sizes=(20, 500),
        alpha=0.7,
        legend=False
    )

    # Label highest-impact merchants
    label_df = (
        plot_df
        .sort_values(
            "avg_monthly_spend",
            ascending=False
        )
        .head(top_n_labels)
    )

    for _, row in label_df.iterrows():

        plt.annotate(
            row["merchant_clean"],
            (
                row["monthly_frequency"],
                row["avg_monthly_spend"]
            ),
            fontsize=8
        )

    plt.title(
        "Merchant Frequency vs Average Monthly Spend"
    )

    plt.xlabel(
        "Monthly Frequency\n(1.0 = Transaction Every Month)"
    )

    plt.ylabel(
        "Average Monthly Spend ($)"
    )

    plt.tight_layout()
    plt.show()