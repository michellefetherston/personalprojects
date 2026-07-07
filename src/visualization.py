import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")


def plot_category_spending(category_df,
                           top_n=None,
                           figsize=(10, 6)):

    plot_df = category_df.copy()

    if top_n:
        plot_df = plot_df.head(top_n)

    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x="total_spend",
        y="category",
        palette="Blues_r"
    )

    plt.title("Total Spending by Category")
    plt.xlabel("Total Spend ($)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_avg_monthly_category_spending(avg_category_df,
                                       top_n=None,
                                       figsize=(10, 6)):

    plot_df = avg_category_df.copy()

    if top_n:
        plot_df = plot_df.head(top_n)

    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x="avg_monthly_spend",
        y="category",
        palette="Greens_r"
    )

    plt.title("Average Monthly Spending by Category")
    plt.xlabel("Average Monthly Spend ($)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_top_merchants(merchant_df,
                       top_n=20,
                       figsize=(10, 8)):

    plot_df = merchant_df.head(top_n)

    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x="total_spend",
        y="merchant_clean",
        palette="Oranges_r"
    )

    plt.title(f"Top {top_n} Merchants by Spend")
    plt.xlabel("Total Spend ($)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_avg_monthly_merchants(avg_merchant_df,
                               top_n=20,
                               figsize=(10, 8)):

    plot_df = avg_merchant_df.head(top_n)

    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x="avg_monthly_spend",
        y="merchant_clean",
        palette="Purples_r"
    )

    plt.title(f"Top {top_n} Merchants by Average Monthly Spend")
    plt.xlabel("Average Monthly Spend ($)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_monthly_spending(monthly_df, figsize=(12, 6)):

    plot_df = monthly_df.copy()

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

    plt.title("Monthly Spending Trend")
    plt.xlabel("Month")
    plt.ylabel("Spend ($)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()