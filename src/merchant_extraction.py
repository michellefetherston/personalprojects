import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from rapidfuzz import (
    process,
    fuzz
)


DEFAULT_STOPWORDS = [
    "SQ",
    "TST",
    "PY",
    "POS",
    "ONLINE",
    "STORE",
    "INC",
    "LLC",
    "CO",
    "CORP",
    "THE",
    "PAYMENT",
    "PURCHASE",
    "DEBIT",
    "CREDIT",
    "CHECKCARD",
    "CHECK",
    "VISA",
    "MC",
    "WI",
    "IL",
    "MN",
    "MI"
]


def create_vectorizer(custom_stopwords=None):

    stopwords = DEFAULT_STOPWORDS.copy()

    if custom_stopwords:
        stopwords.extend(custom_stopwords)

    return TfidfVectorizer(
        stop_words=stopwords,
        lowercase=False,
        token_pattern=r"\b[A-Z0-9]+\b"
    )


def extract_top_tfidf_token(
    row,
    feature_names
):

    arr = row.toarray()[0]

    if arr.sum() == 0:
        return "UNKNOWN"

    idx = np.argmax(arr)

    return feature_names[idx]


def assign_merchant_candidates(
    df: pd.DataFrame,
    custom_stopwords=None
) -> pd.DataFrame:
    """
    Use TF-IDF to identify
    the most distinctive token in
    each merchant description.
    """

    vectorizer = create_vectorizer(
        custom_stopwords
    )

    X = vectorizer.fit_transform(
        df["desc_clean"]
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    df = df.copy()

    df["merchant_candidate"] = [
        extract_top_tfidf_token(
            X[i],
            feature_names
        )
        for i in range(X.shape[0])
    ]

    return df


def merge_candidate_names(
    df: pd.DataFrame,
    threshold=90
) -> pd.DataFrame:
    """
    Merge candidate names like:

    WALGREEN
    WALGREENS

    MCDONALD
    MCDONALDS
    """

    candidates = sorted(
        df["merchant_candidate"]
        .dropna()
        .unique()
    )

    canonical_names = []
    mapping = {}

    for candidate in candidates:

        if not canonical_names:
            canonical_names.append(candidate)
            mapping[candidate] = candidate
            continue

        match, score, _ = process.extractOne(
            candidate,
            canonical_names,
            scorer=fuzz.ratio
        )

        if score >= threshold:
            mapping[candidate] = match
        else:
            canonical_names.append(candidate)
            mapping[candidate] = candidate

    df = df.copy()

    df["merchant_clean"] = (
        df["merchant_candidate"]
        .map(mapping)
    )

    return df


def extract_merchants(
    df: pd.DataFrame,
    custom_stopwords=None,
    fuzzy_threshold=90
) -> pd.DataFrame:
    """
    Full merchant extraction pipeline.
    """

    df = assign_merchant_candidates(
        df,
        custom_stopwords
    )

    df = merge_candidate_names(
        df,
        threshold=fuzzy_threshold
    )

    return df