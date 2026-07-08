import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from rapidfuzz import process, fuzz


# =====================================================
# Default Stopwords
# =====================================================

DEFAULT_STOPWORDS = [

    # processors
    "SQ",
    "TST",
    "PY",
    "PAYPAL",
    "ADYEN",

    # URL artifacts
    "COM",
    "WWW",
    "HTTP",
    "HTTPS",
    "US",
    "EN",

    # corporate terms
    "INC",
    "LLC",
    "CORP",
    "CO",

    # transaction noise
    "PAYMENT",
    "PURCHASE",
    "DEBIT",
    "CREDIT",
    "CHECKCARD",
    "CHECK",

    # generic retail noise
    "ONLINE",
    "STORE",
    "PREMIUM",

    # state abbreviations
    "WI",
    "IL",
    "IN",
    "IA",
    "MN",
    "MI",
    "CA",
    "PA",
    "NY",
    "TX",
    "FL",

    # card types
    "VISA",
    "MC"
]


# =====================================================
# TF-IDF Vectorizer
# =====================================================

def create_vectorizer(custom_stopwords=None):

    stopwords = DEFAULT_STOPWORDS.copy()

    if custom_stopwords:
        stopwords.extend(custom_stopwords)

    return TfidfVectorizer(
        stop_words=stopwords,
        lowercase=False,

        # Must begin with a letter.
        # Prevents:
        # 00004911691
        # 25866091
        token_pattern=r"\b[A-Z][A-Z0-9]+\b"
    )


# =====================================================
# Extract Highest-Value Tokens
# =====================================================

def extract_top_tfidf_tokens(
    row,
    feature_names,
    top_n=3,
    min_length=4
):
    """
    Return highest-weight TF-IDF tokens.
    """

    arr = row.toarray()[0]

    if arr.sum() == 0:
        return ["UNKNOWN"]

    indices = np.argsort(arr)[::-1]

    tokens = []

    for idx in indices:

        if arr[idx] <= 0:
            continue

        token = feature_names[idx]

        if len(token) < min_length:
            continue

        tokens.append(token)

        if len(tokens) >= top_n:
            break

    if len(tokens) == 0:
        return ["UNKNOWN"]

    return tokens


# =====================================================
# Preserve Original Token Order
# =====================================================

def build_ordered_candidate(
    description,
    tokens,
    max_tokens=2
):
    """
    Build merchant label using
    original token order from the
    transaction description.

    Example:

    ACRES | AROMATIC
    ->
    AROMATIC ACRES
    """

    words = description.split()

    matched = []

    for word in words:

        if (
            word in tokens
            and word not in matched
        ):
            matched.append(word)

    return " ".join(
        matched[:max_tokens]
    )


# =====================================================
# Assign Merchant Candidates
# =====================================================

def assign_merchant_candidates(
    df,
    custom_stopwords=None
):

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

    top_token_output = []

    merchant_candidates = []

    for i in range(X.shape[0]):

        top_tokens = extract_top_tfidf_tokens(
            X[i],
            feature_names,
            top_n=3
        )

        candidate = build_ordered_candidate(
            df.iloc[i]["desc_clean"],
            top_tokens,
            max_tokens=2
        )

        top_token_output.append(
            " | ".join(top_tokens)
        )

        merchant_candidates.append(
            candidate
            if candidate
            else "UNKNOWN"
        )

    df["top_tokens"] = top_token_output

    df["merchant_candidate"] = merchant_candidates

    return df


# =====================================================
# Optional Fuzzy Consolidation
# =====================================================

def merge_candidate_names(
    df,
    threshold=92
):
    """
    Merge similar merchant names.

    Examples:

    WALGREEN
    WALGREENS

    PANERA
    PANERA BREAD

    COSTCO
    COSTCO WHOLESALE
    """

    candidates = (
        df["merchant_candidate"]
        .dropna()
        .unique()
    )

    candidates = sorted(candidates)

    canonical_names = []

    mapping = {}

    for candidate in candidates:

        if not canonical_names:

            canonical_names.append(
                candidate
            )

            mapping[candidate] = candidate

            continue

        match, score, _ = process.extractOne(
            candidate,
            canonical_names,
            scorer=fuzz.token_sort_ratio
        )

        if score >= threshold:

            mapping[candidate] = match

        else:

            canonical_names.append(
                candidate
            )

            mapping[candidate] = candidate

    df = df.copy()

    df["merchant_clean"] = (
        df["merchant_candidate"]
        .map(mapping)
    )

    return df


# =====================================================
# Main Pipeline
# =====================================================

def extract_merchants(
    df,
    custom_stopwords=None,
    fuzzy_threshold=92
):
    """
    Full merchant extraction workflow.

    Description
       ↓
    TF-IDF
       ↓
    Top Tokens
       ↓
    Ordered Candidate
       ↓
    Fuzzy Merge
       ↓
    merchant_clean
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