import re
import pandas as pd

###############################################################################
# LOAD MERCHANT MAP
###############################################################################

def load_merchant_map(csv_path="merchant_map.csv"):
    """
    merchant_map.csv format:

    pattern,merchant_clean
    AMAZON MKTPL|AMZN MKTP,AMAZON
    PEACOCKTVLL|PEACOCK.*PREMIUM,PEACOCK
    ...

    Order matters.
    More specific patterns should appear first.
    """

    mapping_df = pd.read_csv(csv_path)

    merchant_map = []

    for _, row in mapping_df.iterrows():
        merchant_map.append(
            (
                re.compile(str(row["pattern"]), re.IGNORECASE),
                str(row["merchant_clean"]).strip()
            )
        )

    return merchant_map


###############################################################################
# CLEANUP REGEX
###############################################################################

PROCESSOR_RE = re.compile(
    r"""
    ^
    (
        PAYPAL\s+\*|
        PP\*|
        PPY\*|
        PY\s+\*|
        SP\s+
        |PX\*
        |PAW\*
        |FH\*
        |RF\s+\*
        |ACT\*
        |AT\s+\*
        |DD\s+\*
        |EB\s+\*
        |IN\s+\*
        |MED\*
        |OTTER\*
        |TRYOTTER\*
        |LGC\*
        |CUR8\*
        |ICP\*
        |SQ\s+\*
        |TST\*
        |VCO\*
        |NIC\*
        |FEVOINC\*
        |DIRECTSUPP\*
        |BCS\*
        |GLOSS\*
        |PTI\*
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

APPLE_PAY_RE = re.compile(
    r"APPLE PAY ENDING IN \d{4}",
    re.IGNORECASE
)

PHONE_RE = re.compile(
    r"\b(?:\d{3}[- ]?\d{3}[- ]?\d{4}|\d{10})\b"
)

STORE_NUMBER_RE = re.compile(
    r"(?:#\d+|\bF\d+\b)"
)

LONG_ID_RE = re.compile(
    r"""
    \b[A-Z0-9]{8,}\b
    |
    \b\d{6,}\b
    """,
    re.VERBOSE
)

STATE_RE = re.compile(
    r"\b(?:WI|IL|MN|CA|TX|PA|FL|CO|VA|OH|NC|NY|DC|GA|SC|IN|UT|WA|MI|NJ|DE|MD|KS|IA|TN)\b"
)

CITY_RE = re.compile(
    r"""
    \b(
        WAUWATOSA|
        MILWAUKEE|
        BROOKFIELD|
        MEQUON|
        GLENDALE|
        GREENFIELD|
        WEST\ ALLIS|
        MENOMONEE\ FALLS|
        MENOMONEE|
        STURGEON\ BAY|
        FISH\ CREEK|
        SISTER\ BAY|
        EGG\ HARBOR|
        ELLISON\ BAY|
        BAILEYS\ HARBOR|
        GERMANTOWN|
        WAUKESHA|
        SHEBOYGAN|
        CHICAGO|
        PITTSBURGH|
        BOULDER|
        MINNEAPOLIS|
        ORLANDO|
        GREENDALE|
        FRANKLIN|
        SOUTH\ BEND|
        LAKE\ BUENA\ VISTA
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


###############################################################################
# CLEANUP
###############################################################################

def cleanup_description(description):

    if pd.isna(description):
        return ""

    desc = str(description).upper()

    desc = PROCESSOR_RE.sub("", desc)

    desc = APPLE_PAY_RE.sub(" ", desc)

    desc = PHONE_RE.sub(" ", desc)

    desc = STORE_NUMBER_RE.sub(" ", desc)

    desc = STATE_RE.sub(" ", desc)

    desc = CITY_RE.sub(" ", desc)

    desc = LONG_ID_RE.sub(" ", desc)

    desc = re.sub(r"[^\w\s&'\.-]", " ", desc)

    desc = re.sub(r"\s+", " ", desc)

    return desc.strip()


###############################################################################
# LOOKUP
###############################################################################

def canonical_lookup(cleaned_desc, merchant_map):

    for pattern, merchant in merchant_map:

        if pattern.search(cleaned_desc):
            return merchant

    return cleaned_desc


###############################################################################
# MAIN NORMALIZATION
###############################################################################

def normalize_merchant(description, merchant_map):

    cleaned_desc = cleanup_description(description)

    merchant_clean = canonical_lookup(
        cleaned_desc,
        merchant_map
    )

    return merchant_clean


###############################################################################
# CSV PROCESSING
###############################################################################

def normalize_csv(
    input_csv,
    output_csv,
    merchant_map_csv="merchant_map.csv"
):

    merchant_map = load_merchant_map(
        merchant_map_csv
    )

    df = pd.read_csv(input_csv)

    df["merchant_clean"] = (
        df["description"]
        .fillna("")
        .apply(
            lambda x:
            normalize_merchant(
                x,
                merchant_map
            )
        )
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(f"Output saved: {output_csv}")
    print(
        f"Unique merchants: "
        f"{df['merchant_clean'].nunique():,}"
    )

    return df


###############################################################################
# EXAMPLE
###############################################################################

if __name__ == "__main__":

    normalize_csv(
        input_csv="unique_descriptions.csv",
        output_csv="merchant_normalized.csv",
        merchant_map_csv="merchant_map.csv"
    )