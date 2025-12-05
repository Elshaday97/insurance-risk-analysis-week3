import pandas as pd
from ..constants import (
    PARSE_TO_BOOL_COLS,
    PARSE_TO_DATE_COLS,
    PARSE_TO_NUMERIC_COLS,
    OUTLIER_COLS,
    threashold,
)


def _clean_date_column(series: pd.Series) -> pd.Series:
    # parse normal dates
    parsed = pd.to_datetime(series, errors="coerce")

    # detect mm/yyyy formats
    mask = parsed.isna() & series.astype(str).str.match(r"^\d{1,2}/\d{4}$")

    # convert mm/yyyy to first day of that month
    parsed[mask] = series[mask].apply(lambda x: pd.to_datetime(x, format="%m/%Y"))

    return parsed


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for date_col in PARSE_TO_DATE_COLS:
            if date_col in df.columns:
                df[date_col] = _clean_date_column(df[date_col])
        return df
    except ValueError as e:
        raise e


def parse_yes_no_cols(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for yes_no_col in PARSE_TO_BOOL_COLS:
            if yes_no_col in df.columns:
                df[yes_no_col] = (
                    df[yes_no_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"yes": True, "no": False})
                    .astype("boolean")
                )
        return df
    except ValueError as e:
        raise e


def parse_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for numeric_col in PARSE_TO_NUMERIC_COLS:
            if numeric_col in df.columns:
                df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
        return df
    except ValueError as e:
        raise e


def _detect_outliers(series: pd.Series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threashold * IQR
    upper_bound = Q3 - threashold * IQR
    outliers = (series < lower_bound) | (series > upper_bound)

    return outliers


def flag_outliers(df: pd.DataFrame):
    for outlier_col in OUTLIER_COLS:
        if outlier_col in df.columns:
            df[f"Outlier_{outlier_col}"] = _detect_outliers(df[outlier_col])

    return df
