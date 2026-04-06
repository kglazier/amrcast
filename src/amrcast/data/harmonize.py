"""MIC value harmonization and normalization."""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_mic_value(value, sign: str | None = None) -> float | None:
    """Parse a MIC measurement value to float.

    Handles various formats from BV-BRC:
    - Numeric: "4", "0.5", "16.0"
    - Range: ">=32", "<=0.25", ">256"
    - Text: "Resistant", "Susceptible" (returns None)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    value_str = str(value).strip()

    # Strip comparison operators
    value_str = re.sub(r"^[<>=]+\s*", "", value_str)

    try:
        return float(value_str)
    except (ValueError, TypeError):
        return None


def mic_to_log2(mic: float) -> float:
    """Convert MIC in ug/mL to log2 scale."""
    if mic <= 0:
        raise ValueError(f"MIC must be positive, got {mic}")
    return np.log2(mic)


def log2_to_mic(log2_mic: float) -> float:
    """Convert log2(MIC) back to ug/mL."""
    return float(2 ** log2_mic)


def harmonize_mic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and harmonize MIC data from BV-BRC.

    Expects columns: genome_id, antibiotic, measurement_value, measurement_sign

    Returns DataFrame with added columns: mic_value, log2_mic
    """
    df = df.copy()

    # Parse MIC values
    df["mic_value"] = df.apply(
        lambda row: parse_mic_value(
            row.get("measurement_value"),
            row.get("measurement_sign"),
        ),
        axis=1,
    )

    # Drop rows without valid MIC
    n_before = len(df)
    df = df.dropna(subset=["mic_value"])
    df = df[df["mic_value"] > 0]
    n_after = len(df)

    if n_before != n_after:
        logger.info(f"Dropped {n_before - n_after} rows with invalid MIC values")

    # Convert to log2
    df["log2_mic"] = df["mic_value"].apply(mic_to_log2)

    # Normalize antibiotic names to lowercase
    df["antibiotic"] = df["antibiotic"].str.lower().str.strip()

    # Ensure genome_id is string
    df["genome_id"] = df["genome_id"].astype(str)

    return df
