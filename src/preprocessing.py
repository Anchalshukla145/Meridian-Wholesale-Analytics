"""
Meridian Wholesale Analytics - Preprocessing & Feature Pipeline Module
Computes IQR-based outlier boundaries and executes Spark MLlib VectorAssembler and StandardScaler.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, StandardScalerModel

from src.config import (
    SPENDING_FEATURES,
    SPARK_RAW_FEATURE_COL,
    SPARK_SCALED_FEATURE_COL,
)
from src.utils import logger


def calculate_iqr_outliers(
    df_pandas: pd.DataFrame,
    features: List[str] = SPENDING_FEATURES
) -> pd.DataFrame:
    """
    Computes quartiles (Q1, Q3), Interquartile Range (IQR = Q3 - Q1),
    and Tukey's bounds (1.5 * IQR) for each spending category.

    Note on methodology:
    Uses standard linear quantile interpolation (numpy/pandas type 7).
    Lower bound is calculated as Q1 - 1.5 * IQR, with an effective wholesale floor at 0.
    Identifies high-value enterprise accounts without automatically discarding them.
    """
    records = []
    total_customers = len(df_pandas)

    for col in features:
        series = df_pandas[col].dropna()
        q1 = float(np.percentile(series, 25))
        median = float(np.percentile(series, 50))
        q3 = float(np.percentile(series, 75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_lower = int((series < lower_bound).sum())
        outliers_upper = int((series > upper_bound).sum())
        total_outliers = outliers_lower + outliers_upper
        outlier_pct = (total_outliers / total_customers) * 100.0 if total_customers > 0 else 0.0

        records.append({
            "Feature": col,
            "Min": float(series.min()),
            "Q1 (25%)": q1,
            "Median (50%)": median,
            "Q3 (75%)": q3,
            "Max": float(series.max()),
            "IQR": iqr,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Lower Outliers": outliers_lower,
            "Upper Outliers": outliers_upper,
            "Total Outliers": total_outliers,
            "Outlier %": round(outlier_pct, 2)
        })

    return pd.DataFrame(records)


def check_zero_variance(
    df_pandas: pd.DataFrame,
    features: List[str] = SPENDING_FEATURES
) -> List[str]:
    """
    Identifies any spending features that exhibit zero variance.
    Prevent division-by-zero errors in standardization pipelines.
    """
    zero_variance_cols = []
    for col in features:
        if col in df_pandas.columns:
            if df_pandas[col].std() == 0 or np.isnan(df_pandas[col].std()):
                zero_variance_cols.append(col)
    return zero_variance_cols


def prepare_features_pipeline(
    df_spark: DataFrame,
    input_cols: List[str] = SPENDING_FEATURES,
    raw_feature_col: str = SPARK_RAW_FEATURE_COL,
    scaled_feature_col: str = SPARK_SCALED_FEATURE_COL,
    with_mean: bool = True,
    with_std: bool = True
) -> Tuple[DataFrame, StandardScalerModel]:
    """
    Builds and executes the Spark MLlib feature preparation pipeline:
    1. VectorAssembler: Combines the 6 numerical spending features into a dense vector.
    2. StandardScaler: Centers data around mean (withMean=True) and scales to unit variance (withStd=True).

    Retains all original columns and customer IDs for downstream reporting and profiling.
    """
    logger.info("Executing Spark MLlib VectorAssembler...")
    assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol=raw_feature_col,
        handleInvalid="error"
    )
    assembled_df = assembler.transform(df_spark)

    logger.info(f"Fitting Spark MLlib StandardScaler (withMean={with_mean}, withStd={with_std})...")
    scaler = StandardScaler(
        inputCol=raw_feature_col,
        outputCol=scaled_feature_col,
        withMean=with_mean,
        withStd=with_std
    )
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    logger.info("Feature preparation pipeline successfully executed.")
    return scaled_df, scaler_model
