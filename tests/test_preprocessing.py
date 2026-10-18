"""
Unit Tests for Preprocessing & Feature Pipeline
Tests IQR outlier calculations, bounds sanity, VectorAssembler, and StandardScaler behavior.
"""

import pytest
import pandas as pd
from pyspark.sql import SparkSession

from src.preprocessing import calculate_iqr_outliers, check_zero_variance, prepare_features_pipeline
from src.config import SPENDING_FEATURES



def test_iqr_outliers_calculation():
    """Verifies that IQR and Tukey bounds are calculated mathematically accurately."""
    data = {
        "Fresh": [10, 12, 11, 13, 12, 14, 12, 100],  # 100 is an extreme outlier
        "Milk": [20, 22, 21, 23, 22, 20, 21, 22],
        "Grocery": [5, 6, 5, 7, 6, 5, 6, 5],
        "Frozen": [2, 3, 2, 4, 3, 2, 3, 2],
        "Detergents_Paper": [1, 2, 1, 3, 2, 1, 2, 1],
        "Delicassen": [4, 5, 4, 6, 5, 4, 5, 4]
    }
    df = pd.DataFrame(data)
    outlier_df = calculate_iqr_outliers(df, features=SPENDING_FEATURES)

    fresh_row = outlier_df[outlier_df["Feature"] == "Fresh"].iloc[0]
    assert fresh_row["Total Outliers"] >= 1
    assert fresh_row["Upper Bound"] < 100.0
    assert fresh_row["IQR"] > 0


def test_zero_variance_detection():
    """Verifies detection of constant/zero-variance features."""
    data = {
        "Fresh": [10, 10, 10, 10],  # Constant
        "Milk": [5, 10, 15, 20],
    }
    df = pd.DataFrame(data)
    zero_cols = check_zero_variance(df, features=["Fresh", "Milk"])
    assert "Fresh" in zero_cols
    assert "Milk" not in zero_cols


def test_feature_pipeline_execution(spark, tmp_path):
    """Verifies that VectorAssembler and StandardScaler produce expected 6-dimensional vectors."""
    csv_file = tmp_path / "test_features.csv"
    data = {
        "Fresh": [1000, 2000, 3000, 4000],
        "Milk": [2000, 3000, 4000, 5000],
        "Grocery": [3000, 4000, 5000, 6000],
        "Frozen": [4000, 5000, 6000, 7000],
        "Detergents_Paper": [5000, 6000, 7000, 8000],
        "Delicassen": [6000, 7000, 8000, 9000]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    spark_df = spark.read.csv(str(csv_file), header=True, inferSchema=True)

    scaled_df, model = prepare_features_pipeline(spark_df, input_cols=SPENDING_FEATURES)
    first_record = scaled_df.first()

    assert "features" in scaled_df.columns
    assert "scaled_features" in scaled_df.columns
    assert len(first_record["features"]) == 6
    assert len(first_record["scaled_features"]) == 6
