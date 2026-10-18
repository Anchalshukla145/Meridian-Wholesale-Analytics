"""
Unit Tests for Data Loading & Schema Validation
Tests schema verification, missing columns, negative values, and empty data scenarios
using small synthetic in-memory fixtures.
"""

import pytest
import pandas as pd
from pyspark.sql import SparkSession

from src.data_loader import load_and_validate_dataset, ValidationReport
from src.config import ALL_EXPECTED_COLUMNS, SPENDING_FEATURES



def test_valid_schema_loading(spark, tmp_path):
    """Verifies that a valid CSV file with all required columns passes validation."""
    csv_file = tmp_path / "valid_data.csv"
    data = {
        "Channel": [1, 2, 1],
        "Region": [3, 3, 1],
        "Fresh": [12000, 7000, 6000],
        "Milk": [9000, 8000, 2000],
        "Grocery": [7000, 9500, 4000],
        "Frozen": [200, 1500, 6000],
        "Detergents_Paper": [2500, 3200, 500],
        "Delicassen": [1300, 1700, 1800]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df, report = load_and_validate_dataset(spark, csv_file)
    assert report.is_valid is True
    assert df is not None
    assert report.total_records == 3
    assert len(report.missing_columns) == 0
    assert report.duplicate_rows_count == 0


def test_missing_column_detection(spark, tmp_path):
    """Verifies that missing required columns are detected and reported accurately."""
    csv_file = tmp_path / "missing_col.csv"
    data = {
        "Channel": [1, 2],
        "Region": [3, 3],
        "Fresh": [12000, 7000],
        # "Milk" is missing
        "Grocery": [7000, 9500],
        "Frozen": [200, 1500],
        "Detergents_Paper": [2500, 3200],
        "Delicassen": [1300, 1700]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df, report = load_and_validate_dataset(spark, csv_file)
    assert report.is_valid is False
    assert df is None
    assert "Milk" in report.missing_columns
    assert any("Missing required columns" in msg for msg in report.error_messages)


def test_negative_spending_detection(spark, tmp_path):
    """Verifies that negative spending records trigger validation failure."""
    csv_file = tmp_path / "negative_data.csv"
    data = {
        "Channel": [1, 2],
        "Region": [3, 3],
        "Fresh": [12000, -500],  # Negative spending
        "Milk": [9000, 8000],
        "Grocery": [7000, 9500],
        "Frozen": [200, 1500],
        "Detergents_Paper": [2500, 3200],
        "Delicassen": [1300, 1700]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df, report = load_and_validate_dataset(spark, csv_file)
    assert report.is_valid is False
    assert df is None
    assert report.negative_counts.get("Fresh", 0) == 1
    assert any("Negative spending values detected" in msg for msg in report.error_messages)


def test_empty_dataset_handling(spark, tmp_path):
    """Verifies that an empty CSV file is caught safely without unhandled crashes."""
    csv_file = tmp_path / "empty_data.csv"
    csv_file.write_text("Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen\n")

    df, report = load_and_validate_dataset(spark, csv_file)
    assert report.is_valid is False
    assert df is None
    assert report.total_records == 0
    assert any("empty" in msg.lower() for msg in report.error_messages)
