"""
Meridian Wholesale Analytics - Data Loading & Integrity Validation Module
Loads wholesale customer dataset, verifies schema, enforces non-destructive validation,
and generates structured data-quality diagnostics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from src.config import (
    DEFAULT_DATA_FILE,
    FALLBACK_WINDOWS_DATA_FILE,
    ALL_EXPECTED_COLUMNS,
    SPENDING_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.utils import logger


@dataclass
class ValidationReport:
    """Detailed summary of data integrity and quality checks."""
    is_valid: bool = True
    filepath: str = ""
    total_records: int = 0
    total_columns: int = 0
    columns_found: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    data_types: Dict[str, str] = field(default_factory=dict)
    null_counts: Dict[str, int] = field(default_factory=dict)
    negative_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_rows_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_nulls(self) -> bool:
        return any(cnt > 0 for cnt in self.null_counts.values())

    @property
    def has_negatives(self) -> bool:
        return any(cnt > 0 for cnt in self.negative_counts.values())

    @property
    def quality_score_pct(self) -> float:
        """Returns estimated completeness score as a percentage."""
        if self.total_records == 0:
            return 0.0
        total_cells = self.total_records * max(1, self.total_columns)
        total_issues = sum(self.null_counts.values()) + sum(self.negative_counts.values())
        return max(0.0, round((1.0 - (total_issues / total_cells)) * 100.0, 2))


def resolve_data_path(custom_path: Optional[Path] = None) -> Path:
    """
    Resolves data file path using project-relative resolution with Windows absolute fallback.
    """
    if custom_path and custom_path.exists():
        return custom_path
    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE
    if FALLBACK_WINDOWS_DATA_FILE.exists():
        return FALLBACK_WINDOWS_DATA_FILE
    raise FileNotFoundError(
        f"Wholesale customers dataset not found. Checked primary path '{DEFAULT_DATA_FILE}' "
        f"and fallback path '{FALLBACK_WINDOWS_DATA_FILE}'."
    )


def load_and_validate_dataset(
    spark: SparkSession,
    filepath: Optional[Path] = None
) -> Tuple[Optional[DataFrame], ValidationReport]:
    """
    Loads dataset into Spark DataFrame and performs rigorous data validation.
    Does not silently drop rows or columns.
    """
    report = ValidationReport()

    try:
        resolved_path = resolve_data_path(filepath)
        report.filepath = str(resolved_path)
    except FileNotFoundError as e:
        report.is_valid = False
        report.error_messages.append(str(e))
        return None, report

    # Attempt Spark CSV read
    try:
        raw_df = spark.read.csv(str(resolved_path), header=True, inferSchema=True)
    except Exception as e:
        report.is_valid = False
        report.error_messages.append(f"Failed to read CSV into Spark: {str(e)}")
        return None, report

    # 1. Row count & empty dataset validation
    total_records = raw_df.count()
    report.total_records = total_records
    if total_records == 0:
        report.is_valid = False
        report.error_messages.append("Dataset is completely empty (0 records).")
        return None, report

    cols = raw_df.columns
    report.columns_found = cols
    report.total_columns = len(cols)

    # 2. Schema check (missing / extra columns)
    missing = [c for c in ALL_EXPECTED_COLUMNS if c not in cols]
    unexpected = [c for c in cols if c not in ALL_EXPECTED_COLUMNS]
    report.missing_columns = missing
    report.unexpected_columns = unexpected

    if missing:
        report.is_valid = False
        report.error_messages.append(f"Missing required columns: {', '.join(missing)}")
        return None, report

    if unexpected:
        report.warnings.append(
            f"Detected unexpected additional columns: {', '.join(unexpected)}. "
            "These will not be used in clustering features."
        )

    # 3. Data types validation
    schema_dict = {field.name: field.dataType.simpleString() for field in raw_df.schema}
    report.data_types = schema_dict

    for col_name in SPENDING_FEATURES:
        col_type = raw_df.schema[col_name].dataType
        if not isinstance(col_type, NumericType):
            report.is_valid = False
            report.error_messages.append(
                f"Spending feature '{col_name}' has non-numeric type '{col_type.simpleString()}'. "
                "All spending features must be numeric."
            )

    if not report.is_valid:
        return None, report

    # 4. Missing / Null values check
    null_exprs = [F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c) for c in cols]
    null_counts_row = raw_df.select(null_exprs).collect()[0]
    report.null_counts = {c: int(null_counts_row[c]) for c in cols}

    if report.has_nulls:
        null_details = [f"{c}: {cnt}" for c, cnt in report.null_counts.items() if cnt > 0]
        report.warnings.append(f"Null or NaN values detected: {', '.join(null_details)}")

    # 5. Negative values check across spending features
    neg_exprs = [F.count(F.when(F.col(c) < 0, c)).alias(c) for c in SPENDING_FEATURES]
    neg_counts_row = raw_df.select(neg_exprs).collect()[0]
    report.negative_counts = {c: int(neg_counts_row[c]) for c in SPENDING_FEATURES}

    if report.has_negatives:
        neg_details = [f"{c}: {cnt}" for c, cnt in report.negative_counts.items() if cnt > 0]
        report.is_valid = False
        report.error_messages.append(
            f"Negative spending values detected in columns: {', '.join(neg_details)}. "
            "Wholesale spending cannot be negative."
        )
        return None, report

    # 6. Exact duplicate records count
    distinct_count = raw_df.dropDuplicates().count()
    duplicate_count = total_records - distinct_count
    report.duplicate_rows_count = duplicate_count
    if duplicate_count > 0:
        report.warnings.append(
            f"Detected {duplicate_count} exact duplicate customer records. "
            "Retaining all records to preserve original wholesale distribution."
        )

    # Add an explicit stable customer identifier column for strict alignment
    validated_df = raw_df.withColumn("customer_id", F.monotonically_increasing_id())

    logger.info(
        f"Dataset loaded and validated successfully: {total_records} rows, "
        f"{len(cols)} columns, duplicates: {duplicate_count}."
    )
    return validated_df, report
