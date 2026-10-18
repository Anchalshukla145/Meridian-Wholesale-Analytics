"""
Meridian Wholesale Analytics - Utilities & Helper Functions
Provides safe data formatting, SparkSession lifecycle management, and logging helpers.
"""

import sys
import logging
from typing import Any, Optional, Dict
from pyspark.sql import SparkSession

from src.config import PRODUCT_NAME, CLUSTER_COLOR_PALETTE

logger = logging.getLogger("meridian_analytics")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_or_create_spark_session(app_name: str = PRODUCT_NAME) -> SparkSession:
    """
    Initializes or retrieves an active SparkSession optimized for local execution.
    Configures minimal logging, sensible partition counts, and prevents UI port collisions.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    # Suppress verbose JVM log spam
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def format_currency(value: Any) -> str:
    """Formats numeric spending into monetary units (e.g., $12,345 or 12,345 m.u.)."""
    try:
        val = float(value)
        return f"${val:,.0f}"
    except (ValueError, TypeError):
        return "N/A"


def format_pct(value: Any, decimals: int = 1) -> str:
    """Formats a ratio or float into a clean percentage string."""
    try:
        val = float(value)
        # If value is already in 0-100 scale or 0-1 scale
        return f"{val:.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_number(value: Any, decimals: int = 2) -> str:
    """Formats a float with standard commas and fixed decimals."""
    try:
        val = float(value)
        return f"{val:,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def get_cluster_color(cluster_idx: int) -> str:
    """Returns a deterministic, consistent hex color for a given cluster index."""
    try:
        idx = int(cluster_idx)
        return CLUSTER_COLOR_PALETTE[idx % len(CLUSTER_COLOR_PALETTE)]
    except (ValueError, TypeError):
        return "#64748B"


def get_cluster_color_map(num_clusters: int) -> Dict[int, str]:
    """Returns a dictionary mapping cluster indices to their application-wide palette color."""
    return {i: CLUSTER_COLOR_PALETTE[i % len(CLUSTER_COLOR_PALETTE)] for i in range(num_clusters)}
