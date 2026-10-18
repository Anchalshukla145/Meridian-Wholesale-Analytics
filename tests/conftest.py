"""
Shared Pytest Fixtures for Meridian Wholesale Analytics
Defines a single, session-scoped local Apache SparkSession for all automated tests.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Initializes a single local SparkSession shared across all test suites."""
    session = (
        SparkSession.builder
        .appName("MeridianTestSession")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    # Note: Do not stop sc prematurely to allow clean teardown across all modules
    try:
        session.stop()
    except Exception:
        pass
