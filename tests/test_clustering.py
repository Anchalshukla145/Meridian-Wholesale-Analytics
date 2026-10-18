"""
Unit Tests for Clustering, Evaluation, and PCA Modules
Tests Spark MLlib KMeans training, cluster size conservation, profile generation,
and PCA row alignment using synthetic CSV fixtures.
"""

import pytest
import pandas as pd
from pyspark.sql import SparkSession

from src.preprocessing import prepare_features_pipeline
from src.clustering import train_kmeans_model, compute_cluster_profiles
from src.pca_analysis import fit_transform_pca, extract_aligned_pca_table
from src.config import SPENDING_FEATURES, RANDOM_SEED



@pytest.fixture(scope="session")
def sample_dataset(spark, tmp_path_factory):
    """Generates synthetic dataset of 15 records with clear multi-cluster separation loaded from CSV."""
    tmp_dir = tmp_path_factory.mktemp("clustering_data")
    csv_file = tmp_dir / "sample_clusters.csv"

    records = []
    # Cluster A: High Fresh, Low Grocery
    for i in range(5):
        records.append({
            "customer_id": i, "Channel": 1, "Region": 3,
            "Fresh": 25000 + i * 100, "Milk": 2000, "Grocery": 2500,
            "Frozen": 5000, "Detergents_Paper": 800, "Delicassen": 1000
        })
    # Cluster B: High Grocery & Milk
    for i in range(5, 10):
        records.append({
            "customer_id": i, "Channel": 2, "Region": 3,
            "Fresh": 3000, "Milk": 18000 + i * 100, "Grocery": 22000 + i * 100,
            "Frozen": 800, "Detergents_Paper": 9000, "Delicassen": 1500
        })
    # Cluster C: Low multi-category
    for i in range(10, 15):
        records.append({
            "customer_id": i, "Channel": 1, "Region": 1,
            "Fresh": 1500 + i * 10, "Milk": 1200, "Grocery": 1400,
            "Frozen": 600, "Detergents_Paper": 400, "Delicassen": 500
        })

    pdf = pd.DataFrame(records)
    pdf.to_csv(csv_file, index=False)

    df = spark.read.csv(str(csv_file), header=True, inferSchema=True)
    scaled_df, _ = prepare_features_pipeline(df, input_cols=SPENDING_FEATURES)
    return scaled_df


def test_kmeans_training_and_size_conservation(sample_dataset):
    """Verifies that KMeans trains and sum of cluster sizes equals total customer records."""
    total_records = sample_dataset.count()
    k = 2
    model, predictions_df = train_kmeans_model(sample_dataset, k=k, seed=RANDOM_SEED)

    assert "prediction" in predictions_df.columns
    size_counts = predictions_df.groupBy("prediction").count().collect()
    total_assigned = sum(row["count"] for row in size_counts)
    assert total_assigned == total_records


def test_cluster_profiles_generation(sample_dataset):
    """Verifies that cluster profiles compute valid averages and non-empty dynamic labels."""
    model, predictions_df = train_kmeans_model(sample_dataset, k=2, seed=RANDOM_SEED)
    profile_df = compute_cluster_profiles(predictions_df, spending_cols=SPENDING_FEATURES)

    assert len(profile_df) == 2
    assert "Label" in profile_df.columns
    assert "Customer Count" in profile_df.columns
    assert "Mean_Fresh" in profile_df.columns
    assert profile_df["Customer Count"].sum() == sample_dataset.count()
    # Ensure labels are descriptive non-empty strings
    assert all(len(label) > 5 for label in profile_df["Label"])


def test_pca_transformation_and_strict_alignment(sample_dataset):
    """Verifies that PCA transforms scaled features into 2D coordinates and aligns with predictions."""
    model, predictions_df = train_kmeans_model(sample_dataset, k=2, seed=RANDOM_SEED)
    pca_model, pca_transformed_df, variance_info = fit_transform_pca(sample_dataset, k_components=2)

    assert "pca_x" in pca_transformed_df.columns
    assert "pca_y" in pca_transformed_df.columns
    assert variance_info["total_explained_variance"] > 0.0

    aligned_table = extract_aligned_pca_table(predictions_df, pca_transformed_df, spending_cols=SPENDING_FEATURES)
    assert len(aligned_table) == sample_dataset.count()
    assert "customer_id" in aligned_table.columns
    assert "prediction" in aligned_table.columns
    assert "pca_x" in aligned_table.columns
    assert "pca_y" in aligned_table.columns
    assert not aligned_table["pca_x"].isnull().any()
