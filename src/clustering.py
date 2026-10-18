"""
Meridian Wholesale Analytics - K-Means Clustering & Profiling Module
Trains Spark MLlib KMeans models, computes comprehensive cluster profiles, and derives
evidence-based cluster labels from empirical spending distributions.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel

from src.config import (
    SPENDING_FEATURES,
    SPARK_SCALED_FEATURE_COL,
    SPARK_PREDICTION_COL,
    RANDOM_SEED,
)
from src.utils import logger


def train_kmeans_model(
    scaled_df: DataFrame,
    k: int,
    seed: int = RANDOM_SEED,
    features_col: str = SPARK_SCALED_FEATURE_COL,
    prediction_col: str = SPARK_PREDICTION_COL
) -> Tuple[KMeansModel, DataFrame]:
    """
    Trains a Spark MLlib KMeans model for a specified k and attaches cluster predictions.
    """
    logger.info(f"Training Spark MLlib KMeans model with k={k}, seed={seed}...")
    kmeans = KMeans(
        featuresCol=features_col,
        predictionCol=prediction_col,
        k=k,
        seed=seed,
        maxIter=30,
        tol=1e-4
    )
    model = kmeans.fit(scaled_df)
    predictions_df = model.transform(scaled_df)
    return model, predictions_df


def derive_dynamic_cluster_label(
    cluster_means: pd.Series,
    overall_means: pd.Series,
    cluster_size: int,
    total_customers: int
) -> str:
    """
    Derives an evidence-based business archetype name from actual spending averages
    compared to the overall wholesale customer population.
    Never hardcodes labels to arbitrary cluster numbers.
    """
    relative_indexes = {col: (cluster_means[col] / overall_means[col]) for col in overall_means.index}
    top_categories = sorted(relative_indexes.items(), key=lambda item: item[1], reverse=True)

    highest_col, highest_ratio = top_categories[0]
    second_col, second_ratio = top_categories[1]

    # Archetype derivation based on spending intensity
    if highest_ratio > 2.0 and second_ratio > 1.8:
        if set([highest_col, second_col]) <= {"Grocery", "Detergents_Paper", "Milk"}:
            return "Grocery & Household Staples Specialists"
        elif set([highest_col, second_col]) <= {"Fresh", "Frozen"}:
            return "Perishable Produce & Cold-Storage Leaders"
        else:
            return f"High-Volume {highest_col.replace('_', ' ')} & {second_col.replace('_', ' ')} Specialists"
    elif highest_ratio > 1.7:
        if highest_col == "Fresh":
            return "Fresh Produce Focused Wholesale Clients"
        elif highest_col in ["Grocery", "Detergents_Paper"]:
            return "Retail Grocery & Consumables Wholesale Segment"
        elif highest_col == "Frozen":
            return "Frozen & Prepared Food Wholesale Specialists"
        elif highest_col == "Milk":
            return "Dairy & Liquid Food Distribution Segment"
        elif highest_col == "Delicassen":
            return "Gourmet & Delicatessen Boutique Segment"
        else:
            return f"{highest_col.replace('_', ' ')} Dominant Segment"
    elif all(ratio < 0.85 for ratio in relative_indexes.values()):
        return "Lean Volume / Multi-Category Micro-Wholesale Segment"
    elif all(0.80 <= ratio <= 1.30 for ratio in relative_indexes.values()):
        return "Balanced Multi-Category Core Wholesale Segment"
    else:
        return f"{highest_col.replace('_', ' ')} & Multi-Category Wholesale Segment"


def compute_cluster_profiles(
    predictions_df: DataFrame,
    spending_cols: List[str] = SPENDING_FEATURES
) -> pd.DataFrame:
    """
    Extracts summary statistics (Count, Percentage, Mean, Median, Min, Max, StdDev)
    for each cluster directly from the assigned records.
    """
    # Convert to Pandas for rich aggregation and formatting
    cols_to_pull = ["customer_id", SPARK_PREDICTION_COL] + spending_cols
    if "Channel" in predictions_df.columns:
        cols_to_pull.append("Channel")
    if "Region" in predictions_df.columns:
        cols_to_pull.append("Region")

    pdf = predictions_df.select(cols_to_pull).toPandas()
    total_customers = len(pdf)

    # Overall dataset means for indexing
    overall_means = pdf[spending_cols].mean()

    clusters = sorted(pdf[SPARK_PREDICTION_COL].unique())
    profile_rows = []

    for cluster_id in clusters:
        cluster_data = pdf[pdf[SPARK_PREDICTION_COL] == cluster_id]
        count = len(cluster_data)
        pct = (count / total_customers) * 100.0

        cluster_means = cluster_data[spending_cols].mean()
        cluster_medians = cluster_data[spending_cols].median()
        cluster_stds = cluster_data[spending_cols].std().fillna(0)

        label = derive_dynamic_cluster_label(
            cluster_means=cluster_means,
            overall_means=overall_means,
            cluster_size=count,
            total_customers=total_customers
        )

        row_dict = {
            "Cluster": cluster_id,
            "Label": label,
            "Customer Count": count,
            "Share (%)": round(pct, 2),
        }

        # Average spending per category
        for col in spending_cols:
            row_dict[f"Mean_{col}"] = round(float(cluster_means[col]), 2)
            row_dict[f"Median_{col}"] = round(float(cluster_medians[col]), 2)
            row_dict[f"Std_{col}"] = round(float(cluster_stds[col]), 2)
            # Relative Index: cluster mean / overall mean * 100 (100 = average)
            ratio = (cluster_means[col] / overall_means[col]) if overall_means[col] > 0 else 1.0
            row_dict[f"Index_{col}"] = round(ratio * 100.0, 1)

        # Dominant Channel and Region
        if "Channel" in cluster_data.columns:
            top_channel = cluster_data["Channel"].mode()
            row_dict["Dominant_Channel"] = int(top_channel[0]) if not top_channel.empty else 1
        if "Region" in cluster_data.columns:
            top_region = cluster_data["Region"].mode()
            row_dict["Dominant_Region"] = int(top_region[0]) if not top_region.empty else 3

        profile_rows.append(row_dict)

    profile_df = pd.DataFrame(profile_rows)
    return profile_df
