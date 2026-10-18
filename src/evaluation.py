"""
Meridian Wholesale Analytics - Model Evaluation & Comparison Module
Evaluates K-Means candidate models (k=2..6) using Spark MLlib ClusteringEvaluator
and analyzes cluster distribution, silhouette metrics, and business viability.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

from src.config import (
    K_MIN,
    K_MAX,
    RECOMMENDED_BUSINESS_K,
    RANDOM_SEED,
    SPARK_SCALED_FEATURE_COL,
)
from src.utils import logger


# Historical benchmark results from preliminary notebook experiments for transparency
HISTORICAL_NOTEBOOK_BENCHMARKS: Dict[int, Dict[str, Any]] = {
    2: {"silhouette": 0.900368, "sizes": [427, 13]},
    3: {"silhouette": 0.740622, "sizes": [382, 43, 15]},
    4: {"silhouette": 0.567311, "sizes": [327, 68, 44, 1]},
    5: {"silhouette": 0.367377, "sizes": [295, 111, 27, 5, 2]},
    6: {"silhouette": 0.494810, "sizes": [258, 94, 75, 10, 2, 1]},
}


def evaluate_candidate_models(
    scaled_df: DataFrame,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    seed: int = RANDOM_SEED,
    features_col: str = SPARK_SCALED_FEATURE_COL
) -> Tuple[pd.DataFrame, Dict[int, Tuple[KMeansModel, DataFrame]]]:
    """
    Evaluates candidate models across range [k_min, k_max] using Spark MLlib ClusteringEvaluator.
    Returns:
    - comparison_df: Summary table with silhouette scores, cluster size lists, min/max sizes, and imbalance ratios.
    - models_dict: Mapping of k -> (fitted KMeansModel, predictions DataFrame) for reuse without retraining.
    """
    logger.info(f"Evaluating candidate models for k in [{k_min}, {k_max}] with seed {seed}...")
    total_customers = scaled_df.count()

    evaluator = ClusteringEvaluator(
        featuresCol=features_col,
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    comparison_records = []
    models_dict = {}

    for k in range(k_min, k_max + 1):
        if k > total_customers:
            logger.warning(f"k={k} exceeds total customers ({total_customers}). Skipping.")
            continue

        try:
            kmeans = KMeans(featuresCol=features_col, predictionCol="prediction", k=k, seed=seed)
            model = kmeans.fit(scaled_df)
            predictions_df = model.transform(scaled_df)

            # Compute Silhouette score using Spark MLlib ClusteringEvaluator
            silhouette_score = float(evaluator.evaluate(predictions_df))

            # Calculate cluster sizes
            size_rows = predictions_df.groupBy("prediction").count().orderBy("count", ascending=False).collect()
            cluster_sizes = [int(row["count"]) for row in size_rows]
            non_empty_clusters = len(cluster_sizes)

            smallest_size = min(cluster_sizes) if cluster_sizes else 0
            largest_size = max(cluster_sizes) if cluster_sizes else 0
            imbalance_ratio = round(largest_size / max(1, smallest_size), 2)

            # Format cluster sizes as string
            sizes_str = ", ".join(str(s) for s in cluster_sizes)

            # Check consistency with historical benchmark
            benchmark = HISTORICAL_NOTEBOOK_BENCHMARKS.get(k)
            diff_from_benchmark = (
                round(abs(silhouette_score - benchmark["silhouette"]), 6)
                if benchmark else None
            )

            comparison_records.append({
                "k": k,
                "Silhouette Score": round(silhouette_score, 6),
                "Non-Empty Clusters": non_empty_clusters,
                "Cluster Sizes": sizes_str,
                "Cluster Sizes List": cluster_sizes,
                "Smallest Cluster": smallest_size,
                "Largest Cluster": largest_size,
                "Imbalance Ratio": imbalance_ratio,
                "Total Assigned": sum(cluster_sizes),
                "Benchmark Match": "Exact Match" if diff_from_benchmark == 0.0 else f"Δ {diff_from_benchmark}",
                "Error": None
            })

            models_dict[k] = (model, predictions_df)

        except Exception as e:
            logger.error(f"Error evaluating model for k={k}: {str(e)}")
            comparison_records.append({
                "k": k,
                "Silhouette Score": 0.0,
                "Non-Empty Clusters": 0,
                "Cluster Sizes": "Failed",
                "Cluster Sizes List": [],
                "Smallest Cluster": 0,
                "Largest Cluster": 0,
                "Imbalance Ratio": 0.0,
                "Total Assigned": 0,
                "Benchmark Match": "N/A",
                "Error": str(e)
            })

    comparison_df = pd.DataFrame(comparison_records)
    logger.info("Candidate models evaluation completed successfully.")
    return comparison_df, models_dict


def get_model_diagnostics(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies best silhouette model, compares with recommended business candidate (k=3),
    and derives diagnostic commentary.
    """
    valid_df = comparison_df[comparison_df["Error"].isnull()]
    if valid_df.empty:
        return {
            "best_silhouette_k": None,
            "best_silhouette_score": None,
            "recommended_k": RECOMMENDED_BUSINESS_K,
            "analysis_text": "No valid models evaluated."
        }

    best_row = valid_df.sort_values(by="Silhouette Score", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    best_score = float(best_row["Silhouette Score"])

    rec_row = valid_df[valid_df["k"] == RECOMMENDED_BUSINESS_K]
    rec_score = float(rec_row.iloc[0]["Silhouette Score"]) if not rec_row.empty else None

    return {
        "best_silhouette_k": best_k,
        "best_silhouette_score": best_score,
        "recommended_k": RECOMMENDED_BUSINESS_K,
        "recommended_score": rec_score,
        "best_imbalance": float(best_row["Imbalance Ratio"]),
        "has_single_customer_clusters": any(row["Smallest Cluster"] == 1 for _, row in valid_df.iterrows())
    }
