"""
Meridian Wholesale Analytics - Principal Component Analysis (PCA) Module
Computes 2D dimensionality reduction using Spark MLlib PCA and guarantees
strict row alignment with customer records and cluster predictions.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col as F_col
from pyspark.ml.feature import PCA, PCAModel
from pyspark.ml.functions import vector_to_array

from src.config import (
    SPARK_SCALED_FEATURE_COL,
    SPARK_PCA_COL,
    SPARK_PREDICTION_COL,
    SPENDING_FEATURES,
)
from src.utils import logger


def fit_transform_pca(
    scaled_df: DataFrame,
    k_components: int = 2,
    input_col: str = SPARK_SCALED_FEATURE_COL,
    output_col: str = SPARK_PCA_COL
) -> Tuple[PCAModel, DataFrame, Dict[str, Any]]:
    """
    Fits Spark MLlib PCA on scaled features and projects data into 2D coordinates.
    Unpacks coordinates directly onto the Spark DataFrame using vector_to_array
    to guarantee zero row-misalignment errors.
    """
    logger.info(f"Fitting Spark MLlib PCA with k={k_components} on '{input_col}'...")
    pca = PCA(k=k_components, inputCol=input_col, outputCol=output_col)
    pca_model = pca.fit(scaled_df)
    transformed_df = pca_model.transform(scaled_df)

    # Extract explained variance
    explained_variance_list = [float(v) for v in pca_model.explainedVariance]
    pc1_var = explained_variance_list[0] if len(explained_variance_list) > 0 else 0.0
    pc2_var = explained_variance_list[1] if len(explained_variance_list) > 1 else 0.0
    total_var = pc1_var + pc2_var

    variance_info = {
        "pc1_variance": round(pc1_var * 100.0, 2),
        "pc2_variance": round(pc2_var * 100.0, 2),
        "total_explained_variance": round(total_var * 100.0, 2),
        "variance_vector": explained_variance_list
    }

    # Extract PC1 and PC2 directly into named scalar columns within the same Spark DataFrame
    transformed_with_coords = (
        transformed_df
        .withColumn("_pca_arr", vector_to_array(output_col))
        .withColumn("pca_x", F_col("_pca_arr")[0])
        .withColumn("pca_y", F_col("_pca_arr")[1])
        .drop("_pca_arr")
    )

    logger.info(
        f"PCA transformation complete. PC1 explains {variance_info['pc1_variance']}%, "
        f"PC2 explains {variance_info['pc2_variance']}%. Total: {variance_info['total_explained_variance']}%."
    )
    return pca_model, transformed_with_coords, variance_info




def extract_aligned_pca_table(
    predictions_df: DataFrame,
    pca_transformed_df: DataFrame,
    spending_cols: List[str] = SPENDING_FEATURES
) -> pd.DataFrame:
    """
    Ensures predictions and PCA coordinates are strictly merged on customer_id,
    preventing any potential row-order drift during Spark-to-Pandas export.
    """
    # If pca_transformed_df already contains prediction, export directly
    if SPARK_PREDICTION_COL in pca_transformed_df.columns and "pca_x" in pca_transformed_df.columns:
        cols = ["customer_id", SPARK_PREDICTION_COL, "pca_x", "pca_y"] + spending_cols
        if "Channel" in pca_transformed_df.columns:
            cols.append("Channel")
        if "Region" in pca_transformed_df.columns:
            cols.append("Region")
        return pca_transformed_df.select(cols).toPandas()

    # Otherwise join explicitly on customer_id
    pca_subset = pca_transformed_df.select("customer_id", "pca_x", "pca_y")
    joined = predictions_df.join(pca_subset, on="customer_id", how="inner")
    cols = ["customer_id", SPARK_PREDICTION_COL, "pca_x", "pca_y"] + spending_cols
    if "Channel" in joined.columns:
        cols.append("Channel")
    if "Region" in joined.columns:
        cols.append("Region")
    return joined.select(cols).toPandas()
