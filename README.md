# Meridian Wholesale Analytics — Customer Intelligence Platform

**Meridian Wholesale Analytics** is an enterprise-grade commercial analytics and customer segmentation platform built with **Streamlit** and powered by **Apache Spark 3.5.9 MLlib**. 

The platform enables wholesale distributors, supply-chain executives, and commercial managers to uncover empirical customer archetypes from annual transactional expenditure across six core categories: **Fresh Produce, Milk, Grocery, Frozen Goods, Detergents & Paper Products, and Delicatessen**.

---

## 1. Executive Summary & Architecture

Unlike generic demonstration dashboards or supervised classification models, **Meridian Wholesale Analytics** operates as an unsupervised customer intelligence pipeline:

- **Distributed Data Processing**: Powered by local Apache Spark 3.5.9 with minimal partition overhead and cached session management.
- **Spark MLlib Feature Engineering**: Automated feature pipeline utilizing `pyspark.ml.feature.VectorAssembler` and `pyspark.ml.feature.StandardScaler` ($z$-score standardization with mean-centering and unit variance).
- **Model Evaluation**: Systematic candidate evaluation across $k \in [2, 6]$ utilizing `pyspark.ml.evaluation.ClusteringEvaluator` with `squaredEuclidean` distance.
- **PCA Dimensionality Reduction**: Real-time 2D spatial projections utilizing Spark MLlib `PCA`, retaining $>72\%$ of total variance with strictly enforced row alignment.
- **Strategic Commercial Intelligence**: Dynamic business archetype derivation, tailored category bundling, bulk contract recommendations, and inventory replenishment guidelines.

---

## 2. Project Directory Structure

```text
project/
├── app.py                       # Streamlit application entry point, routing, and layout
├── requirements.txt             # Pinned Python package dependencies
├── README.md                    # Comprehensive technical & execution documentation
├── .gitignore                   # Git exclusion rules for environments, temp & caches
├── assets/
│   └── food_delivery.png        # Official wholesale distribution illustration
├── data/
│   └── Wholesale customers data.csv  # Verified 440-record wholesale transactions dataset
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # System paths, schemas, hyperparameters, and design tokens
│   ├── data_loader.py           # Ingestion, schema verification, and non-destructive validation
│   ├── preprocessing.py         # IQR outlier detection and Spark MLlib VectorAssembler/Scaler
│   ├── clustering.py            # Spark MLlib KMeans training, profiling, and dynamic labeling
│   ├── evaluation.py            # ClusteringEvaluator silhouette benchmarking across k=2..6
│   ├── pca_analysis.py          # Spark MLlib PCA 2D projection with guaranteed row alignment
│   ├── visualizations.py        # Enterprise Plotly charts with consistent design tokens
│   ├── recommendations.py       # Empirical business strategy and inventory prioritization
│   └── utils.py                 # SparkSession lifecycle, formatting helpers, and logging
└── tests/
    ├── conftest.py              # Shared session-scoped Spark fixture
    ├── test_data_loader.py      # Automated schema, negative spending, and null value tests
    ├── test_preprocessing.py   # IQR calculation sanity and feature pipeline vector tests
    └── test_clustering.py       # KMeans convergence, cluster size conservation, and PCA alignment
```

---

## 3. Technology Stack & Prerequisites

- **Language**: Python 3.13 / 3.14 (64-bit)
- **Big Data Engine**: Apache Spark 3.5.9 (PySpark 3.5.9 pinned)
- **Java Runtime**: Oracle JDK 1.8 / OpenJDK 11+
- **Interactive UI**: Streamlit 1.35.0 - 1.58.0
- **Data & Computation**: Pandas 2.2+, NumPy 1.26+
- **Interactive Visualizations**: Plotly 5.20+
- **Statistical Analysis**: Seaborn 0.13+, Matplotlib 3.8+
- **Test Suite**: Pytest 8.0+

---

## 4. Quick Start & Execution Guide (Windows)

Follow these exact Windows PowerShell or Command Prompt commands:

### Step 1: Open Project Directory
```powershell
cd D:\subjects\year4_sem7\Spark\CA2\project
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Verify Environment Versions
```powershell
python -c "import pyspark, streamlit, pandas; print('PySpark:', pyspark.__version__, '| Streamlit:', streamlit.__version__, '| Pandas:', pandas.__version__)"
```

### Step 4: Run Automated Test Suite
```powershell
python -m pytest tests/
```
*Expected result: 10 passed.*

### Step 5: Launch the Meridian Analytics Platform
```powershell
streamlit run app.py
```
*The platform will open automatically in your browser at `http://localhost:8501`.*

### Step 6: Stop the Application
Press `Ctrl + C` in the active terminal window to gracefully terminate the Streamlit server and release Spark resources.

---

## 5. Machine Learning & Mathematical Workflow

### 1. Data Ingestion & Schema Verification
- Dataset source: Wholesale Customers Dataset (440 wholesale customer accounts).
- Attributes: 2 operational categories (`Channel`, `Region`) + 6 numeric spending features (`Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`).
- Schema validation confirms zero missing fields, zero negative records, and 100% record integrity without silent row-dropping.

### 2. Feature Assembly & Standardization
- `VectorAssembler` transforms the 6 numerical categories into a unified 6D feature vector.
- `StandardScaler(withMean=True, withStd=True)` standardizes each feature:
  $$z = \frac{x - \mu}{\sigma}$$
  *Rationale:* In distance-based clustering algorithms like K-Means, features with large absolute ranges (e.g. `Fresh` up to 112,151) would drown out smaller-scale categories (e.g. `Delicassen`). Standardization ensures equal geometric weighting.

### 3. K-Means Training & Objective Function
- Algorithm: Spark MLlib `KMeans` with $k$-means|| initialization, `maxIter=30`, and fixed `seed=42`.
- Objective: Minimize Within-Set Sum of Squared Errors (WSSSE):
  $$WSSSE = \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$

### 4. Silhouette Evaluation Benchmarking
Silhouette score ($s$) is computed via `ClusteringEvaluator` using `squaredEuclidean` metric:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance.

#### Empirical Benchmark Comparison Across Candidate Models:

| Model ($k$) | Spark MLlib Silhouette Score | Cluster Headcounts | Imbalance Ratio | Commercial Assessment |
|:---:|:---:|:---|:---:|:---|
| **$k=2$** | **0.900368** | 427, 13 | 32.85 | **Highest Silhouette Score**: Separates only 13 massive enterprise buyers from an undifferentiated mass of 427 clients. Low operational utility. |
| **$k=3$** | **0.740622** | 382, 43, 15 | 25.47 | **Recommended Business Model**: Preserves exceptional mathematical separation while delineating three clear commercial archetypes: Retail/Grocery specialists (43), Fresh Produce specialists (15), and Core clients (382). |
| **$k=4$** | 0.567311 | 327, 68, 44, 1 | 327.00 | Degrades separation; creates single-customer outlier cluster ($n=1$). |
| **$k=5$** | 0.367377 | 295, 111, 27, 5, 2 | 147.50 | Pronounced fragmentation; micro-clusters ($n=2, 5$). |
| **$k=6$** | 0.494810 | 258, 94, 75, 10, 2, 1 | 258.00 | Over-segmentation with isolated single-record artifacts ($n=1, 2$). |

### 5. Spark MLlib Principal Component Analysis (PCA)
- 2D dimensionality reduction projects 6D scaled vectors onto orthogonal principal eigenvectors.
- **PC1 Variance**: ~44.08%
- **PC2 Variance**: ~28.38%
- **Total Explained Variance**: **72.46%**
- **Strict Row Alignment**: Coordinates (`pca_x`, `pca_y`) and cluster labels are extracted directly on the Spark DataFrame via `vector_to_array` and merged on `customer_id`, preventing row-order drift.

---

## 6. Application Views & Core Features

1. **Overview & Landing**: Hero section, commercial objectives, wholesale delivery illustration (`food_delivery.png`), 3 core benefit cards, and 3-step platform workflow.
2. **Exploratory Data Analysis (EDA)**: Key metrics (440 clients, 8 features, 100% quality, 0 duplicates), interactive histograms, boxplots, Tukey 1.5 × IQR outlier boundaries, and Pearson correlation heatmap ($r=0.92$ between Grocery & Detergents).
3. **Model Evaluation**: Candidate model comparison ($k=2..6$), Silhouette curve line chart, cluster size stacked bar chart, and mathematical vs. business decision analysis.
4. **Customer Segmentation**: Active model summary, cluster size metrics, share donut chart, category expenditure grouped bar chart, standardized $z$-score profile heatmap, and 2D PCA projection scatter plot.
5. **Business Insights**: Segment-specific commercial strategies, cross-selling bundle recommendations, and operational inventory replenishment guidelines.
6. **Data Explorer & Export**: Real-time search and multi-parameter filtering (Cluster, Channel, Region) with customer table pagination and 3 verified CSV download buttons.
7. **Methodology Documentation**: Comprehensive 14-step engineering walkthrough designed for technical viva and presentation review.

---

## 7. Troubleshooting & Diagnostic Guide

### Issue: `JAVA_HOME is not set` or `Java gateway process exited`
- **Root Cause**: Apache Spark requires a valid 64-bit Java Runtime Environment.
- **Fix**: Verify your Java installation and set the environment variable:
  ```powershell
  $env:JAVA_HOME = "C:\Program Files\Java\latest\jdk-1.8"
  $env:SPARK_HOME = "D:\spark"
  ```

### Issue: Port 8501 Collision
- **Fix**: Launch Streamlit on an alternate port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```

### Issue: `Out of Memory` or Temp Directory Permission Errors
- **Fix**: Purge pip and temporary caches on the Windows drive:
  ```powershell
  pip cache purge
  ```

---

## 8. Viva & Presentation Summary

When presenting this project during a technical viva:

1. **Unsupervised vs. Supervised**: This project performs **unsupervised clustering**. The dataset does not possess target labels; the algorithm uncovers latent patterns directly from multivariate expenditure vectors.
2. **Why Standardize?**: K-Means relies on Euclidean distance. Without standardization via `StandardScaler`, attributes with large numerical values (`Fresh`) would dominate the distance metric, masking subtle patterns in smaller categories (`Delicassen`).
3. **Why $k=3$ instead of $k=2$?**: Although $k=2$ achieves a higher silhouette score ($0.900$ vs $0.740$), it creates an extreme $427:13$ split that provides minimal business differentiation. $k=3$ isolates three actionable commercial segments (Retail/Grocery, Fresh, and Core) without creating single-record clusters.
4. **How PCA Alignment is Guaranteed**: PCA coordinates and K-Means predictions are attached as native columns on the exact same Spark DataFrame using `vector_to_array` and exported with explicit `customer_id` alignment.
