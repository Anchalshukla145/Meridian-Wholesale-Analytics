"""
Meridian Wholesale Analytics - Customer Intelligence Platform
Streamlit Analytics Application for Wholesale Customer Segmentation using Apache Spark MLlib.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Meridian Wholesale Analytics",
    page_icon="assets/food_delivery.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    PRODUCT_NAME,
    PRODUCT_BADGE,
    PRODUCT_TAGLINE,
    PRODUCT_DESCRIPTION,
    DEFAULT_DATA_FILE,
    FALLBACK_WINDOWS_DATA_FILE,
    LANDING_IMAGE_FILE,
    FALLBACK_WINDOWS_IMAGE_FILE,
    CATEGORICAL_FEATURES,
    SPENDING_FEATURES,
    ALL_EXPECTED_COLUMNS,
    CHANNEL_LABELS,
    REGION_LABELS,
    K_MIN,
    K_MAX,
    DEFAULT_K,
    RECOMMENDED_BUSINESS_K,
    RANDOM_SEED,
    CLUSTER_COLOR_PALETTE,
    THEME_COLORS,
    PAGES,
)
from src.utils import (
    get_or_create_spark_session,
    format_currency,
    format_pct,
    format_number,
    get_cluster_color,
    get_cluster_color_map,
    logger,
)
from src.data_loader import load_and_validate_dataset, ValidationReport
from src.preprocessing import calculate_iqr_outliers, prepare_features_pipeline, check_zero_variance
from src.clustering import train_kmeans_model, compute_cluster_profiles
from src.evaluation import evaluate_candidate_models, get_model_diagnostics
from src.pca_analysis import fit_transform_pca, extract_aligned_pca_table
from src.visualizations import (
    plot_spending_distributions,
    plot_iqr_outliers_bar,
    plot_correlation_heatmap,
    plot_silhouette_curve,
    plot_cluster_sizes_comparison,
    plot_cluster_size_bar,
    plot_cluster_distribution_donut,
    plot_cluster_spending_grouped_bar,
    plot_standardized_profile_heatmap,
    plot_pca_scatter,
)
from src.recommendations import generate_cluster_insights


# Enterprise CSS Design System
CUSTOM_CSS = """
<style>
    /* Global Typography & Base Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1360px;
    }

    /* Enterprise Metric Card */
    .meridian-metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .meridian-metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08);
    }
    .meridian-metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .meridian-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .meridian-metric-subtext {
        font-size: 0.8rem;
        color: #0284C7;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Section Card Container */
    .meridian-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    /* Header & Badge Styling */
    .meridian-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background-color: #E0F2FE;
        color: #0369A1;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .meridian-badge-teal {
        background-color: #CCFBF1;
        color: #0F766E;
    }
    .meridian-badge-amber {
        background-color: #FEF3C7;
        color: #B45309;
    }

    /* Hero Section Styling */
    .hero-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 36px 40px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 24px;
        max-width: 620px;
    }

    /* Benefit Highlights */
    .benefit-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 14px;
    }
    .benefit-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .benefit-desc {
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Table & Callout Polish */
    .dataframe {
        font-size: 0.85rem !important;
        border-collapse: collapse !important;
    }
    .dataframe th {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }

    /* Sidebar Refinements */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-brand-box {
        padding: 12px 4px 18px 4px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 18px;
    }
    .sidebar-brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748B;
    }

    /* Footer */
    .meridian-footer {
        border-top: 1px solid #E2E8F0;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        text-align: center;
        font-size: 0.8rem;
        color: #94A3B8;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Application State Initialization
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

if "selected_k" not in st.session_state:
    st.session_state["selected_k"] = DEFAULT_K


# Cached Spark Session Management
@st.cache_resource(show_spinner="Initializing Apache Spark Session...")
def get_spark_context():
    """Initializes and caches the local SparkSession."""
    try:
        return get_or_create_spark_session()
    except Exception as e:
        st.error(f"Failed to start Apache Spark: {str(e)}")
        return None


# Cached Data Pipeline Execution
@st.cache_resource(show_spinner="Loading and validating wholesale customer data...")
def get_cached_raw_data():
    """Loads CSV via Spark and extracts Pandas representations for fast interactive charting."""
    spark = get_spark_context()
    if spark is None:
        return None, None, None

    spark_df, report = load_and_validate_dataset(spark)
    if spark_df is None or not report.is_valid:
        return None, None, report

    # Convert full dataset to Pandas once for deterministic EDA and table views
    pdf = spark_df.toPandas()
    return spark_df, pdf, report


@st.cache_resource(show_spinner="Evaluating candidate K-Means models (k=2 to 6)...")
def get_cached_clustering_pipeline():
    """
    Executes feature scaling and evaluates candidate K-Means models across k=2..6.
    Stores models in cache to prevent costly recomputation during page navigation.
    """
    spark = get_spark_context()
    if spark is None:
        return None, None, None, None

    spark_df, report = load_and_validate_dataset(spark)
    if spark_df is None or not report.is_valid:
        return None, None, None, None

    scaled_df, scaler_model = prepare_features_pipeline(spark_df)
    comparison_df, models_dict = evaluate_candidate_models(scaled_df, k_min=K_MIN, k_max=K_MAX)
    pca_model, pca_df, variance_info = fit_transform_pca(scaled_df, k_components=2)

    return scaled_df, comparison_df, models_dict, (pca_model, pca_df, variance_info)


# Sidebar Navigation & Product Branding
def render_sidebar():
    with st.sidebar:
        # Product Header Box
        st.markdown(
            f"""
            <div class="sidebar-brand-box">
                <div class="meridian-badge">{PRODUCT_BADGE}</div>
                <div class="sidebar-brand-name">{PRODUCT_NAME}</div>
                <div class="sidebar-brand-sub">Wholesale Segmentation & Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Navigation**")
        selected = st.radio(
            "Select View",
            options=PAGES,
            index=PAGES.index(st.session_state["current_page"]),
            label_visibility="collapsed"
        )
        if selected != st.session_state["current_page"]:
            st.session_state["current_page"] = selected
            st.rerun()

        st.markdown("---")

        # Global Model Hyperparameter Controls
        st.markdown("**Segmentation Configuration**")
        chosen_k = st.slider(
            "Active Cluster Count (k)",
            min_value=K_MIN,
            max_value=K_MAX,
            value=st.session_state["selected_k"],
            step=1,
            help="Select the number of clusters to inspect across segmentation views."
        )
        if chosen_k != st.session_state["selected_k"]:
            st.session_state["selected_k"] = chosen_k
            st.rerun()

        st.caption(f"Default: k={DEFAULT_K} | Recommended: k={RECOMMENDED_BUSINESS_K}")

        st.markdown("---")
        # System status badge
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#64748B;">
                <strong>Engine:</strong> Apache Spark 3.5.9 MLlib<br>
                <strong>Seed:</strong> 42 (Reproducible)<br>
                <strong>Dataset:</strong> 440 wholesale clients<br>
                <strong>Architecture:</strong> VectorAssembler + StandardScaler
            </div>
            """,
            unsafe_allow_html=True
        )


# Page 1: Overview / Landing Page
def render_overview():
    col_hero_left, col_hero_right = st.columns([1.15, 0.85], gap="large")

    with col_hero_left:
        st.markdown(f'<div class="meridian-badge">{PRODUCT_BADGE}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-title">{PRODUCT_TAGLINE}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="hero-desc">{PRODUCT_DESCRIPTION}</div>',
            unsafe_allow_html=True
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Explore Analytics", type="primary", use_container_width=True):
                st.session_state["current_page"] = "Exploratory Data Analysis"
                st.rerun()
        with col_btn2:
            if st.button("View Methodology", use_container_width=True):
                st.session_state["current_page"] = "Methodology"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Three compact benefit callouts
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                """
                <div class="benefit-card">
                    <div class="benefit-title">Customer Segmentation</div>
                    <div class="benefit-desc">Spark MLlib K-Means clustering across 6 core wholesale spending categories.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with b2:
            st.markdown(
                """
                <div class="benefit-card">
                    <div class="benefit-title">Spending Insights</div>
                    <div class="benefit-desc">Statistical profiling, IQR anomaly detection, and 2D PCA projections.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with b3:
            st.markdown(
                """
                <div class="benefit-card">
                    <div class="benefit-title">Business Action</div>
                    <div class="benefit-desc">Actionable merchandising bundles, order replenishment, and cross-sell roadmaps.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_hero_right:
        # Display image from assets/food_delivery.png
        resolved_img = None
        if LANDING_IMAGE_FILE.exists():
            resolved_img = LANDING_IMAGE_FILE
        elif FALLBACK_WINDOWS_IMAGE_FILE.exists():
            resolved_img = FALLBACK_WINDOWS_IMAGE_FILE

        if resolved_img:
            st.markdown(
                """
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.06); text-align:center;">
                """,
                unsafe_allow_html=True
            )
            st.image(str(resolved_img), caption="Wholesale Logistics & Category Distribution Operations", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"Landing image not found at expected location: {LANDING_IMAGE_FILE}")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Three-step platform workflow
    st.subheader("Analytical Framework")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown(
            """
            <div class="meridian-panel">
                <span class="meridian-badge">Step 1</span>
                <h4>Explore Wholesale Spending</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.5;">
                    Inspect the distribution of annual expenditure across Fresh, Milk, Grocery, Frozen,
                    Detergents & Paper, and Delicassen. Uncover category skews and high-value wholesale accounts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with w2:
        st.markdown(
            """
            <div class="meridian-panel">
                <span class="meridian-badge">Step 2</span>
                <h4>Discover Segments with Spark</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.5;">
                    Apply Spark MLlib StandardScaler and train K-Means models from k=2 through k=6.
                    Evaluate clustering quality with Euclidean Silhouette scores and balance segment interpretability.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with w3:
        st.markdown(
            """
            <div class="meridian-panel">
                <span class="meridian-badge">Step 3</span>
                <h4>Turn Insights into Actions</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.5;">
                    Generate evidence-based commercial strategies: targeted category bundling, tailored credit terms,
                    and cold-chain replenishment prioritization tailored to each empirical archetype.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Page 2: Exploratory Data Analysis
def render_eda():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Data Diagnostics</div>
            <h2>Exploratory Data Analysis</h2>
            <p style="color:#475569;">Empirical evaluation of dataset schema, distribution skews, outlier boundaries, and inter-category correlations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, pdf, report = get_cached_raw_data()
    if pdf is None or report is None or not report.is_valid:
        st.error(f"Failed to load dataset: {report.error_messages if report else 'Unknown error'}")
        return

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Total Wholesale Clients</div>
                <div class="meridian-metric-value">{report.total_records:,}</div>
                <div class="meridian-metric-subtext">Verified customer records</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Feature Columns</div>
                <div class="meridian-metric-value">{report.total_columns}</div>
                <div class="meridian-metric-subtext">2 categorical + 6 spending</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Data Quality Score</div>
                <div class="meridian-metric-value">{report.quality_score_pct:.1f}%</div>
                <div class="meridian-metric-subtext">Zero nulls & zero negatives</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Duplicate Rows</div>
                <div class="meridian-metric-value">{report.duplicate_rows_count}</div>
                <div class="meridian-metric-subtext">100% unique transaction profiles</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Tabs for structured exploration
    tab_overview, tab_dist, tab_iqr, tab_corr = st.tabs([
        "Dataset Overview & Schema",
        "Spending Distributions",
        "IQR Outlier Analysis",
        "Correlation Matrix"
    ])

    with tab_overview:
        st.subheader("Dataset Preview & Descriptive Statistics")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("**Sample Customer Records (First 5 Rows)**")
            st.dataframe(pdf.head(5), use_container_width=True)

        with c_right:
            st.markdown("**Schema & Column Definitions**")
            schema_data = []
            for col in pdf.columns:
                col_type = str(pdf[col].dtype)
                role = "Categorical / Context" if col in CATEGORICAL_FEATURES else "Clustering Feature"
                schema_data.append({"Column": col, "Data Type": col_type, "Role": role, "Nulls": int(pdf[col].isnull().sum())})
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

        st.markdown("**Summary Statistics for Spending Categories (Monetary Units)**")
        desc = pdf[SPENDING_FEATURES].describe().T
        desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        desc.columns = ["Count", "Mean ($)", "StdDev ($)", "Min ($)", "Q1 ($)", "Median ($)", "Q3 ($)", "Max ($)"]
        st.dataframe(desc.style.format("{:,.1f}"), use_container_width=True)

    with tab_dist:
        st.subheader("Univariate Spending Distributions")
        selected_cat = st.selectbox("Select Spending Category to Inspect", SPENDING_FEATURES, index=0)
        fig_dist = plot_spending_distributions(pdf, selected_cat)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Category metrics bar
        ser = pdf[selected_cat]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Minimum", f"${ser.min():,.0f}")
        c2.metric("Q1 (25%)", f"${np.percentile(ser, 25):,.0f}")
        c3.metric("Median (50%)", f"${ser.median():,.0f}")
        c4.metric("Mean", f"${ser.mean():,.0f}")
        c5.metric("Maximum", f"${ser.max():,.0f}")

    with tab_iqr:
        st.subheader("Interquartile Range (IQR) Anomaly Detection")
        st.markdown(
            """
            *Statistical Rule:* Tukey's 1.5 × IQR standard rule flags points exceeding `Q3 + 1.5 × IQR` or below `Q1 - 1.5 × IQR`.
            In wholesale analytics, high outliers represent large enterprise buyers (key accounts) rather than corrupted records.
            Consequently, all records are preserved in the clustering pipeline to capture the full commercial spectrum.
            """
        )
        outlier_df = calculate_iqr_outliers(pdf, SPENDING_FEATURES)
        st.dataframe(
            outlier_df.style.format({
                "Min": "${:,.0f}",
                "Q1 (25%)": "${:,.0f}",
                "Median (50%)": "${:,.0f}",
                "Q3 (75%)": "${:,.0f}",
                "Max": "${:,.0f}",
                "IQR": "${:,.0f}",
                "Lower Bound": "${:,.0f}",
                "Upper Bound": "${:,.0f}",
                "Total Outliers": "{:,}",
                "Outlier %": "{:.1f}%"
            }),
            use_container_width=True
        )

        fig_iqr = plot_iqr_outliers_bar(outlier_df)
        st.plotly_chart(fig_iqr, use_container_width=True)

    with tab_corr:
        st.subheader("Feature Correlation Analysis")
        st.markdown(
            """
            Pearson correlation matrix measuring linear co-dependence between wholesale spending categories.
            Notice high positive correlation between **Grocery and Detergents_Paper (r = 0.92)** and **Milk and Grocery (r = 0.73)**.
            These patterns reflect purchasing behavior (retail stores ordering dry goods and cleaning supplies together).
            """
        )
        fig_corr = plot_correlation_heatmap(pdf, SPENDING_FEATURES)
        st.plotly_chart(fig_corr, use_container_width=True)


# Page 3: Model Evaluation
def render_evaluation():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">MLlib Clustering Engine</div>
            <h2>K-Means Model Evaluation & Selection</h2>
            <p style="color:#475569;">Rigorous model comparison across candidate cluster counts (k=2 through 6) using Apache Spark MLlib ClusteringEvaluator.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    scaled_df, comparison_df, models_dict, pca_tuple = get_cached_clustering_pipeline()
    if comparison_df is None or models_dict is None:
        st.error("Model evaluation pipeline could not be executed.")
        return

    diagnostics = get_model_diagnostics(comparison_df)
    best_k = diagnostics["best_silhouette_k"]
    best_score = diagnostics["best_silhouette_score"]
    rec_k = diagnostics["recommended_k"]
    rec_score = diagnostics["recommended_score"]
    curr_k = st.session_state["selected_k"]

    # Comparative Metric Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-badge">Highest Silhouette Score</div>
                <div class="meridian-metric-value">k = {best_k} ({best_score:.6f})</div>
                <div class="meridian-metric-subtext" style="color:#D97706;">Sizes: 427, 13 (High Imbalance)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-badge meridian-badge-teal">Recommended Business Model</div>
                <div class="meridian-metric-value">k = {rec_k} ({rec_score:.6f})</div>
                <div class="meridian-metric-subtext" style="color:#0D9488;">Sizes: 382, 43, 15 (Balanced Segments)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        curr_row = comparison_df[comparison_df["k"] == curr_k].iloc[0]
        curr_score = float(curr_row["Silhouette Score"])
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-badge meridian-badge-amber">Currently Selected Model</div>
                <div class="meridian-metric-value">k = {curr_k} ({curr_score:.6f})</div>
                <div class="meridian-metric-subtext">Sizes: {curr_row['Cluster Sizes']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Visualizations
    col_curve, col_sizes = st.columns([1, 1])
    with col_curve:
        fig_curve = plot_silhouette_curve(comparison_df, selected_k=curr_k, highest_k=best_k, recommended_k=rec_k)
        st.plotly_chart(fig_curve, use_container_width=True)

    with col_sizes:
        fig_sizes = plot_cluster_sizes_comparison(comparison_df)
        st.plotly_chart(fig_sizes, use_container_width=True)

    # Detailed Table
    st.subheader("Candidate Model Comparison Table")
    display_table = comparison_df[[
        "k", "Silhouette Score", "Non-Empty Clusters", "Cluster Sizes", "Smallest Cluster", "Largest Cluster", "Imbalance Ratio", "Benchmark Match"
    ]]
    st.dataframe(display_table, use_container_width=True)

    # Commercial Selection Analysis Callout
    st.markdown(
        """
        <div class="meridian-panel">
            <h4>Methodological Note: Balancing Mathematical Separation vs. Business Actionability</h4>
            <p style="font-size:0.9rem; color:#334155; line-height:1.6;">
                While <strong>k = 2</strong> yields the maximum mathematical silhouette score (<strong>0.900368</strong>),
                it splits the dataset into an extreme imbalance: a single cluster of <strong>427 customers</strong> and an outlier cluster
                of only <strong>13 customers</strong>. For commercial decision-making, an undifferentiated mass of 427 accounts provides
                minimal strategic value.<br><br>
                Conversely, <strong>k = 3</strong> maintains an exceptional silhouette score of <strong>0.740622</strong> while isolating
                three distinct, meaningful commercial cohorts (Retail/Grocery specialists of 43 clients, Fresh produce specialists of 15 clients,
                and a Core multi-category cohort of 382 clients). As k expands to 4, 5, or 6, cluster sizes fragment into single-customer
                artifacts (e.g., clusters with n=1 or n=2), demonstrating diminishing returns.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Page 4: Customer Segmentation
def render_segmentation():
    curr_k = st.session_state["selected_k"]

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Segment Profiles</div>
            <h2>Customer Segmentation (Model k={curr_k})</h2>
            <p style="color:#475569;">Empirical spending breakdowns, standardized z-score heatmaps, and Spark MLlib 2D PCA spatial distribution.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    scaled_df, comparison_df, models_dict, pca_tuple = get_cached_clustering_pipeline()
    if models_dict is None or curr_k not in models_dict:
        st.error(f"Trained model for k={curr_k} is unavailable.")
        return

    kmeans_model, predictions_df = models_dict[curr_k]
    profile_df = compute_cluster_profiles(predictions_df, spending_cols=SPENDING_FEATURES)
    pca_model, pca_df, variance_info = pca_tuple
    aligned_pca = extract_aligned_pca_table(predictions_df, pca_df, spending_cols=SPENDING_FEATURES)

    # Cluster Size Metric Cards
    cols = st.columns(len(profile_df))
    for idx, (_, row) in enumerate(profile_df.iterrows()):
        c_id = int(row["Cluster"])
        color = get_cluster_color(c_id)
        with cols[idx]:
            st.markdown(
                f"""
                <div class="meridian-metric-card" style="border-top: 4px solid {color};">
                    <div class="meridian-metric-label">Cluster {c_id}</div>
                    <div class="meridian-metric-value" style="font-size:1.4rem;">{row['Customer Count']} clients</div>
                    <div class="meridian-metric-subtext" style="color:#64748B;">{row['Share (%)']}% of wholesale base</div>
                    <div style="font-size:0.75rem; font-weight:600; color:{color}; margin-top:4px;">{row['Label']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Charts: Sizes & Shares
    c_bar, c_donut = st.columns([1, 1])
    with c_bar:
        st.plotly_chart(plot_cluster_size_bar(profile_df), use_container_width=True)
    with c_donut:
        st.plotly_chart(plot_cluster_distribution_donut(profile_df), use_container_width=True)

    # Category Spending Grouped Bar Chart
    st.plotly_chart(plot_cluster_spending_grouped_bar(profile_df, SPENDING_FEATURES), use_container_width=True)

    # Standardized Profile Heatmap & PCA Scatter Plot
    c_heat, c_pca = st.columns([1, 1])
    with c_heat:
        st.plotly_chart(plot_standardized_profile_heatmap(profile_df, SPENDING_FEATURES), use_container_width=True)
        st.caption(
            "Note: Standardized values represent relative standard deviations (σ) above/below the cluster-average mean for each category. "
            "They illustrate relative category intensity, not raw monetary expenditure."
        )

    with c_pca:
        st.plotly_chart(plot_pca_scatter(aligned_pca, variance_info, SPENDING_FEATURES), use_container_width=True)
        st.caption(
            "Spark MLlib PCA 2D projection. Explains over 72% of total variance across the 6 standardized features. "
            "All coordinates are strictly aligned by customer_id."
        )

    # Detailed Numerical Profiles Table
    st.subheader(f"Detailed Statistical Profile: Model k={curr_k}")
    display_profile = profile_df[[
        "Cluster", "Label", "Customer Count", "Share (%)",
        "Mean_Fresh", "Mean_Milk", "Mean_Grocery", "Mean_Frozen", "Mean_Detergents_Paper", "Mean_Delicassen"
    ]]
    st.dataframe(
        display_profile.style.format({
            "Share (%)": "{:.1f}%",
            "Mean_Fresh": "${:,.0f}",
            "Mean_Milk": "${:,.0f}",
            "Mean_Grocery": "${:,.0f}",
            "Mean_Frozen": "${:,.0f}",
            "Mean_Detergents_Paper": "${:,.0f}",
            "Mean_Delicassen": "${:,.0f}"
        }),
        use_container_width=True
    )


# Page 5: Business Insights
def render_business_insights():
    curr_k = st.session_state["selected_k"]

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Strategic Action</div>
            <h2>Evidence-Based Commercial Recommendations</h2>
            <p style="color:#475569;">Derived directly from empirical cluster spending distributions for the selected model (k={curr_k}).</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, pdf, report = get_cached_raw_data()
    scaled_df, comparison_df, models_dict, pca_tuple = get_cached_clustering_pipeline()
    if pdf is None or models_dict is None or curr_k not in models_dict:
        st.error("Data or models unavailable for generating business recommendations.")
        return

    kmeans_model, predictions_df = models_dict[curr_k]
    profile_df = compute_cluster_profiles(predictions_df, spending_cols=SPENDING_FEATURES)
    overall_means = pdf[SPENDING_FEATURES].mean()
    insights = generate_cluster_insights(profile_df, overall_means)

    for item in insights:
        c_id = item["cluster_id"]
        color = get_cluster_color(c_id)

        st.markdown(
            f"""
            <div class="meridian-panel" style="border-left: 6px solid {color};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span class="meridian-badge" style="background-color:{color}22; color:{color};">Cluster {c_id}</span>
                        <h3 style="margin:4px 0 0 0; color:#0F172A;">{item['label']}</h3>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:1.15rem; font-weight:700; color:#0F172A;">{item['customer_count']} accounts</span><br>
                        <span style="font-size:0.8rem; color:#64748B;">{item['share_pct']:.1f}% of wholesale client base</span>
                    </div>
                </div>
                <hr style="margin:12px 0; border:none; border-top:1px solid #E2E8F0;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:12px;">
                    <div>
                        <strong style="font-size:0.85rem; color:#059669;">Dominant / High Expenditure Categories:</strong><br>
                        <span style="font-size:0.9rem; color:#0F172A;">{item['high_categories_str']}</span>
                    </div>
                    <div>
                        <strong style="font-size:0.85rem; color:#DC2626;">Under-Indexed Expenditure Categories:</strong><br>
                        <span style="font-size:0.9rem; color:#0F172A;">{item['low_categories_str']}</span>
                    </div>
                </div>
                <div style="margin-bottom:14px;">
                    <strong style="font-size:0.85rem; color:#334155;">Behavioral Spending Pattern:</strong>
                    <p style="font-size:0.88rem; color:#475569; margin:4px 0 0 0; line-height:1.5;">{item['behavior_summary']}</p>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; background-color:#F8FAFC; padding:16px; border-radius:6px; border:1px solid #E2E8F0;">
                    <div>
                        <strong style="font-size:0.85rem; color:#0284C7;">Recommended Commercial & Merchandising Strategy:</strong>
                        <p style="font-size:0.85rem; color:#334155; margin:6px 0 0 0; line-height:1.5;">{item['commercial_strategy']}</p>
                    </div>
                    <div>
                        <strong style="font-size:0.85rem; color:#0D9488;">Inventory & Replenishment Operational Guidelines:</strong>
                        <p style="font-size:0.85rem; color:#334155; margin:6px 0 0 0; line-height:1.5;">{item['inventory_strategy']}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Strategic validation disclaimer
    st.info(
        "Governance & Commercial Validation Note: The strategic initiatives outlined above represent data-driven suggestions "
        "derived exclusively from mathematical spending clustering. Real-world execution should be calibrated against account payment "
        "histories, logistical route density, and seasonal margin targets."
    )


# Page 6: Data Explorer
def render_data_explorer():
    curr_k = st.session_state["selected_k"]

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Records & Export</div>
            <h2>Wholesale Customer Data Explorer</h2>
            <p style="color:#475569;">Interactive record inspection, multi-attribute filtering, and verified CSV downloads.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    scaled_df, comparison_df, models_dict, pca_tuple = get_cached_clustering_pipeline()
    if models_dict is None or curr_k not in models_dict:
        st.error("Cluster assignment data is not ready.")
        return

    kmeans_model, predictions_df = models_dict[curr_k]
    cols_to_pull = ["customer_id", "Channel", "Region", "prediction"] + SPENDING_FEATURES
    explorer_df = predictions_df.select(cols_to_pull).toPandas()

    # Friendly label mappings
    explorer_df["Channel_Name"] = explorer_df["Channel"].map(CHANNEL_LABELS).fillna("Unknown")
    explorer_df["Region_Name"] = explorer_df["Region"].map(REGION_LABELS).fillna("Other")

    # Filter Bar
    st.markdown("**Search & Filter Controls**")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        search_id = st.text_input("Search Customer ID", placeholder="e.g. 0, 10, 42")
    with f_col2:
        cluster_options = ["All"] + sorted([str(c) for c in explorer_df["prediction"].unique()])
        selected_cluster = st.selectbox("Filter by Cluster", cluster_options)
    with f_col3:
        channel_options = ["All"] + list(CHANNEL_LABELS.values())
        selected_channel = st.selectbox("Filter by Channel", channel_options)
    with f_col4:
        region_options = ["All"] + list(REGION_LABELS.values())
        selected_region = st.selectbox("Filter by Region", region_options)

    # Apply Filters
    filtered_df = explorer_df.copy()
    if search_id.strip():
        try:
            cid = int(search_id.strip())
            filtered_df = filtered_df[filtered_df["customer_id"] == cid]
        except ValueError:
            st.warning("Customer ID must be an integer.")

    if selected_cluster != "All":
        filtered_df = filtered_df[filtered_df["prediction"] == int(selected_cluster)]

    if selected_channel != "All":
        filtered_df = filtered_df[filtered_df["Channel_Name"] == selected_channel]

    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region_Name"] == selected_region]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(explorer_df):,} customer records**")

    # Pagination controls
    page_size = st.selectbox("Rows per page", [15, 25, 50, 100], index=0)
    total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
    curr_page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (curr_page_num - 1) * page_size
    end_idx = start_idx + page_size

    display_subset = filtered_df.iloc[start_idx:end_idx]

    if filtered_df.empty:
        st.warning("No records match the specified search and filter criteria.")
    else:
        st.dataframe(
            display_subset.style.format({
                "Fresh": "${:,.0f}",
                "Milk": "${:,.0f}",
                "Grocery": "${:,.0f}",
                "Frozen": "${:,.0f}",
                "Detergents_Paper": "${:,.0f}",
                "Delicassen": "${:,.0f}"
            }),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Data Export Center")
    st.markdown("Download verified analytical artifacts directly from Spark computations:")

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_customers = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Customer Assignments CSV",
            data=csv_customers,
            file_name=f"meridian_customer_assignments_k{curr_k}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        profile_df = compute_cluster_profiles(predictions_df, spending_cols=SPENDING_FEATURES)
        csv_profiles = profile_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cluster Profiles Summary CSV",
            data=csv_profiles,
            file_name=f"meridian_cluster_profiles_k{curr_k}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d3:
        csv_eval = comparison_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Model Evaluation Benchmark CSV",
            data=csv_eval,
            file_name="meridian_kmeans_evaluation_benchmark.csv",
            mime="text/csv",
            use_container_width=True
        )


# Page 7: Methodology
def render_methodology():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Engineering & Theory</div>
            <h2>Analytical Workflow & System Methodology</h2>
            <p style="color:#475569;">Comprehensive technical documentation of the machine learning pipeline and architectural decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        ("1. Business Context & Objective",
         "Wholesale distribution is characterized by substantial heterogeneity in client transaction volume and commodity preferences. "
         "The objective is discovering coherent customer archetypes using unsupervised machine learning to tailor merchandising, delivery logistics, "
         "and pricing contracts without requiring pre-labeled training targets."),

        ("2. Dataset Taxonomy & Feature Isolation",
         "The dataset comprises 440 client records across 8 attributes. Two features (Channel and Region) represent operational categorizations and are "
         "intentionally quarantined from feature vectors. Only the six empirical spending categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen) "
         "are fed into the mathematical clustering space."),

        ("3. Non-Destructive Data Integrity Verification",
         "Our ingestion layer enforces zero data modification without verification. Rigorous checks validate that record count equals 440, all spending "
         "values are non-negative numeric floats/integers, null/NaN counts are exactly 0, and duplicate records are explicitly verified."),

        ("4. Exploratory Distribution Analysis",
         "Wholesale spending features exhibit strong positive right-skewness. Median values are systematically lower than means, demonstrating that the market "
         "is driven by a high-spending enterprise minority alongside a broader base of moderate accounts."),

        ("5. IQR Anomaly & Outlier Preservation",
         "Tukey's 1.5 × IQR rule identifies potential outliers. In contrast to standard classification workflows where outliers are often trimmed, wholesale "
         "outliers represent high-value enterprise accounts. Removing them would blind the business to its highest-grossing customers."),

        ("6. Spark MLlib VectorAssembler",
         "Apache Spark MLlib requires numerical feature vectors. We utilize pyspark.ml.feature.VectorAssembler with handleInvalid='error' to concatenate the "
         "six spending columns into a single 6-dimensional dense vector."),

        ("7. Feature Standardization via StandardScaler",
         "K-Means computes Euclidean distances. Categories with large absolute variance (e.g., Fresh with values up to 112,151) would naturally dominate "
         "categories with smaller ranges (e.g., Delicassen with max 47,943). StandardScaler scales each feature to mean 0 and unit standard deviation 1."),

        ("8. Unsupervised K-Means Formulation",
         "We train pyspark.ml.clustering.KMeans using Euclidean distance. The objective minimizes the Within-Set Sum of Squared Errors (WSSSE) across k centroids. "
         "To ensure exact scientific reproducibility, the random seed is pinned to 42."),

        ("9. Silhouette Score Evaluation via ClusteringEvaluator",
         "ClusteringEvaluator calculates the Silhouette score using squaredEuclidean distance. The score contrasts average within-cluster distance (a) "
         "with average nearest-cluster distance (b) via (b - a) / max(a, b), bounded between -1.0 and +1.0."),

        ("10. Multi-Model Comparison & Complexity Trade-Offs",
         "Candidate models for k in [2, 6] are systematically evaluated. While k=2 maximizes mathematical separation (0.900368), it creates an extreme "
         "imbalance (427 vs 13). The recommended commercial candidate is k=3 (0.740622), isolating meaningful business segments without single-customer artifacts."),

        ("11. Dynamic Archetype Profiling",
         "Cluster profiles are derived by aggregating mean, median, and relative indices directly from assigned records. Segment labels are assigned dynamically "
         "based on empirical spending intensity rather than hardcoded cluster indices."),

        ("12. Dimensionality Reduction via Spark MLlib PCA",
         "To project the 6-dimensional scaled space into a 2D visualization, Spark MLlib PCA is fitted with k=2. Principal components 1 and 2 capture "
         "over 72% of total cumulative variance. Coordinates and cluster predictions are strictly bound by customer_id to guarantee row alignment."),

        ("13. Evidence-Backed Strategy Derivation",
         "Actionable merchandising recommendations (bundled contracts, bulk volume discounts, cold-chain scheduling) are linked to empirical high-spending "
         "and low-spending categories for each segment."),

        ("14. Governance, Limitations & Roadmap",
         "Unsupervised clustering identifies correlational spending patterns, not causal determinants or firmographic identities. Future roadmap enhancements "
         "include incorporating longitudinal order frequency and time-series replenishment seasonality.")
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div class="meridian-panel">
                <h4 style="color:#0F172A; margin-bottom:8px;">{title}</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.6; margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Global Layout Router
def main():
    render_sidebar()

    current_page = st.session_state["current_page"]

    if current_page == "Overview":
        render_overview()
    elif current_page == "Exploratory Data Analysis":
        render_eda()
    elif current_page == "Model Evaluation":
        render_evaluation()
    elif current_page == "Customer Segmentation":
        render_segmentation()
    elif current_page == "Business Insights":
        render_business_insights()
    elif current_page == "Data Explorer":
        render_data_explorer()
    elif current_page == "Methodology":
        render_methodology()
    else:
        render_overview()

    # Enterprise Footer
    st.markdown(
        f"""
        <div class="meridian-footer">
            <strong>{PRODUCT_NAME}</strong> &bull; {PRODUCT_BADGE} &bull; Powered by Apache Spark MLlib & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
