"""
Meridian Wholesale Analytics - Visualizations Module
Creates enterprise-grade Plotly interactive charts with consistent color palettes,
slate-900 typography, subtle gridlines, and clean responsive layouts.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import (
    SPENDING_FEATURES,
    CLUSTER_COLOR_PALETTE,
    THEME_COLORS,
)
from src.utils import get_cluster_color, get_cluster_color_map


# Enterprise Layout Defaults
DEFAULT_PLOT_FONT = dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12, color="#0F172A")
DEFAULT_MARGIN = dict(l=40, r=40, t=50, b=40)


def plot_spending_distributions(
    df_pandas: pd.DataFrame,
    category: str
) -> go.Figure:
    """
    Creates a dual-panel distribution view for a spending category:
    Left: Histogram with Kernel Density curve
    Right: Detailed box plot showing median, quartiles, and high-value points.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{category} Spending Frequency", f"{category} Distribution Boxplot"],
        horizontal_spacing=0.12
    )

    series = df_pandas[category].dropna()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=40,
            marker=dict(color="#0284C7", line=dict(color="#0369A1", width=1)),
            name="Frequency",
            opacity=0.85
        ),
        row=1, col=1
    )

    # Boxplot
    fig.add_trace(
        go.Box(
            y=series,
            name=category,
            marker=dict(color="#0D9488"),
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
            boxmean=True
        ),
        row=1, col=2
    )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        showlegend=False,
        height=380
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", title_text="Spending Amount (Monetary Units)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", title_text="Customer Count", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", title_text="Spending Amount", row=1, col=2)

    return fig


def plot_iqr_outliers_bar(outlier_df: pd.DataFrame) -> go.Figure:
    """
    Displays the count and proportion of IQR-flagged high-value spending outliers per category.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=outlier_df["Feature"],
            y=outlier_df["Total Outliers"],
            text=[f"{count} ({pct}%)" for count, pct in zip(outlier_df["Total Outliers"], outlier_df["Outlier %"])],
            textposition="auto",
            marker=dict(
                color="#0284C7",
                line=dict(color="#0369A1", width=1.5)
            ),
            name="Outlier Count"
        )
    )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=360,
        title=dict(text="Identified High-Spending Customer Outliers per Category (1.5 × IQR Rule)", font=dict(size=14, color="#0F172A")),
        xaxis=dict(showgrid=False, title="Spending Category"),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Number of Outlier Records")
    )
    return fig


def plot_correlation_heatmap(
    df_pandas: pd.DataFrame,
    cols: List[str] = SPENDING_FEATURES
) -> go.Figure:
    """
    Generates an interactive correlation matrix heatmap for the 6 wholesale spending features.
    """
    corr = df_pandas[cols].corr().round(3)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=cols,
            y=cols,
            colorscale="Blues",
            zmin=-0.2,
            zmax=1.0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=12, color="#0F172A"),
            hoverongaps=False,
            colorbar=dict(title="Pearson r", len=0.8)
        )
    )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=dict(l=60, r=40, t=50, b=60),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=420,
        title=dict(text="Wholesale Spending Feature Correlation Matrix", font=dict(size=14, color="#0F172A")),
        xaxis=dict(tickangle=-25)
    )
    return fig


def plot_silhouette_curve(
    comparison_df: pd.DataFrame,
    selected_k: int,
    highest_k: Optional[int] = None,
    recommended_k: Optional[int] = 3
) -> go.Figure:
    """
    Plots the candidate models' silhouette scores across k=2..6,
    visually highlighting the selected, highest silhouette, and recommended candidate models.
    """
    fig = go.Figure()

    # Main silhouette trend line
    fig.add_trace(
        go.Scatter(
            x=comparison_df["k"],
            y=comparison_df["Silhouette Score"],
            mode="lines+markers",
            line=dict(color="#0284C7", width=3),
            marker=dict(size=10, color="#0369A1"),
            name="Silhouette Score"
        )
    )

    # Highlight currently selected k
    selected_row = comparison_df[comparison_df["k"] == selected_k]
    if not selected_row.empty:
        fig.add_trace(
            go.Scatter(
                x=[selected_k],
                y=[selected_row.iloc[0]["Silhouette Score"]],
                mode="markers",
                marker=dict(size=16, color="#D97706", symbol="star", line=dict(color="#92400E", width=2)),
                name=f"Selected (k={selected_k})"
            )
        )

    # Highlight recommended business candidate
    if recommended_k is not None:
        rec_row = comparison_df[comparison_df["k"] == recommended_k]
        if not rec_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=[recommended_k],
                    y=[rec_row.iloc[0]["Silhouette Score"]],
                    mode="markers",
                    marker=dict(size=14, color="#0D9488", symbol="diamond", line=dict(color="#115E59", width=2)),
                    name=f"Recommended (k={recommended_k})"
                )
            )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=380,
        title=dict(text="Clustering Evaluation: Spark MLlib Silhouette Score vs. k", font=dict(size=14, color="#0F172A")),
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Number of Clusters (k)", dtick=1),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Silhouette Score (squaredEuclidean)", range=[0.2, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cluster_sizes_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Displays cluster size distribution across all evaluated models (k=2 to 6).
    """
    fig = go.Figure()

    # Stack or group cluster sizes for each k
    for idx, row in comparison_df.iterrows():
        k_val = row["k"]
        sizes = row["Cluster Sizes List"]
        for c_idx, size in enumerate(sizes):
            color = CLUSTER_COLOR_PALETTE[c_idx % len(CLUSTER_COLOR_PALETTE)]
            fig.add_trace(
                go.Bar(
                    x=[f"k={k_val}"],
                    y=[size],
                    name=f"Cluster {c_idx}" if idx == 0 else None,
                    showlegend=(idx == 0),
                    marker=dict(color=color),
                    hoverinfo="text",
                    text=f"{size} customers",
                    textposition="inside" if size > 30 else "none"
                )
            )

    fig.update_layout(
        barmode="stack",
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=380,
        title=dict(text="Cluster Size Distribution Across Candidate Models (k=2 to 6)", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Candidate Model"),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Customer Count (Total = 440)")
    )
    return fig


def plot_cluster_size_bar(profile_df: pd.DataFrame) -> go.Figure:
    """
    Generates a clear bar chart of customer counts per cluster for the selected model.
    """
    colors = [get_cluster_color(c) for c in profile_df["Cluster"]]

    fig = go.Figure(
        data=go.Bar(
            x=[f"Cluster {c}" for c in profile_df["Cluster"]],
            y=profile_df["Customer Count"],
            text=[f"{count} ({share}%)" for count, share in zip(profile_df["Customer Count"], profile_df["Share (%)"])],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#334155", width=1))
        )
    )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=340,
        title=dict(text="Customer Headcount by Cluster", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Segment Identifier"),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Customer Count")
    )
    return fig


def plot_cluster_distribution_donut(profile_df: pd.DataFrame) -> go.Figure:
    """
    Renders a clean customer share donut chart with consistent cluster colors.
    """
    colors = [get_cluster_color(c) for c in profile_df["Cluster"]]

    fig = go.Figure(
        data=go.Pie(
            labels=[f"Cluster {c}: {lbl}" for c, lbl in zip(profile_df["Cluster"], profile_df["Label"])],
            values=profile_df["Customer Count"],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent+label",
            hoverinfo="label+value+percent"
        )
    )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#FFFFFF",
        height=340,
        showlegend=False,
        title=dict(text="Customer Share Distribution (%)", font=dict(size=14, color="#0F172A"))
    )
    return fig


def plot_cluster_spending_grouped_bar(
    profile_df: pd.DataFrame,
    spending_cols: List[str] = SPENDING_FEATURES
) -> go.Figure:
    """
    Creates a grouped bar chart comparing average spending in monetary units across
    the 6 wholesale categories for each cluster.
    """
    fig = go.Figure()

    for _, row in profile_df.iterrows():
        cluster_id = int(row["Cluster"])
        color = get_cluster_color(cluster_id)
        means = [row[f"Mean_{col}"] for col in spending_cols]

        fig.add_trace(
            go.Bar(
                name=f"Cluster {cluster_id}: {row['Label'][:24]}...",
                x=spending_cols,
                y=means,
                marker=dict(color=color),
                text=[f"${m:,.0f}" for m in means],
                textposition="auto"
            )
        )

    fig.update_layout(
        barmode="group",
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=400,
        title=dict(text="Average Annual Wholesale Spending by Category (Monetary Units)", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Spending Category"),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Mean Annual Spending ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_standardized_profile_heatmap(
    profile_df: pd.DataFrame,
    spending_cols: List[str] = SPENDING_FEATURES
) -> go.Figure:
    """
    Standardizes cluster average spending for visual comparison across categories.
    Computes relative z-scores across clusters for each category.
    Uses a diverging palette centered at zero with explicit annotations.
    """
    # Extract mean matrix
    means_matrix = []
    cluster_labels = []
    for _, row in profile_df.iterrows():
        c_id = int(row["Cluster"])
        cluster_labels.append(f"Cluster {c_id}")
        means_matrix.append([row[f"Mean_{col}"] for col in spending_cols])

    matrix_arr = np.array(means_matrix, dtype=float)

    # Standardize column-wise (across clusters)
    # z = (mean - col_mean) / col_std
    z_matrix = np.zeros_like(matrix_arr)
    for j in range(matrix_arr.shape[1]):
        col_vals = matrix_arr[:, j]
        std = np.std(col_vals)
        if std == 0 or np.isnan(std):
            z_matrix[:, j] = 0.0
        else:
            z_matrix[:, j] = (col_vals - np.mean(col_vals)) / std

    z_matrix = np.round(z_matrix, 2)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=spending_cols,
            y=cluster_labels,
            colorscale="RdBu_r",
            zmid=0.0,
            text=z_matrix,
            texttemplate="%{text:+.2f}σ",
            textfont=dict(size=12, color="#0F172A"),
            colorbar=dict(title="Relative Std. Dev. (σ)", len=0.8)
        )
    )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=340,
        title=dict(text="Standardized Cluster Spending Profile (Relative σ Across Clusters)", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Spending Feature"),
        yaxis=dict(title="Customer Segment")
    )
    return fig


def plot_pca_scatter(
    aligned_pca_df: pd.DataFrame,
    variance_info: Dict[str, Any],
    spending_cols: List[str] = SPENDING_FEATURES
) -> go.Figure:
    """
    Interactive 2D PCA projection scatter plot.
    Guarantees exact alignment with cluster predictions and customer records.
    Hover tooltip shows customer attributes and spending details.
    """
    fig = go.Figure()

    pc1_pct = variance_info.get("pc1_variance", 0.0)
    pc2_pct = variance_info.get("pc2_variance", 0.0)

    clusters = sorted(aligned_pca_df["prediction"].unique())

    for cluster_id in clusters:
        cluster_points = aligned_pca_df[aligned_pca_df["prediction"] == cluster_id]
        color = get_cluster_color(cluster_id)

        custom_text = []
        for _, r in cluster_points.iterrows():
            txt = (
                f"Customer ID: {r['customer_id']}<br>"
                f"Cluster: {cluster_id}<br>"
                f"Fresh: ${r['Fresh']:,.0f}<br>"
                f"Milk: ${r['Milk']:,.0f}<br>"
                f"Grocery: ${r['Grocery']:,.0f}<br>"
                f"Frozen: ${r['Frozen']:,.0f}<br>"
                f"Detergents & Paper: ${r['Detergents_Paper']:,.0f}<br>"
                f"Delicassen: ${r['Delicassen']:,.0f}"
            )
            custom_text.append(txt)

        fig.add_trace(
            go.Scatter(
                x=cluster_points["pca_x"],
                y=cluster_points["pca_y"],
                mode="markers",
                name=f"Cluster {cluster_id} (n={len(cluster_points)})",
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.8,
                    line=dict(color="#FFFFFF", width=0.5)
                ),
                text=custom_text,
                hoverinfo="text"
            )
        )

    fig.update_layout(
        font=DEFAULT_PLOT_FONT,
        margin=DEFAULT_MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=480,
        title=dict(
            text=f"Principal Component Analysis (PCA) 2D Projection (Explains {variance_info.get('total_explained_variance', 0.0)}% Total Variance)",
            font=dict(size=14, color="#0F172A")
        ),
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=f"Principal Component 1 ({pc1_pct}% Variance)"),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=f"Principal Component 2 ({pc2_pct}% Variance)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
