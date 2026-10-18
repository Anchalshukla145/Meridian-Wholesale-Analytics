"""
Meridian Wholesale Analytics - Configuration Module
Defines core application parameters, file paths, feature schemas, and styling constants.
"""

from pathlib import Path
from typing import Dict, List

# Product Identity
PRODUCT_NAME = "Meridian Wholesale Analytics"
PRODUCT_BADGE = "Customer Intelligence Platform"
PRODUCT_TAGLINE = "Understand Your Customers. Discover Your Growth Opportunities."
PRODUCT_DESCRIPTION = (
    "Enterprise-grade wholesale customer segmentation engine powered by Apache Spark MLlib. "
    "Transform raw transactional spending into actionable customer archetypes, optimized "
    "inventory allocation, and data-driven commercial strategies."
)

# File Paths
# Primary: Project-relative resolution based on this config module's directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

DEFAULT_DATA_FILE = DATA_DIR / "Wholesale customers data.csv"
FALLBACK_WINDOWS_DATA_FILE = Path(r"D:\subjects\year4_sem7\Spark\CA2\project\data\Wholesale customers data.csv")

LANDING_IMAGE_FILE = ASSETS_DIR / "food_delivery.png"
FALLBACK_WINDOWS_IMAGE_FILE = Path(r"D:\subjects\year4_sem7\Spark\CA2\project\assets\food_delivery.png")

# Dataset Feature Schemas
CATEGORICAL_FEATURES: List[str] = ["Channel", "Region"]

SPENDING_FEATURES: List[str] = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

ALL_EXPECTED_COLUMNS: List[str] = CATEGORICAL_FEATURES + SPENDING_FEATURES

# Domain Mappings
CHANNEL_LABELS: Dict[int, str] = {
    1: "Horeca (Hotel/Restaurant/Café)",
    2: "Retail Channel"
}

REGION_LABELS: Dict[int, str] = {
    1: "Lisbon Metropolitan Region",
    2: "Oporto Region",
    3: "Other Domestic Regions"
}

# Machine Learning & Clustering Hyperparameters
K_MIN: int = 2
K_MAX: int = 6
DEFAULT_K: int = 3
RECOMMENDED_BUSINESS_K: int = 3
RANDOM_SEED: int = 42

SPARK_SCALED_FEATURE_COL: str = "scaled_features"
SPARK_RAW_FEATURE_COL: str = "features"
SPARK_PREDICTION_COL: str = "prediction"
SPARK_PCA_COL: str = "pca_features"

# Visual Theme & Design Tokens
# Consistent cluster colors across all application views
CLUSTER_COLOR_PALETTE: List[str] = [
    "#0284C7",  # Deep Sky Blue
    "#0D9488",  # Rich Teal
    "#D97706",  # Warm Amber
    "#7C3AED",  # Royal Violet
    "#DB2777",  # Vivid Rose
    "#059669",  # Emerald Green
]

THEME_COLORS = {
    "background": "#F8FAFC",      # Slate 50
    "card_bg": "#FFFFFF",         # Pure White
    "card_border": "#E2E8F0",     # Slate 200
    "text_primary": "#0F172A",   # Slate 900
    "text_secondary": "#475569", # Slate 600
    "text_muted": "#94A3B8",     # Slate 400
    "primary": "#0284C7",        # Sky 600
    "primary_dark": "#0369A1",   # Sky 700
    "accent_teal": "#0D9488",    # Teal 600
    "accent_amber": "#D97706",   # Amber 600
    "danger": "#EF4444",         # Red 500
    "success": "#10B981",        # Emerald 500
}

# Navigation Pages
PAGES = [
    "Overview",
    "Exploratory Data Analysis",
    "Model Evaluation",
    "Customer Segmentation",
    "Business Insights",
    "Data Explorer",
    "Methodology"
]
