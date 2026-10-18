"""
Meridian Wholesale Analytics - Evidence-Based Business Recommendations Module
Derives contextual commercial strategies, inventory prioritization, and replenishment recommendations
directly from empirical cluster spending distributions.
"""

from typing import Dict, List, Any
import pandas as pd
from src.config import SPENDING_FEATURES


def generate_cluster_insights(
    profile_df: pd.DataFrame,
    overall_means: pd.Series
) -> List[Dict[str, Any]]:
    """
    Synthesizes empirical spending data for each cluster into structured, evidence-backed
    commercial intelligence, actionable merchandising strategies, and operational inventory guidelines.
    """
    insights_list = []

    for _, row in profile_df.iterrows():
        cluster_id = int(row["Cluster"])
        label = row["Label"]
        count = int(row["Customer Count"])
        share = float(row["Share (%)"])

        # Determine relative over-indexing and under-indexing
        ratios = {}
        means = {}
        for col in SPENDING_FEATURES:
            mean_val = float(row[f"Mean_{col}"])
            overall_val = float(overall_means[col])
            ratio = (mean_val / overall_val) if overall_val > 0 else 1.0
            ratios[col] = ratio
            means[col] = mean_val

        # Sort categories by relative ratio
        sorted_by_ratio = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        high_categories = [col for col, r in sorted_by_ratio if r >= 1.25]
        low_categories = [col for col, r in sorted_by_ratio if r < 0.75]
        moderate_categories = [col for col, r in sorted_by_ratio if 0.75 <= r < 1.25]

        # In case no category is >= 1.25 (e.g. low-volume cluster), top 2 relative categories
        if not high_categories:
            high_categories = [sorted_by_ratio[0][0]]

        if not low_categories:
            low_categories = [sorted_by_ratio[-1][0]]

        # Format high and low category text
        high_cat_str = ", ".join(c.replace("_", " ") for c in high_categories)
        low_cat_str = ", ".join(c.replace("_", " ") for c in low_categories)

        # Spending Behavior Summary
        top_cat_name = high_categories[0].replace("_", " ")
        top_cat_mean = means[high_categories[0]]
        behavior_summary = (
            f"Segment accounts for {share:.1f}% of total wholesale clients (n={count}). "
            f"Spending is heavily concentrated in {high_cat_str}, with top average annual expenditure "
            f"reaching ${top_cat_mean:,.0f} in {top_cat_name}. Conversely, purchasing volume is "
            f"substantially curtailed in {low_cat_str}."
        )

        # Commercial & Merchandising Strategy
        if "Grocery" in high_categories and "Detergents_Paper" in high_categories:
            commercial_strategy = (
                "Deploy bundled contracts combining high-turnover Grocery commodities with premium "
                "Detergents & Paper consumables. Introduce tiered volume rebates (e.g., quarterly threshold "
                "discounts) to incentivize consolidated purchasing across dry goods and packaging."
            )
            inventory_strategy = (
                "Establish bulk-pallet warehousing agreements with suppliers to secure preferential margin "
                "spreads on dry grocery items. High shelf-life allows optimized buffer stock without rapid "
                "spoilage risks; schedule recurring bi-weekly consolidated truckload deliveries."
            )
        elif "Fresh" in high_categories and "Frozen" in high_categories:
            commercial_strategy = (
                "Target accounts with guaranteed cold-chain delivery windows and dynamic morning arrival "
                "commitments. Offer pre-season supply commitments for fresh produce and blast-frozen protein batches."
            )
            inventory_strategy = (
                "Prioritize rapid cold-storage throughput with FIFO rotation to minimize perishable spoilage. "
                "Align local farm and cooperative procurement cycles directly with 48-hour order lead times."
            )
        elif "Fresh" in high_categories:
            commercial_strategy = (
                "Develop premium produce subscription tiers with quality grading guarantees. Cross-sell "
                "specialty olive oils, dressings, and complementary delicatessen lines to raise basket margin."
            )
            inventory_strategy = (
                "Operate tight just-in-time (JIT) replenishment to limit shrinkage on fresh vegetables, fruits, "
                "and greens. Reorder daily or tri-weekly based on rolling short-horizon predictive forecasts."
            )
        elif all(r < 0.9 for r in ratios.values()):
            commercial_strategy = (
                "Focus on order frequency stimulation and account retention rather than aggressive bulk deals. "
                "Offer lower minimum-order-value (MOV) thresholds paired with small consolidated delivery fees, "
                "or self-service digital catalog ordering."
            )
            inventory_strategy = (
                "Fulfill orders via shared warehouse routing and multi-stop localized delivery vans to preserve "
                "logistics margin on smaller drop sizes. Avoid dedicating exclusive warehouse reserve bays."
            )
        else:
            commercial_strategy = (
                f"Develop tailored cross-category packages pairing dominant category ({top_cat_name}) with "
                "under-indexed categories. Pilot promotional multi-buy incentives to test account elasticity."
            )
            inventory_strategy = (
                "Maintain balanced standard replenishment schedules with standard safety stock coverage. "
                "Monitor order volatility across seasons to prevent stockouts during peak ordering cycles."
            )

        insights_list.append({
            "cluster_id": cluster_id,
            "label": label,
            "customer_count": count,
            "share_pct": share,
            "high_categories": high_categories,
            "high_categories_str": high_cat_str,
            "low_categories": low_categories,
            "low_categories_str": low_cat_str,
            "behavior_summary": behavior_summary,
            "commercial_strategy": commercial_strategy,
            "inventory_strategy": inventory_strategy,
            "means": means,
            "ratios": ratios,
        })

    return insights_list
