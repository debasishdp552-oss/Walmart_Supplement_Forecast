"""
app.py
======
Streamlit deployment for the supplement demand-forecasting + inventory
planning project.

Run locally:
    streamlit run app.py

Tabs:
    1. Overview  - dataset KPIs, category/product/location breakdown
    2. Forecast  - per-product model comparison (walk-forward MAPE) +
                   next-8-week forecast chart using each product's winning model
    3. Planning  - reorder point, safety stock, EOQ, stockout probability,
                   and a forward inventory simulation
"""

import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from planning import PlanningInputs, run_planning, simulate_inventory

st.set_page_config(page_title="Supplement Demand & Inventory Planning", layout="wide")

ARTIFACT_PATH = "artifacts/artifacts.pkl"
PROCESSED_PATH = "data/processed.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])
    return df


@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


import os
if not (os.path.exists(ARTIFACT_PATH) and os.path.exists(PROCESSED_PATH)):
    st.error("Model artifacts not found. Run `python train_pipeline.py` first "
             "(it builds data/processed.csv and artifacts/artifacts.pkl).")
    st.stop()

df = load_data()
artifacts = load_artifacts()
metrics_table = artifacts["metrics_table"]
forecasts = artifacts["forecasts"]
test_horizon = artifacts["test_horizon"]
last_date = pd.to_datetime(artifacts["last_date"])

st.title("Supplement Retail Demand Forecasting & Inventory Planning")
st.caption(
    "Weekly SKU-level demand forecasting (ETS / Prophet / "
    "pooled XGBoost, selected per product via walk-forward CV) with a downstream "
    "reorder-point and stockout-risk layer."
)

tab_overview, tab_forecast, tab_planning = st.tabs(["Overview", "Forecast", "Planning"])

# ------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
    c2.metric("Total Units Sold", f"{df['Units Sold'].sum():,.0f}")
    c3.metric("Products Tracked", df["Product Name"].nunique())
    c4.metric("Avg Return Rate", f"{df['Return_Rate'].mean() * 100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        weekly = df.groupby("Date")["Units Sold"].sum().reset_index()
        fig = px.line(weekly, x="Date", y="Units Sold", title="Total Weekly Units Sold (All Products)")
        st.plotly_chart(fig, width="stretch")

        cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(cat_rev, x="Category", y="Revenue", title="Revenue by Category")
        st.plotly_chart(fig2, width="stretch")

    with col_b:
        prod_rev = df.groupby("Product Name")["Revenue"].sum().sort_values(ascending=False).reset_index()
        fig3 = px.bar(prod_rev, x="Revenue", y="Product Name", orientation="h", title="Revenue by Product")
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, width="stretch")

        loc_units = df.groupby("Location")["Units Sold"].sum().reset_index()
        fig4 = px.pie(loc_units, names="Location", values="Units Sold", title="Units Sold Share by Market")
        st.plotly_chart(fig4, width="stretch")

# ------------------------------------------------------------------
# TAB 2: FORECAST
# ------------------------------------------------------------------
with tab_forecast:
    st.subheader("Model benchmark (walk-forward CV, mean MAPE %)")
    st.dataframe(metrics_table.style.highlight_min(
        subset=[c for c in metrics_table.columns if c not in ("Best Model", "Best MAPE")],
        axis=1, color="#c6f6d5"
    ), width="stretch")

    st.markdown("---")
    product = st.selectbox("Select a product", sorted(df["Product Name"].unique()))

    prod_hist = df[df["Product Name"] == product].sort_values("Date")
    winner = metrics_table.loc[product, "Best Model"]
    winner_mape = metrics_table.loc[product, "Best MAPE"]
    fc_values = forecasts[product]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=test_horizon, freq="W-MON")

    col1, col2, col3 = st.columns(3)
    col1.metric("Winning model", winner)
    col2.metric("Backtest MAPE", f"{winner_mape:.2f}%")
    col3.metric(f"Avg forecast (next {test_horizon}w)", f"{np.mean(fc_values):.0f} units/wk")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prod_hist["Date"].iloc[-52:], y=prod_hist["Units Sold"].iloc[-52:],
                              mode="lines", name="Actual (last 52 weeks)"))
    fig.add_trace(go.Scatter(x=future_dates, y=fc_values, mode="lines+markers",
                              name=f"Forecast ({winner})", line=dict(dash="dash")))
    fig.update_layout(title=f"{product}: Actual vs {test_horizon}-Week Forecast", xaxis_title="Week", yaxis_title="Units Sold")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Per-fold MAPE detail"):
        detail = artifacts["results"][product]["fold_details"]
        if detail:
            st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
        else:
            st.info("Not enough history for a full walk-forward backtest on this product; using ETS's internal seasonal-naive fallback.")

# ------------------------------------------------------------------
# TAB 3: PLANNING
# ------------------------------------------------------------------
with tab_planning:
    st.subheader("Reorder point, safety stock & stockout risk")
    st.caption(
        "This dataset has no real inventory/lead-time feed, so lead time, service "
        "level, and current stock are your inputs below. Demand mean/variability "
        "come from the model's own forecast for the selected product."
    )

    plan_product = st.selectbox("Product", sorted(df["Product Name"].unique()), key="plan_product")
    prod_hist = df[df["Product Name"] == plan_product].sort_values("Date")
    fc_values = np.array(forecasts[plan_product])
    avg_weekly_demand = float(np.mean(fc_values))
    # empirical demand uncertainty from recent actuals (last 26 weeks) rather than
    # from the point forecast alone, since a point forecast has ~zero spread
    demand_std = float(prod_hist["Units Sold"].iloc[-26:].std())
    asp = float(prod_hist["Price"].iloc[-1])

    left, right = st.columns([1, 1.4])

    with left:
        lead_time = st.slider("Supplier lead time (weeks)", 1, 8, 2)
        service_level = st.select_slider(
            "Target service level", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
            format_func=lambda x: f"{x*100:.1f}%"
        )
        current_inventory = st.number_input("Current on-hand inventory (units)", min_value=0, value=int(avg_weekly_demand * 3))
        ordering_cost = st.number_input("Ordering cost per PO ($)", min_value=1.0, value=50.0)
        holding_pct = st.slider("Annual holding cost (% of unit cost)", 5, 40, 20) / 100

        inputs = PlanningInputs(
            avg_weekly_demand=avg_weekly_demand,
            demand_std=demand_std,
            lead_time_weeks=lead_time,
            service_level=service_level,
            current_inventory=current_inventory,
            unit_cost=asp,
            ordering_cost=ordering_cost,
            holding_cost_pct=holding_pct,
        )
        plan = run_planning(inputs)

        st.metric("Reorder Point (ROP)", f"{plan['reorder_point']:.0f} units")
        st.metric("Safety Stock", f"{plan['safety_stock']:.0f} units")
        st.metric("Economic Order Qty (EOQ)", f"{plan['eoq']:.0f} units")
        st.metric("Stockout Probability (this lead time)", f"{plan['stockout_probability']:.1f}%")
        st.metric("Weeks of Supply on Hand", f"{plan['weeks_of_supply']:.1f} wk")

        if plan["need_to_order"]:
            st.error(f"Current inventory ({current_inventory:.0f}) is at/below ROP "
                      f"({plan['reorder_point']:.0f}) - place a purchase order now.")
        else:
            st.success(f"Current inventory ({current_inventory:.0f}) is above ROP "
                        f"({plan['reorder_point']:.0f}). No order needed yet.")

    with right:
        sim = simulate_inventory(
            forecast=fc_values,
            starting_inventory=current_inventory,
            reorder_qty=plan["eoq"],
            rop=plan["reorder_point"],
            lead_time_weeks=lead_time,
        )
        future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=test_horizon, freq="W-MON")
        sim_df = pd.DataFrame({"Date": future_dates, "Projected On-Hand": sim["on_hand"], "Forecast Demand": fc_values})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_df["Date"], y=sim_df["Projected On-Hand"], mode="lines+markers", name="Projected On-Hand"))
        fig.add_hline(y=plan["reorder_point"], line_dash="dot", line_color="orange", annotation_text="Reorder Point")
        fig.add_hline(y=0, line_color="red")
        if sim["stockout_weeks"]:
            fig.add_trace(go.Scatter(
                x=[future_dates[i] for i in sim["stockout_weeks"]],
                y=[sim["on_hand"][i] for i in sim["stockout_weeks"]],
                mode="markers", marker=dict(color="red", size=12, symbol="x"), name="Stockout week"
            ))
        fig.update_layout(title=f"{plan_product}: {test_horizon}-Week Inventory Projection (no-reorder-until-ROP policy)",
                           xaxis_title="Week", yaxis_title="Units")
        st.plotly_chart(fig, width="stretch")

        if sim["stockout_weeks"]:
            st.warning(f"Projected stockout in week(s): "
                        f"{', '.join(future_dates[i].strftime('%Y-%m-%d') for i in sim['stockout_weeks'])} "
                        f"under current inventory and lead-time assumptions.")
        else:
            st.info("No stockouts projected over the forecast horizon at current settings.")

        sim_display = sim_df.copy()
        sim_display[["Projected On-Hand", "Forecast Demand"]] = sim_display[["Projected On-Hand", "Forecast Demand"]].round(1)
        st.dataframe(sim_display, width="stretch", hide_index=True)

st.markdown("---")
st.caption("Business logic assumptions: demand uncertainty = std of last 26 weeks of actuals; "
           "lead-time demand assumed approximately Normal for the stockout-probability and safety-stock "
           "calculations (standard newsvendor approach). Retrain via `python train_pipeline.py` when new "
           "sales data lands.")
