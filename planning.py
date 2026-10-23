"""
planning.py
===========
Converts a demand forecast into inventory/business decisions:
  - Safety stock & Reorder Point (ROP) under configurable lead time and
    service level
  - Stockout probability given a current on-hand inventory position
  - Economic Order Quantity (EOQ) using ASP as a cost proxy
  - A simple forward inventory simulation (on-hand vs forecasted demand)
    to flag which future weeks would stock out under a "do nothing" policy

No real inventory/lead-time data ships with this dataset, so lead time,
holding-cost %, and ordering cost are user-configurable assumptions (the
same pattern used in the Walmart project's supply-planning layer). Demand
uncertainty is estimated empirically from each product's forecast errors
(walk-forward CV residual std) rather than assumed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class PlanningInputs:
    avg_weekly_demand: float
    demand_std: float          # std of weekly demand (or forecast error)
    lead_time_weeks: float
    service_level: float       # e.g. 0.95
    current_inventory: float
    unit_cost: float           # ASP - average selling price, used as cost proxy
    ordering_cost: float = 50.0     # cost per purchase order (assumption)
    holding_cost_pct: float = 0.20  # annual holding cost as % of unit cost


def z_score(service_level: float) -> float:
    return float(norm.ppf(service_level))


def safety_stock(demand_std: float, lead_time_weeks: float, service_level: float) -> float:
    z = z_score(service_level)
    return z * demand_std * np.sqrt(max(lead_time_weeks, 0.01))


def reorder_point(avg_weekly_demand: float, lead_time_weeks: float, ss: float) -> float:
    return avg_weekly_demand * lead_time_weeks + ss


def economic_order_quantity(annual_demand: float, ordering_cost: float, unit_cost: float, holding_cost_pct: float) -> float:
    holding_cost_per_unit = max(unit_cost * holding_cost_pct, 0.01)
    return float(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit))


def stockout_probability(current_inventory: float, avg_weekly_demand: float,
                          demand_std: float, lead_time_weeks: float) -> float:
    """P(demand during lead time > current inventory), assuming ~Normal
    lead-time demand. This is the classic newsvendor-style risk metric."""
    lt_demand_mean = avg_weekly_demand * lead_time_weeks
    lt_demand_std = demand_std * np.sqrt(max(lead_time_weeks, 0.01))
    if lt_demand_std <= 0:
        return 0.0 if current_inventory >= lt_demand_mean else 1.0
    z = (current_inventory - lt_demand_mean) / lt_demand_std
    return float(1 - norm.cdf(z))


def run_planning(inputs: PlanningInputs) -> dict:
    ss = safety_stock(inputs.demand_std, inputs.lead_time_weeks, inputs.service_level)
    rop = reorder_point(inputs.avg_weekly_demand, inputs.lead_time_weeks, ss)
    annual_demand = inputs.avg_weekly_demand * 52
    eoq = economic_order_quantity(annual_demand, inputs.ordering_cost, inputs.unit_cost, inputs.holding_cost_pct)
    p_stockout = stockout_probability(
        inputs.current_inventory, inputs.avg_weekly_demand, inputs.demand_std, inputs.lead_time_weeks
    )
    need_to_order = inputs.current_inventory <= rop
    weeks_of_supply = inputs.current_inventory / inputs.avg_weekly_demand if inputs.avg_weekly_demand > 0 else np.inf

    return {
        "safety_stock": round(ss, 1),
        "reorder_point": round(rop, 1),
        "eoq": round(eoq, 1),
        "stockout_probability": round(p_stockout * 100, 1),
        "need_to_order": bool(need_to_order),
        "weeks_of_supply": round(weeks_of_supply, 1),
        "z_score": round(z_score(inputs.service_level), 2),
    }


def simulate_inventory(forecast: np.ndarray, starting_inventory: float,
                        reorder_qty: float, rop: float, lead_time_weeks: float) -> dict:
    """Week-by-week simulation: inventory depletes by forecast demand each
    week; a replenishment order fires once inventory <= ROP and arrives
    after `lead_time_weeks`. Returns projected on-hand levels and any
    weeks flagged as stockouts (on-hand < 0, clipped to 0 for display)."""
    horizon = len(forecast)
    on_hand = np.zeros(horizon)
    stockout_weeks = []
    pending_orders = {}  # arrival_week -> qty
    inv = starting_inventory
    order_placed = False
    order_week = None

    for t in range(horizon):
        if t in pending_orders:
            inv += pending_orders[t]

        inv -= forecast[t]
        if inv < 0:
            stockout_weeks.append(t)
            inv = max(inv, -forecast[t])  # keep for visualization but flag as backorder
        on_hand[t] = inv

        if inv <= rop and not order_placed:
            arrival = t + max(int(round(lead_time_weeks)), 1)
            if arrival < horizon:
                pending_orders[arrival] = pending_orders.get(arrival, 0) + reorder_qty
            order_placed = True
            order_week = t

    return {
        "on_hand": on_hand,
        "stockout_weeks": stockout_weeks,
        "order_week": order_week,
    }
