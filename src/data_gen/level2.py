from __future__ import annotations

import re
import random
import math
import warnings
import xarray as xr

from src.utils.random_sampling import *
from src.utils.data_io_and_subset import (
    get_region_spec,
    load_variable,
    load_variable_auto_experiment, 
    load_depth_levels,
    load_area_weights,
    subset_time,
    subset_region,
    subset_depth,
    subset_season,
    subset_single_depth,
    subset_data,
)
from src.utils.numerical_diagnostics import (
    compute_basic_statistic,
    compute_linear_trend,
    compute_trend_significance,
    evaluate_condition_or_compare,
    parse_threshold,
    area_weighted_mean,
    annual_area_weighted_mean,
    compute_monthly_climatology,
    compute_seasonal_climatology,
    compute_anomaly,
    filter_complete_seasons,
)
from src.utils.domain_knowledge import (
    get_variable_unit,
    experiment_for_year,
    experiment_for_period,
    resolve_implicit_depth,
    apply_implicit_depth,
    resolve_implicit_time_period,
    resolve_qualified_area_criteria,
    resolve_descriptive_adj_criteria,
    iter_criteria_with_fallbacks,
    build_enso_phase_labels,
    build_qualified_area_mask,
    compute_potential_density,
    compute_stratification_index,
    compute_mixed_layer_depth,
    compute_thermocline_depth,
    compute_layer_thickness,
    select_enso_phase,
)
 

# ===================================================================
# Per-task solvers
# ===================================================================
 
def _solve_L2_1(data_meta, task_meta):
    """Global mean {variable} anomaly in {implicit_depth} during boreal winter,
    relative to {clim_time_period} climatology."""
    sid = data_meta["source_id"]
    var = task_meta["variable_code"]
    root = data_meta["data_root"]
 
    # 1. Climatology period
    clim_start, clim_end = resolve_implicit_time_period(task_meta["clim_time_period_code"])
    clim_data = load_variable_auto_experiment(sid, var, clim_start, clim_end, root)
    clim_data = apply_implicit_depth(clim_data, task_meta["implicit_depth"])
    djf_clim = compute_seasonal_climatology(clim_data, "DJF")

    # 2. Analysis period
    ana_data = load_variable_auto_experiment(
        sid, var, task_meta["start_month"], task_meta["end_month"], root,
    )
    ana_data = apply_implicit_depth(ana_data, task_meta["implicit_depth"])
    ana_djf = subset_season(ana_data, "DJF")

    # 3. Subtract climatology -> anomaly time series; mean over time
    anomaly = (ana_djf - djf_clim).mean(dim="time", skipna=True)

    if "depth" in anomaly.dims:
        anomaly = anomaly.mean(dim="depth", skipna=True)

    area = load_area_weights(data_meta)
    result = area_weighted_mean(anomaly, area)
    return {"type": "scalar", "value": float(result.values), "unit": get_variable_unit(var),}
 
 
def _solve_L2_2(data_meta, task_meta):
    """Is {implicit_depth} mean {variable} anomaly over {region} positive
    (end-of-century vs pre-industrial)?"""
    sid = data_meta["source_id"]
    var = task_meta["variable_code"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
 
    # Pre-industrial climatology (1850-1900, historical)
    pi = load_variable_auto_experiment(sid, var, "1850-01", "1900-12", root)
    pi = apply_implicit_depth(pi, task_meta["implicit_depth"])
    pi = subset_region(pi, region_spec)
    pi_clim = compute_monthly_climatology(pi)

    # End-of-century (2081-2100, ssp245)
    eoc = load_variable_auto_experiment(sid, var, "2081-01", "2100-12", root)
    eoc = apply_implicit_depth(eoc, task_meta["implicit_depth"])
    eoc = subset_region(eoc, region_spec)

    # TERM_4: deseasonalize against the matching calendar-month mean.
    anomaly = compute_anomaly(eoc, pi_clim).mean(dim="time", skipna=True)
    if "depth" in anomaly.dims:
        anomaly = anomaly.mean(dim="depth", skipna=True)
    
    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
    val = float(area_weighted_mean(anomaly, area).values)
    return {"type": "boolean", "value": bool(val > 0) if not math.isnan(val) else None, 
            "anomaly": val, "anomaly_unit": get_variable_unit(var),}
 
 
def _solve_L2_3(data_meta, task_meta):
    """Mean {term_for_depth_layer} depth in {implicit_region} during {season}
    over {time_period}."""
    sid = data_meta["source_id"]
    exp = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["implicit_region_code"])
    term = task_meta["term_for_depth_layer"]
    season = task_meta["season"]
    start, end = task_meta["start_month"], task_meta["end_month"]
 
    thetao = load_variable(exp, sid, "thetao", start, end, root)
    thetao = subset_region(thetao, region_spec)

    if term == "mixed layer":
        so = load_variable(exp, sid, "so", start, end, root)
        so = subset_region(so, region_spec)
        depth_field_full = compute_mixed_layer_depth(
            thetao, so, delta_rho=0.03, ref_depth=10.0,
        )
    elif term == "thermocline":
        depth_field_full = compute_thermocline_depth(
            thetao, search_range=(10.0, 500.0),
        )
    else:
        raise ValueError(f"Unknown depth-layer term: '{term}'")
    
    depth_field = subset_season(depth_field_full, season)
    depth_field = filter_complete_seasons(depth_field, season)

    if depth_field.sizes.get("time", 0) == 0:
        return {"type": "scalar", "value": float("nan"),
                "note": f"no complete {season} season in {task_meta['time_period']}"}

    # Time mean -> area-weighted spatial mean over the implicit region
    depth_field_tmean = depth_field.mean(dim="time", skipna=True)
    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
    val = float(area_weighted_mean(depth_field_tmean, area).values)
    return {"type": "scalar", "value": val, "unit": "m"}
 
 
def _solve_L2_4(data_meta, task_meta):
    """{region} mean {ratio} ratio anomaly in {implicit_depth} relative to
    {clim_time_period} climatology."""
    sid = data_meta["source_id"]
    v1 = task_meta["ratio_var1_code"]
    v2 = task_meta["ratio_var2_code"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
 
    def _load_pair(start, end):
        d1 = load_variable_auto_experiment(sid, v1, start, end, root)
        d2 = load_variable_auto_experiment(sid, v2, start, end, root)
        d1 = apply_implicit_depth(d1, task_meta["implicit_depth"])
        d2 = apply_implicit_depth(d2, task_meta["implicit_depth"])
        d1 = subset_region(d1, region_spec)
        d2 = subset_region(d2, region_spec)
        return d1, d2
    
    clim_start, clim_end = resolve_implicit_time_period(task_meta["clim_time_period_code"])
    d1_clim, d2_clim = _load_pair(clim_start, clim_end)
    ratio_clim_series = d1_clim / d2_clim
    # TERM_4: calendar-month climatology of the ratio (deseasonalized).
    ratio_clim = compute_monthly_climatology(ratio_clim_series)

    ana_start = task_meta.get("ana_start_month", "1995-01")
    ana_end = task_meta.get("ana_end_month", "2014-12")
    d1_ana, d2_ana = _load_pair(ana_start, ana_end)
    ratio_ana_monthly = d1_ana / d2_ana

    anomaly = compute_anomaly(ratio_ana_monthly, ratio_clim).mean(dim="time", skipna=True)
 
    if "depth" in anomaly.dims:
        anomaly = anomaly.mean(dim="depth", skipna=True)

    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
    val = float(area_weighted_mean(anomaly, area).values)
    return {"type": "scalar", "value": val}
 
 
def _solve_L2_5(data_meta, task_meta):
    """Compare {term_for_thickness} thickness between two regions."""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    criteria = resolve_qualified_area_criteria(task_meta["term_for_thickness"])
    var_code, op, threshold = criteria["variable"], criteria["op"], criteria["threshold"]
 
    it_start, it_end = resolve_implicit_time_period(task_meta["implicit_time_period_code"])

    def _thickness_for_region(region_code):
        da = load_variable_auto_experiment(sid, var_code, it_start, it_end, root)
        da = apply_implicit_depth(da, task_meta["implicit_depth"])
        da = subset_region(da, get_region_spec(region_code))
        thick = compute_layer_thickness(da, threshold, op, method="bounded").mean(dim="time", skipna=True)
        return thick

    t1 = float(_thickness_for_region(task_meta["region_1_code"]).mean(skipna=True).values)
    t2 = float(_thickness_for_region(task_meta["region_2_code"]).mean(skipna=True).values)

    # If both regions produce 0 thickness (e.g. implicit_depth is outside the OMZ band),
    # the comparison is degenerate; signal via NaN so the outer loop can reject.
    if t1 == 0.0 and t2 == 0.0:
        return {"type": "boolean", "value": None, "lhs": 0.0, "rhs": 0.0,
                "note": "degenerate: both regions have zero qualifying thickness"}

    result = evaluate_condition_or_compare(t1, task_meta["comparison_operator_code"], t2)
    return {"type": "boolean", "value": bool(result), "lhs": t1, "rhs": t2, "thickness_unit": "m",
            "threshold": threshold, "threshold_unit": "mol/m³"}
 
 
def _solve_L2_6(data_meta, task_meta):
    """Is {implicit_depth} dissolved oxygen in {implicit_region} hypoxic on
    average during the recent period?"""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["implicit_region_code"])
    start = task_meta.get("start_month", "1995-01")
    end = task_meta.get("end_month",   "2014-12")
 
    o2 = load_variable_auto_experiment(sid, "o2", start, end, root)
    o2 = apply_implicit_depth(o2, task_meta["implicit_depth"])
    o2 = subset_region(o2, region_spec)
 
    o2_tmean = o2.mean(dim="time", skipna=True)
    if "depth" in o2_tmean.dims:
        o2_tmean = o2_tmean.mean(dim="depth", skipna=True)

    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
    mean_o2 = float(area_weighted_mean(o2_tmean, area).values)

    criteria = resolve_qualified_area_criteria("hypoxia")
    threshold = criteria["threshold"]

    return {"type": "boolean", "value": mean_o2 < threshold,
            "mean_o2": mean_o2, "threshold": threshold, "threshold_unit": "mol/m³",}
 
 
def _solve_L2_7(data_meta, task_meta):
    """Global mean stratification in the upper 200 m during {time_period}."""
    sid = data_meta["source_id"]
    exp = data_meta["experiment_id"]
    root = data_meta["data_root"]
    start, end = task_meta["start_month"], task_meta["end_month"]

    # restrict to the 0-200m inside compute_stratification_index
    thetao = load_variable(exp, sid, "thetao", start, end, root)
    so = load_variable(exp, sid, "so", start, end, root)
 
    strat = compute_stratification_index(
        thetao, so, depth_range=(0, 200), reduce="mean",
    )
    strat_tmean = strat.mean(dim="time", skipna=True)
    area = load_area_weights(data_meta)
    val = float(area_weighted_mean(strat_tmean, area).values)

    return {"type": "scalar", "value": val, "unit": "s-2"}
 
 
def _solve_L2_8(data_meta, task_meta):
    """Is linear trend in {region} mean {variable} at {depth} significant?"""
    sid = data_meta["source_id"]
    exp = data_meta["experiment_id"]
    var = task_meta["variable_code"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    start, end = task_meta["start_month"], task_meta["end_month"]
 
    da = load_variable(exp, sid, var, start, end, root)
    da = subset_region(da, region_spec)
    if "depth" in da.dims:
        da = subset_single_depth(da, task_meta["depth_code"])
 
    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
    ts = area_weighted_mean(da, area)
    
    # Significance at α = 0.1 per task spec
    alpha = 0.1
    sig = compute_trend_significance(ts, alpha=alpha)
    return {
        "type": "boolean",
        "value": bool(sig["is_significant"].values),
        "slope": float(sig["slope"].values),
        "slope_unit": f"{get_variable_unit(var)} per month",
        "p_value": float(sig["p_value"].values),
        "alpha": alpha,
    }
 
 
def _solve_L2_9(data_meta, task_meta):
    """Mean stratification in {implicit_region} {implicit_depth} during
    {enso_phase} climatology over {clim_time_period}."""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["implicit_region_code"])
    clim_start, clim_end = resolve_implicit_time_period(task_meta["clim_time_period_code"])

    # 1. ENSO phase labels (fixed 1971-2000 baseline + persistence filter).
    phase_labels = build_enso_phase_labels(sid, clim_start, clim_end, root)

    # 2. Load T, S over clim window and select phase months.
    thetao = load_variable_auto_experiment(sid, "thetao", clim_start, clim_end, root)
    so     = load_variable_auto_experiment(sid, "so",     clim_start, clim_end, root)
    thetao_phase = select_enso_phase(thetao, phase_labels, task_meta["enso_phase"])
    so_phase     = select_enso_phase(so,     phase_labels, task_meta["enso_phase"])

    if thetao_phase.sizes.get("time", 0) == 0:
        return {"type": "scalar", "value": float("nan"),
                "note": f"no {task_meta['enso_phase']} months in {task_meta['clim_time_period']}"}

    # 3. Restrict to implicit region, compute N² in implicit-depth window.
    thetao_r = subset_region(thetao_phase, region_spec)
    so_r     = subset_region(so_phase,     region_spec)
    depth_range = resolve_implicit_depth(task_meta["implicit_depth"]) or (0, 200)
    strat = compute_stratification_index(thetao_r, so_r, depth_range=depth_range, reduce="mean")

    # 4. Time mean → area-weighted regional mean.
    strat_tmean = strat.mean(dim="time", skipna=True)
    area = subset_region(load_area_weights(data_meta), region_spec)
    val = float(area_weighted_mean(strat_tmean, area).values)

    return {
        "type": "scalar", "value": val, "unit": "s-2",
        "enso_phase": task_meta["enso_phase"],
        "n_phase_months": int(thetao_phase.sizes["time"]),
    }
    
 
def _solve_L2_10(data_meta, task_meta):
    """Mean of {variable} within {qualified_area} of {region} over {implicit_depth}
    during {clim_time_period}."""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    var = task_meta["variable_code"]
    region_spec = get_region_spec(task_meta["region_code"])
    criteria = resolve_qualified_area_criteria(task_meta["qualified_area"])
    clim_start, clim_end = resolve_implicit_time_period(task_meta["clim_time_period_code"])

    # 1. Load target variable.
    try:
        target = load_variable_auto_experiment(sid, var, clim_start, clim_end, root)
    except FileNotFoundError:
        return {"type": "scalar", "value": float("nan"),
                "note": f"target variable '{var}' unavailable for {sid}"}
    target = apply_implicit_depth(target, task_meta["implicit_depth"])
    target = subset_region(target, region_spec)

    # 2. Build qualified-area mask (criterion fallbacks handled inside helper).
    mask, chosen_crit_var = build_qualified_area_mask(
        sid, criteria, clim_start, clim_end, root,
        region_spec=region_spec, implicit_depth=task_meta["implicit_depth"],
    )
    if mask is None:
        return {"type": "scalar", "value": float("nan"),
                "note": f"no criterion variable available or active for '{task_meta['qualified_area']}'"}

    # Look up op/threshold for the chosen criterion variable (for reporting).
    chosen_op, chosen_threshold = next(
        ((op, t) for cv, op, t in iter_criteria_with_fallbacks(criteria) if cv == chosen_crit_var),
        (None, None),
    )

    # 3. Time-mean target → mask → area-weighted regional mean.
    target_tmean = target.mean(dim="time", skipna=True)
    if "depth" in target_tmean.dims:
        target_tmean = target_tmean.mean(dim="depth", skipna=True)
    target_masked = target_tmean.where(mask)

    area = subset_region(load_area_weights(data_meta), region_spec)
    val = float(area_weighted_mean(target_masked, area).values)

    return {
        "type": "scalar", "value": val, "unit": get_variable_unit(var),
        "qualified_area": task_meta["qualified_area"],
        "criteria_variable_used": chosen_crit_var,
        "criteria_threshold": chosen_threshold,
        "criteria_threshold_unit": get_variable_unit(chosen_crit_var),
        "criteria_operator": chosen_op,
    }
 
 
def _solve_L2_11(data_meta, task_meta):
    """Is seawater more {descriptive_adj} in recent period than pre-industrial?"""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    criteria = resolve_descriptive_adj_criteria(task_meta["descriptive_adj"])
    region_spec = get_region_spec(task_meta["implicit_region_code"])

    area_full = load_area_weights(data_meta)
    area = subset_region(area_full, region_spec)

    def _period_area_weighted_mean(crit_var, start, end):
        try:
            da = load_variable_auto_experiment(sid, crit_var, start, end, root)
        except FileNotFoundError:
            return float("nan")
        da = apply_implicit_depth(da, task_meta["implicit_depth"])
        da = subset_region(da, region_spec)

        # time mean -> depth mean -> area-weighted spatial mean
        if any(da.sizes[d] == 0 for d in da.dims):
            return float("nan")
        
        da_t = da.mean(dim="time", skipna=True)
        if "depth" in da_t.dims:
            da_t = da_t.mean(dim="depth", skipna=True)
        return float(area_weighted_mean(da_t, area).values)

    pi_mean = recent_mean = None
    chosen_var = None

    for crit_var, _, _ in iter_criteria_with_fallbacks(criteria):
        pi_mean     = _period_area_weighted_mean(crit_var, "1850-01", "1900-12")
        recent_mean = _period_area_weighted_mean(crit_var, "1995-01", "2014-12")
        # Require finite values for a meaningful comparison
        if math.isnan(pi_mean) or math.isnan(recent_mean):
            continue
        chosen_var = crit_var
        break

    if chosen_var is None:
        return {"type": "boolean", "value": None,
                "note": f"no criterion variable available for '{task_meta['descriptive_adj']}'"}
    
    delta = recent_mean - pi_mean
    if criteria["direction"] == "decrease":
        is_more = delta < 0
    elif criteria["direction"] == "increase":
        is_more = delta > 0
    else:
        raise ValueError(f"Unknown direction: {criteria['direction']}")
    
    return {
        "type": "boolean",
        "value": bool(is_more),
        "pi_mean": pi_mean,
        "recent_mean": recent_mean,
        "delta": delta,
        "delta_unit":  get_variable_unit(chosen_var),
        "criterion_variable_used": chosen_var,
    }

    
def _solve_L2_12(data_meta, task_meta):
    """Change in {region} {stat_op} {variable} anomaly: end-of-century minus
    pre-industrial climatology."""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    var = task_meta["variable_code"]
    stat_code = task_meta["statistical_operator_code"]
    region_spec = get_region_spec(task_meta["region_code"])

    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
 
    def _scalar_for_period(start, end):
        da = load_variable_auto_experiment(sid, var, start, end, root)
        da = subset_region(da, region_spec)
        da_t = da.mean(dim="time", skipna=True)
        if "depth" in da_t.dims:
            da_t = da_t.mean(dim="depth", skipna=True)

        # Mean → area-weighted; all other stats → unweighted on spatial grid.
        # Weighting percentiles/median/min/max/range would change their meaning.
        if stat_code == "mean":
            return float(area_weighted_mean(da_t, area).values)
        else:
            return float(compute_basic_statistic(da_t, stat_code).values)
 
    pi_val  = _scalar_for_period("1850-01", "1900-12")  # TEMP_3
    eoc_val = _scalar_for_period("2081-01", "2100-12")  # end-of-century, ssp245
    change = eoc_val - pi_val

    return {
        "type": "scalar",
        "value": change,
        "unit": get_variable_unit(var),
        "pi_value":  pi_val,
        "eoc_value": eoc_val,
        "statistical_operator": task_meta.get("statistical_operator", stat_code),
    }
 
 
def _solve_L2_13(data_meta, task_meta):
    """{implicit_region} {variable} climatology for {clim_time_period}."""
    sid = data_meta["source_id"]
    root = data_meta["data_root"]
    var = task_meta["variable_code"]
    region_spec = get_region_spec(task_meta["implicit_region_code"])
 
    clim_start, clim_end = resolve_implicit_time_period(task_meta["clim_time_period_code"])
    da = load_variable_auto_experiment(sid, var, clim_start, clim_end, root)
    da = subset_region(da, region_spec)

    # Time mean -> depth mean (if any) -> area-weighted spatial mean
    da_t = da.mean(dim="time", skipna=True)
    if "depth" in da_t.dims:
        da_t = da_t.mean(dim="depth", skipna=True)

    area = load_area_weights(data_meta)
    area = subset_region(area, region_spec)
    val = float(area_weighted_mean(da_t, area).values)

    return {
        "type": "scalar",
        "value": val,
        "unit": get_variable_unit(var),
        "variable": task_meta["variable"],
        "clim_time_period": task_meta["clim_time_period"],
    }
 
 
def _solve_L2_14(data_meta, task_meta):
    """Is {depth_range} layer of {region} {descriptive_adj} during {time_period}?"""
    sid = data_meta["source_id"]
    exp = data_meta["experiment_id"]
    root = data_meta["data_root"]
    criteria = resolve_descriptive_adj_criteria(task_meta["descriptive_adj"])
    region_spec = get_region_spec(task_meta["region_code"])
    start, end = task_meta["start_month"], task_meta["end_month"]

    area_full = load_area_weights(data_meta)
    area = subset_region(area_full, region_spec)

    chosen = None
    mean_val = None

    for crit_var, op, threshold in iter_criteria_with_fallbacks(criteria):
        try:
            da = load_variable(exp, sid, crit_var, start, end, root)
        except FileNotFoundError:
            continue
        
        if any(da.sizes[d] == 0 for d in da.dims):
            continue
        
        da = subset_region(da, region_spec)
        if "depth" in da.dims:
            da = subset_depth(da, (task_meta["depth_lower"], task_meta["depth_upper"]))

        # time mean -> depth mean (if 3-D) -> area-weighted regional mean
        da_t = da.mean(dim="time", skipna=True)
        if "depth" in da_t.dims:
            da_t = da_t.mean(dim="depth", skipna=True)
        candidate = float(area_weighted_mean(da_t, area).values)

        if math.isnan(candidate):
            continue

        chosen = (crit_var, op, threshold)
        mean_val = candidate
        break

    if chosen is None:
        return {"type": "boolean", "value": None,
                "note": f"no criterion variable available for '{task_meta['descriptive_adj']}'"}

    crit_var, op, threshold = chosen
    is_adj = bool(evaluate_condition_or_compare(mean_val, op, threshold))

    return {
        "type": "boolean",
        "value": is_adj,
        "mean_value": mean_val,
        "mean_value_unit": get_variable_unit(crit_var),
        "criterion_variable_used": crit_var,
        "criterion_threshold":     threshold,
        "criterion_operator":      op,
    }
 
 
def _solve_L2_15(data_meta, task_meta):
    """Is {qualified_area} present in annual-mean {implicit_depth} waters of
    {region} during {time_period}?"""
    sid = data_meta["source_id"]
    exp = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    criteria = resolve_qualified_area_criteria(task_meta["qualified_area"])
    start, end = task_meta["start_time"], task_meta["end_time"]

    chosen = None
    present = None

    for crit_var, op, threshold in iter_criteria_with_fallbacks(criteria):
        try:
            da = load_variable(exp, sid, crit_var, start, end, root)
        except FileNotFoundError:
            continue
        
        if any(da.sizes[d] == 0 for d in da.dims):
            continue
        
        da = apply_implicit_depth(da, task_meta["implicit_depth"])
        da = subset_region(da, region_spec)

        # Annual mean per grid cell: resample monthly -> yearly, NaN-aware
        annual = da.resample(time="YE").mean(skipna=True)

        # Reduce depth (if still 3-D) so the comparison is at one depth-mean
        # value per (year, lat, lon)
        if "depth" in annual.dims:
            annual = annual.mean(dim="depth", skipna=True)

        # Skip fallback if the array is entirely NaN (variable exists but
        # contains no valid data in this region/depth/time)
        if not bool(annual.notnull().any().values):
            chosen = None
            break

        mask = evaluate_condition_or_compare(annual, op, threshold)
        present = bool(mask.any().values)
        chosen = (crit_var, op, threshold)
        break

    if chosen is None:
        return {"type": "boolean", "value": None,
                "note": f"no criterion variable available for '{task_meta['qualified_area']}' / no valid data for qualified areas"}

    crit_var, op, threshold = chosen
    return {
        "type": "boolean",
        "value": present,
        "criterion_variable_used": crit_var,
        "criterion_threshold": threshold,
        "criterion_threshold_unit": get_variable_unit(crit_var),
        "criterion_operator": op,
    }


# ===================================================================
# Main entry point
# ===================================================================

_SOLVERS = {
    "L2_1":  _solve_L2_1,   "L2_2":  _solve_L2_2,
    "L2_3":  _solve_L2_3,   "L2_4":  _solve_L2_4,
    "L2_5":  _solve_L2_5,   "L2_6":  _solve_L2_6,
    "L2_7":  _solve_L2_7,   "L2_8":  _solve_L2_8,
    "L2_9":  _solve_L2_9,   "L2_10": _solve_L2_10,
    "L2_11": _solve_L2_11,  "L2_12": _solve_L2_12,
    "L2_13": _solve_L2_13,  "L2_14": _solve_L2_14,
    "L2_15": _solve_L2_15,
}


def generate_level2_tasks(config, task_id):
    """
    Generate level 2 tasks based on the task ID.

    Returns:
        - task_id (str): The ID of the task.
        - data_metadata (dict): Describe the data requirements for the task, including variables, institutions, experiments, etc.
        - task_metadata (dict): Describe the specific parameters for the task template, such as variable, region, time period, etc.
        - answer_metadata (dict): Include the type of expected answer, computed answer, and any relevant metadata about the answer (e.g. units, etc.)
    """

    experiment_dict = config.cmip6.experiment
    experiment_id = random.choice(list(experiment_dict.keys()))
    source_id = random.choice(experiment_dict[experiment_id])

    data_metadata = {
        "experiment_id": experiment_id,
        "source_id": source_id,
        "data_root": config.cmip6.root,
    } # to add for each task: file_name based on the selected variable(s) and time period


    if task_id == "L2_1":
        # What is the mean of global seasonal {variable} anomaly in the {implicit_depth} during recent boreal winter for {time_period}, relative to the seasonal {clim_time_period} climatology?
        var_code, var_nl = sample_variable_with_depth()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=36, max_months=180)
        clim_time_code, clim_time_period = sample_clim_time_period()
        implicit_depth = sample_implicit_depth(
            surface=True) if var_code in ["chl", "phycos"] else sample_implicit_depth()

        task_metadata = {
            "variable_code": var_code,
            "variable": var_nl,
            "implicit_depth": implicit_depth,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
            "clim_time_period_code": clim_time_code,
            "clim_time_period": clim_time_period,
        }

        ambiguous_terms = ["mean", "seasonal", "anomaly", implicit_depth, "boreal winter", "climatology",]
        pattern = r"\b(18|19|20)\d{2}\s*[-–—]\s*(18|19|20)\d{2}\b"
        if not re.search(pattern, clim_time_period):
            ambiguous_terms.append(clim_time_period)
        task_metadata.update({"ambiguous_terms": ambiguous_terms})

        # data_metadata must cover BOTH the analysis window and the
        # historical clim window; experiment_id may need to span historical+ssp.
        clim_start = f"{clim_time_code.split('-')[0]}-01"
        clim_end   = f"{clim_time_code.split('-')[1]}-12"
        overall_start = min(start_month, clim_start)
        overall_end   = max(end_month,   clim_end)
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id": var_code,
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L2_2":
        # Is the {implicit_depth} mean {variable} anomaly over the {region} during end of century annual conditions positive relative to the pre-industrial climatology?
        var_code, var_nl = sample_variable_with_depth()
        if var_code == "chl":
            implicit_depth = sample_implicit_depth(surface=True)
        else:
            implicit_depth = sample_implicit_depth()
        region_code, region = sample_region()

        task_metadata = {
            "implicit_depth": implicit_depth,
            "variable_code": var_code,
            "variable": var_nl,
            "region_code": region_code, 
            "region": region,
        }
        task_metadata.update({"ambiguous_terms": [
            implicit_depth, "mean", "anomaly", "end of century", "pre-industrial", "climatology"
        ]})
        data_metadata.update({"experiment_id": ["historical", "ssp245"], "variable_id": var_code, "start_time": "1850-01", "end_time": "2100-12"})


    elif task_id == "L2_3":
        # What is the mean {term_for_depth_layer} depth in the {implicit_region} during seasonal {season} conditions over {time_period}?
        layer_var_codes, term_for_depth_layer = sample_term_for_depth_layer()
        implicit_region_code, implicit_region = sample_implicit_region()
        season = sample_season()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=36, max_months=180)

        task_metadata = {
            "term_for_depth_layer": term_for_depth_layer,
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "season": season,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "mean", term_for_depth_layer, implicit_region, "seasonal", 
        ]})
        data_metadata.update({"variable_id": layer_var_codes, "start_time": start_month, "end_time": end_month})

    
    elif task_id == "L2_4":
        # What is the mean anomaly of {ratio} ratio in the {region} over the {implicit_depth} during recent annual conditions relative to the {clim_time_period} climatology?
        region_code, region = sample_region()
        ratio_var_dict = sample_ratio()
        implicit_depth = sample_implicit_depth()
        clim_time_code, clim_time_period = sample_clim_time_period()

        # "recent annual conditions" = last 20 years of the historical record
        ana_start_month, ana_end_month = "1995-01", "2014-12"
        ana_time_period = "1995 ~ 2014"

        task_metadata = {
            "region_code": region_code, 
            "region": region,
            "ratio_var1_code": ratio_var_dict["var1_code"],
            "ratio_var2_code": ratio_var_dict["var2_code"],
            "ratio": ratio_var_dict["ratio_nl"],
            "implicit_depth": implicit_depth,
            "clim_time_period_code": clim_time_code,
            "clim_time_period": clim_time_period,
            "ana_start_month": ana_start_month,
            "ana_end_month": ana_end_month,
            "ana_time_period": ana_time_period,
        }

        ambiguous_terms = ["mean", "anomaly", "ratio", implicit_depth, "recent", "climatology",]
        pattern = r"\b(18|19|20)\d{2}\s*[-–—]\s*(18|19|20)\d{2}\b"
        if not re.search(pattern, clim_time_period):
            ambiguous_terms.append(clim_time_period)
        task_metadata.update({"ambiguous_terms": ambiguous_terms})

        # Analysis is fixed to 1995-2014; clim is one of the historical clim
        # windows. Both lie inside historical, so pin experiment_id to that.
        clim_start = f"{clim_time_code.split('-')[0]}-01"
        clim_end   = f"{clim_time_code.split('-')[1]}-12"
        overall_start = min(ana_start_month, clim_start)
        overall_end   = max(ana_end_month,   clim_end)
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id":   ratio_var_dict["var1_code"],
            "variable_id_2": ratio_var_dict["var2_code"],
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L2_5":
        # Is the {region_1} {term_for_thickness} mean thickness in the {implicit_depth} during {implicit_time_period} conditions {comparison_operator} in the same depth layer and time period of {region_2}?
        while True:
            region_1_code, region_1 = sample_region()
            region_2_code, region_2 = sample_region()
            if region_1 != region_2:
                break

        term_for_thickness = sample_term_for_thickness()
        # Restrict to depth ranges where hypoxia/OMZ is physically relevant
        implicit_depth = random.choice([
            "subsurface", "upper ocean", "intermediate depth", "deep ocean",
        ])
        implicit_time_period_code, implicit_time_period = sample_implicit_time_period()
        comparison_code, comparison_op = sample_comparison_operator()

        task_metadata = {
            "region_1_code": region_1_code,
            "region_1": region_1,
            "region_2_code": region_2_code,
            "region_2": region_2,
            "term_for_thickness": term_for_thickness,
            "implicit_depth": implicit_depth,
            "implicit_time_period_code": implicit_time_period_code,
            "implicit_time_period": implicit_time_period,
            "comparison_operator_code": comparison_code, 
            "comparison_operator": comparison_op,
        }
        task_metadata.update({"ambiguous_terms": [
            term_for_thickness, "mean", "thickness", implicit_depth, implicit_time_period
        ]})
        # Implicit-time-period codes resolve to historical-era windows; the
        # solver loads only that window, so pin experiment_id and start/end
        # accordingly (was previously left as a random experiment_id with no
        # time window, which could leave data_metadata advertising ssp245).
        it_start, it_end = resolve_implicit_time_period(implicit_time_period_code)
        data_metadata.update({
            "experiment_id": experiment_for_period(it_start, it_end),
            "variable_id": "o2",
            "start_time": it_start,
            "end_time":   it_end,
        })


    elif task_id == "L2_6":
        # Is the {implicit_depth} dissolved oxygen in the {implicit_region} hypoxic on average during the recent period?
        implicit_depth = sample_implicit_depth()
        implicit_region_code, implicit_region = sample_implicit_region()

        task_metadata = {
            "implicit_depth": implicit_depth,
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "start_month": "1995-01",
            "end_month": "2014-12",
        }
        task_metadata.update({"ambiguous_terms": [
            implicit_depth, implicit_region, "hypoxic", "average", "recent"
        ]})
        data_metadata.update({"experiment_id": "historical", "variable_id": "o2", "start_time": "1995-01", "end_time": "2014-12"})
        

    elif task_id == "L2_7":
        # What is the global mean stratification in the upper 200 m during {time_period}?
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        task_metadata = {
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "mean", "stratification"
        ]})
        data_metadata.update({"variable_id": "thetao", "variable_id_2": "so", "start_time": start_month, "end_time": end_month})
        

    elif task_id == "L2_8":
        # Is the linear trend in {region} mean {variable} at {depth} significant over {time_period}?
        region_code, region = sample_region()
        var_code, var_nl = sample_variable_with_depth()
        depth_code, depth_nl = sample_depth()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=240)

        task_metadata = {
            "region_code": region_code, 
            "region": region,
            "variable_code": var_code,
            "variable": var_nl,
            "depth_code": depth_code,
            "depth": depth_nl,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "mean", "significant"
        ]})
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})
        

    elif task_id == "L2_9":
        # What is the mean stratification in the {implicit_region} {implicit_depth} during the {enso_phase} within {clim_time_period}?
        implicit_region_code, implicit_region = sample_implicit_region()
        implicit_depth = sample_implicit_depth(surface=True)
        enso_phase = sample_enso_phase()
        clim_time_code, clim_time_period = sample_clim_time_period()

        clim_start_yr = int(clim_time_code.split("-")[0])
        clim_end_yr   = int(clim_time_code.split("-")[1])

        task_metadata = {
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "implicit_depth": implicit_depth,
            "enso_phase": enso_phase,
            "clim_time_period_code": clim_time_code,
            "clim_time_period": clim_time_period,
        }

        ambiguous_terms = ["mean", "stratification", implicit_region, implicit_depth, enso_phase]
        pattern = r"\b(18|19|20)\d{2}\s*[-–—]\s*(18|19|20)\d{2}\b"
        if not re.search(pattern, clim_time_period):
            ambiguous_terms.append(clim_time_period)
        task_metadata.update({"ambiguous_terms": ambiguous_terms})

        data_metadata.update({
            "experiment_id": "historical",
            "variable_id":   "thetao",
            "variable_id_2": "so",
            # Span from ENSO baseline start through clim window end
            "start_time": f"{min(1971, clim_start_yr)}-01",
            "end_time":   f"{max(2000, clim_end_yr)}-12",
        })


    elif task_id == "L2_10":
        # What is the mean of {variable} in the {implicit_depth} over {clim_time_period} within the average {qualified_area} of the {region}?
        var_code, var_nl = sample_variable_with_depth()
        clim_time_code, clim_time_period = sample_clim_time_period()
        qualified_area, primary_crit_var = sample_qualified_area()
        region_code, region = sample_region()
        implicit_depth = sample_implicit_depth()

        criteria = resolve_qualified_area_criteria(qualified_area)
        crit_priority = [criteria["variable"]] + [fb["variable"] for fb in criteria.get("fallbacks", [])]
        
        task_metadata = {
            "variable_code": var_code,
            "variable": var_nl,
            "clim_time_period_code": clim_time_code,
            "clim_time_period": clim_time_period,
            "qualified_area": qualified_area,
            "criteria_variable_primary": primary_crit_var,
            "criteria_variable_priority": crit_priority,
            "region_code": region_code, 
            "region": region,
            "implicit_depth": implicit_depth,
        }

        ambiguous_terms = ["mean", implicit_depth, qualified_area]
        pattern = r"\b(18|19|20)\d{2}\s*[-–—]\s*(18|19|20)\d{2}\b"
        if not re.search(pattern, clim_time_period):
            ambiguous_terms.append(clim_time_period)
        task_metadata.update({"ambiguous_terms": ambiguous_terms})

        clim_start = f"{clim_time_code.split('-')[0]}-01"
        clim_end   = f"{clim_time_code.split('-')[1]}-12"
        data_metadata.update({"experiment_id": "historical", "variable_id": var_code, "variable_id_2": primary_crit_var, "start_time": clim_start,"end_time": clim_end,})


    elif task_id == "L2_11":
        # Is the seawater over the {implicit_depth} in the {implicit_region} more {descriptive_adj} in the recent period than in the pre-industrial period?
        implicit_depth = sample_implicit_depth()
        implicit_region_code, implicit_region = sample_implicit_region()
        descriptive_adj, primary_var = sample_descriptive_adj()

        criteria = resolve_descriptive_adj_criteria(descriptive_adj)
        crit_priority = [criteria["variable"]] + [fb["variable"] for fb in criteria.get("fallbacks", [])]

        task_metadata = {
            "implicit_depth": implicit_depth,
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "descriptive_adj": descriptive_adj,
            "criteria_variable_primary": primary_var,
            "criteria_variable_priority": crit_priority,
        }
        task_metadata.update({"ambiguous_terms": [
            implicit_depth, implicit_region, descriptive_adj, "recent", "pre-industrial"
        ]})
        data_metadata.update({"experiment_id": "historical", "variable_id": primary_var, "variable_id_fallbacks": crit_priority[1:], "start_time": "1850-01", "end_time": "2014-12",})


    elif task_id == "L2_12":
        # What is the change in the {statistical_operator} of {variable} over {region}, at the end of century relative to the pre-industrial, where the statistic is computed after first taking the time mean (and depth mean if applicable) and then applying the spatial reduction?
        region_code, region = sample_region()
        stat_code, stat_op = sample_statistical_operator()
        var_code, var_nl = sample_variable()

        task_metadata = {
            "region_code": region_code, 
            "region": region,
            "statistical_operator_code": stat_code, 
            "statistical_operator": stat_op,
            "variable_code": var_code,
            "variable": var_nl,
        }
        task_metadata.update({"ambiguous_terms": [
            "change", "end of century", "pre-industrial"
        ]})
        data_metadata.update({"experiment_id": ["historical", "ssp245"], "variable_id": var_code, "start_time": "1850-01", "end_time": "2100-12",})


    elif task_id == "L2_13":
        # What is the {implicit_region} {variable} climatology for {clim_time_period}?
        implicit_region_code, implicit_region = sample_implicit_region()
        var_code, var_nl = sample_variable()
        clim_time_code, clim_time_period = sample_clim_time_period()

        task_metadata = {
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "variable_code": var_code,
            "variable": var_nl,
            "clim_time_period_code": clim_time_code,
            "clim_time_period": clim_time_period,
        }

        ambiguous_terms = [implicit_region, "climatology"]
        pattern = r"\b(18|19|20)\d{2}\s*[-–—]\s*(18|19|20)\d{2}\b"
        if not re.search(pattern, clim_time_period):
            ambiguous_terms.append(clim_time_period)
        task_metadata.update({"ambiguous_terms": ambiguous_terms})

        clim_start = f"{clim_time_code.split('-')[0]}-01"
        clim_end   = f"{clim_time_code.split('-')[1]}-12"
        data_metadata.update({"experiment_id": "historical", "variable_id": var_code, "start_time": clim_start, "end_time": clim_end,})


    elif task_id == "L2_14":
        # Is the {depth_range} layer of {region} {descriptive_adj} during {time_period}?
        descriptive_adj, primary_var = sample_descriptive_adj()
        criteria = resolve_descriptive_adj_criteria(descriptive_adj)
        crit_priority = [criteria["variable"]] + [fb["variable"] for fb in criteria.get("fallbacks", [])]

        if descriptive_adj == "oligotrophic":
            max_depth_for_adj = 200
        elif descriptive_adj == "hypoxic":
            max_depth_for_adj = 1500
        else:
            max_depth_for_adj = 5500

        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, primary_var,)
        lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds, max_depth=max_depth_for_adj,)

        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=60)

        task_metadata = {
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "region_code": region_code, 
            "region": region,
            "descriptive_adj": descriptive_adj,
            "criteria_variable_primary": primary_var,
            "criteria_variable_priority": crit_priority,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            descriptive_adj
        ]})
        data_metadata.update({"variable_id": primary_var, "variable_id_fallbacks": crit_priority[1:],
                              "start_time": start_month, "end_time": end_month})
        

    elif task_id == "L2_15":
        # Is {qualified_area} present in the annual-mean {implicit_depth} waters of {region} during {time_period}?
        qualified_area, primary_var = sample_qualified_area()
        criteria = resolve_qualified_area_criteria(qualified_area)
        crit_priority = [criteria["variable"]] + [fb["variable"] for fb in criteria.get("fallbacks", [])]

        implicit_depth = sample_implicit_depth()
        region_code, region = sample_region()
        start_yr, end_yr, time_period = sample_time_period_yr(experiment=experiment_id, min_years=1, max_years=5)

        task_metadata = {
            "qualified_area": qualified_area,
            "criteria_variable_primary":  primary_var,
            "criteria_variable_priority": crit_priority,
            "implicit_depth": implicit_depth,
            "region_code": region_code, 
            "region": region,
            "start_time": start_yr,
            "end_time": end_yr,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            qualified_area, "annual-mean", implicit_depth
        ]})
        data_metadata.update({"variable_id": primary_var, "variable_id_fallbacks": crit_priority[1:], "start_time": start_yr, "end_time": end_yr})


    else:
        raise ValueError(f"Unknown task_id: '{task_id}'. Must be one of L2_1 to L2_15.")
    
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multiple fill values.*", category=xr.SerializationWarning)
        warnings.filterwarnings("ignore", message=".*All-NaN slice.*", category=RuntimeWarning)
        answer_metadata = _SOLVERS[task_id](data_metadata, task_metadata)
    
    val = answer_metadata.get("value")
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except TypeError:
        pass

    return {
        "task_id": task_id,
        "data_metadata": data_metadata,
        "task_metadata": task_metadata,
        "answer_metadata": answer_metadata,
    }