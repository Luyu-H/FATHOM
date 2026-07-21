from __future__ import annotations

import math
import random
import warnings

import numpy as np
import xarray as xr

from src.utils.random_sampling import *
from src.utils.data_io_and_subset import (
    get_region_spec,
    get_spatial_dims,
    load_variable_auto_experiment,
    load_area_for_model,
    load_bases,
    load_bases_for_derived,
    load_depth_levels,
    subset_region,
)
from src.utils.numerical_diagnostics import (
    get_dz,
    compute_basic_statistic,
    compute_basic_statistic_with_area,
    regrid_to_grid,
    compute_linear_trend,
    compute_trend_significance,
    area_weighted_mean,
    annual_area_weighted_mean,
    compute_climatology,
    compute_monthly_climatology,
    compute_anomaly,
    compute_pearson_correlation,
    compute_volume_fraction,
)
from src.utils.domain_knowledge import (
    apply_implicit_depth,
    experiment_for_period,
    resolve_implicit_time_period,
    resolve_qualified_area_criteria,
    resolve_descriptive_noun,
    resolve_derived_variable,
    is_cumulative,
    compute_derived_from_bases,
    compute_derived_at_depth,
    build_descriptive_noun_mask,
    build_enso_phase_labels,
    build_qualified_area_mask,
    compute_mixed_layer_depth,
    compute_thermocline_depth,
    select_enso_phase,
    classify_qualified_time,
    ENSO_BASELINE_START,
    ENSO_BASELINE_END,
)


# ===================================================================
# Per-task solvers
# ===================================================================

def _solve_L3_1(data_meta, task_meta):
    """Decadal linear trend in {implicit_depth} {implicit_derived_variable}
    anomaly within {qualified_area} inside {region} over {time_period_yr}."""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    dc, spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    qa_criteria = resolve_qualified_area_criteria(task_meta["qualified_area"])
    start, end = task_meta["start_year"], task_meta["end_year"]
    
    # 1. Regional cell area (used by both compute_derived and the spatial reduction).
    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    # 2. Load bases and compute derived variable. Pass area so inventory / threshold_volume become per-cell totals while keeping spatial dims.
    bases = load_bases_for_derived(
        sid, spec, start, end, root,
        region_spec=region_spec,
        implicit_depth=task_meta["implicit_depth"],
    )
    if bases is None:
        return {"type": "scalar", "value": float("nan"),
                "note": "incompatible: sea floor with inventory-type derived"}
    derived = compute_derived_from_bases(bases, dc, spec, area=area)

    # 3. Qualified-area 2-D mask.
    mask, chosen_qa = build_qualified_area_mask(
        sid, qa_criteria, start, end, root,
        region_spec=region_spec,
        implicit_depth=task_meta["implicit_depth"],
    )
    if mask is None:
        return {"type": "scalar", "value": float("nan"),
                "note": f"no qualifying region for '{task_meta['qualified_area']}'"}
    
    # 4. Apply mask, then collapse to monthly regional time series.
    derived = derived.where(mask)
    cumulative = is_cumulative(spec)

    if cumulative:
        # Sum on depth (dz-weighted) and on space.
        if "depth" in derived.dims:
            dz = get_dz(derived, depth_dim="depth")
            derived = (derived * dz).sum(dim="depth", skipna=True)
        spatial = [d for d in get_spatial_dims(derived) if d in derived.dims]
        ts_monthly = derived.sum(dim=spatial, skipna=True)
    else:
        # Mean on depth (unweighted), area-weighted mean on space.
        if "depth" in derived.dims:
            derived = derived.mean(dim="depth", skipna=True)
        ts_monthly = area_weighted_mean(derived, area)

    # 5. Annual mean → linear trend → per decade.
    ts_annual = ts_monthly.resample(time="YE").mean(skipna=True)
    if ts_annual.sizes.get("time", 0) < 3:
        return {"type": "scalar", "value": float("nan"),
                "note": "insufficient years for trend"}

    slope_per_year = float(compute_linear_trend(ts_annual).values)
    return {
        "type": "scalar",
        "value": slope_per_year * 10.0,
        "unit": "per decade",
        "qualified_area_variable_used": chosen_qa,
    }


def _solve_L3_2(data_meta, task_meta):
    """Has {implicit_derived_variable} anomaly in {region} at {term_for_depth_layer}
    depth changed in {verb} direction at a significant rate?"""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    dc, spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    term = task_meta["term_for_depth_layer"]

    BASELINE_START, BASELINE_END = "1995-01", "2014-12"
    ALPHA = 0.1

    # 1. Per-cell per-time layer depth field over the analysis period.
    base_codes_layer = ["thetao", "so"] if term == "mixed layer" else ["thetao"]
    ts_bases_layer = load_bases(
        sid, base_codes_layer, start, end, root, region_spec=region_spec,
    )
    if term == "mixed layer":
        layer_depth = compute_mixed_layer_depth(
            ts_bases_layer["thetao"], ts_bases_layer["so"],
        )
    elif term == "thermocline":
        layer_depth = compute_thermocline_depth(ts_bases_layer["thetao"])
    else:
        raise ValueError(f"Unknown depth-layer term: '{term}'")

    # 2. Reduce (time, lat, lon) → scalar: area-weighted spatial mean, then time mean.
    area = subset_region(load_area_for_model(eid, sid, root), region_spec)
    mean_layer_depth = float(
        area_weighted_mean(layer_depth, area).mean(dim="time", skipna=True).values
    )
    if not np.isfinite(mean_layer_depth) or mean_layer_depth <= 0:
        return {"type": "boolean", "value": None,
                "note": "could not estimate layer depth"}

    # 3. Derived series at mean_layer_depth — baseline (for climatology) + analysis (for trend).
    bases_baseline = load_bases(
        sid, spec["base_codes"], BASELINE_START, BASELINE_END, root,
        region_spec=region_spec,
    )
    bases_analysis = load_bases(
        sid, spec["base_codes"], start, end, root, region_spec=region_spec,
    )
    derived_baseline = compute_derived_at_depth(bases_baseline, spec, dc, mean_layer_depth)
    derived_analysis = compute_derived_at_depth(bases_analysis, spec, dc, mean_layer_depth)

    # 4. Anomaly vs baseline climatology (TERM_4: subtract the matching
    #    calendar-month mean to remove the seasonal cycle).
    climatology = compute_monthly_climatology(derived_baseline)
    anom = compute_anomaly(derived_analysis, climatology)

    # 5. Area-weighted regional ts → trend significance at alpha = 0.1.
    ts = area_weighted_mean(anom, area)
    sig = compute_trend_significance(ts, alpha=ALPHA)

    is_sig = bool(sig["is_significant"].values)
    slope = float(sig["slope"].values)
    direction_positive = (task_meta["verb_change_direction"] == "increased")
    answer = is_sig and ((slope > 0) == direction_positive)

    return {
        "type": "boolean",
        "value": answer,
        "is_significant": is_sig,
        "alpha": ALPHA,
        "slope": slope,
        "slope_unit": "per month",
        "p_value": float(sig["p_value"].values),
        "mean_layer_depth": mean_layer_depth,
        "mean_layer_depth_unit": "m",
        "baseline_period": f"{BASELINE_START} ~ {BASELINE_END}",
    }


def _solve_L3_3(data_meta, task_meta):
    """Pearson r between {implicit_derived_variable} anomaly and
    {variable} anomaly in {region} at {implicit_depth}."""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    dc, spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    implicit_depth = task_meta["implicit_depth"]

    BASELINE_START, BASELINE_END = "1995-01", "2014-12"

    # 1. Derived: bases for analysis + baseline.
    bases_analysis = load_bases_for_derived(
        sid, spec, start, end, root,
        region_spec=region_spec, implicit_depth=implicit_depth,
    )
    bases_baseline = load_bases_for_derived(
        sid, spec, BASELINE_START, BASELINE_END, root,
        region_spec=region_spec, implicit_depth=implicit_depth,
    )
    if bases_analysis is None or bases_baseline is None:
        return {"type": "scalar", "value": float("nan"),
                "note": "incompatible: sea floor with inventory-type derived"}
    derived_analysis = compute_derived_from_bases(bases_analysis, dc, spec)
    derived_baseline = compute_derived_from_bases(bases_baseline, dc, spec)

    # 2. Direct variable: analysis + baseline.
    def _load_var(s, e):
        v = load_variable_auto_experiment(
            sid, task_meta["variable_code"], s, e, root,
        )
        v = subset_region(v, region_spec)
        if "depth" in v.dims:
            v = apply_implicit_depth(v, implicit_depth)
        return v

    var_analysis = _load_var(start, end)
    var_baseline = _load_var(BASELINE_START, BASELINE_END)

    # 3. Anomaly w.r.t. baseline climatology → area-weighted regional ts.
    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    def _anom_ts(da_analysis, da_baseline):
        # TERM_4: anomaly = value − calendar-month climatology (deseasonalized).
        anom = compute_anomaly(da_analysis, compute_monthly_climatology(da_baseline))
        if "depth" in anom.dims:
            anom = anom.mean(dim="depth", skipna=True)
        return area_weighted_mean(anom, area)

    ts_d = _anom_ts(derived_analysis, derived_baseline)
    ts_v = _anom_ts(var_analysis, var_baseline)

    corr = compute_pearson_correlation(ts_d, ts_v)
    return {
        "type": "scalar",
        "value": corr["r"],
        "p_value": corr["p_value"],
        "baseline_period": f"{BASELINE_START} ~ {BASELINE_END}",
    }


def _solve_L3_4(data_meta, task_meta):
    """Is the decadal trend in {depth_integrated_inventory} significantly correlated at zero lag with the decadal trend of {implicit_derived_variable} when both are restricted to {enso_phase} months over {time_period} and  within the {implicit_region} in the {implicit_depth}?"""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["implicit_region_code"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    implicit_depth = task_meta["implicit_depth"]

    ALPHA = 0.1

    # 1. ENSO phase labels with proper baseline + persistence filtering.
    phase_labels = build_enso_phase_labels(sid, start, end, root)

    # 2. Regional cell area — used for cumulative per-cell totals AND area-weighted means.
    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    # 3. Inventory: integrate base over 0 ~ mean_layer.
    inv_dc, inv_spec = resolve_derived_variable(task_meta["depth_integrated_inventory"])
    inv_bases = load_bases_for_derived(
        sid, inv_spec, start, end, root,
        region_spec=region_spec, implicit_depth=implicit_depth,
    )
    if inv_bases is None:
        return {"type": "boolean", "value": None,
                "note": "incompatible: sea floor + inventory"}
    inventory = compute_derived_from_bases(inv_bases, inv_dc, inv_spec, area=area)

    # 4. Implicit derived variable.
    dv_dc, dv_spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    dv_bases = load_bases_for_derived(
        sid, dv_spec, start, end, root,
        region_spec=region_spec, implicit_depth=implicit_depth,
    )
    if dv_bases is None:
        return {"type": "boolean", "value": None,
                "note": "incompatible: sea floor + cumulative derived"}
    derived = compute_derived_from_bases(dv_bases, dv_dc, dv_spec, area=area)

    # 5. Area-weighted regional ts (intensive avg; depth simple-mean if still present).
    def _ts(da, spec):
        if is_cumulative(spec):
            if "depth" in da.dims:
                dz = get_dz(da, depth_dim="depth")
                da = (da * dz).sum(dim="depth", skipna=True)
            spatial = [d for d in get_spatial_dims(da) if d in da.dims]
            return da.sum(dim=spatial, skipna=True)
        if "depth" in da.dims:
            da = da.mean(dim="depth", skipna=True)
        return area_weighted_mean(da, area)

    ts_inv = _ts(inventory, inv_spec)
    ts_dv = _ts(derived, dv_spec)

    # 5. Restrict to ENSO phase months → resample to annual.
    ts_inv_sel = select_enso_phase(ts_inv, phase_labels, task_meta["enso_phase"])
    ts_dv_sel  = select_enso_phase(ts_dv, phase_labels, task_meta["enso_phase"])
    if ts_inv_sel.sizes.get("time", 0) == 0 or ts_dv_sel.sizes.get("time", 0) == 0:
        return {"type": "boolean", "value": None,
                "note": f"no months matching '{task_meta['enso_phase']}' in window"}
    ts_inv_annual = ts_inv_sel.resample(time="YE").mean(skipna=True)
    ts_dv_annual  = ts_dv_sel.resample(time="YE").mean(skipna=True)

    if ts_inv_annual.sizes.get("time", 0) < 3:
        return {"type": "boolean", "value": None,
                "note": f"too few annual samples for {task_meta['enso_phase']}"}

    # 6. Pearson correlation at zero lag → significance at alpha = 0.1.
    corr = compute_pearson_correlation(ts_inv_annual, ts_dv_annual)
    p = corr["p_value"]
    is_sig = bool(np.isfinite(p) and p < ALPHA)
    
    return {
        "type": "boolean",
        "value": is_sig,
        "alpha": ALPHA,
        "r": corr["r"],
        "p_value": p,
    }


def _solve_L3_5(data_meta, task_meta):
    """Has annual-mean global {implicit_derived_variable} anomaly in
    {depth_range} changed in {verb} direction significantly?"""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    dc, spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    dr = (task_meta["depth_lower"], task_meta["depth_upper"])

    BASELINE_START, BASELINE_END = "1995-01", "2014-12"
    ALPHA = 0.1

    # 1. Global cell area — for cumulative per-cell totals AND area-weighted means.
    area = load_area_for_model(eid, sid, root)

    # 2. Derived for analysis + baseline (global, depth-band subsetted).
    bases_analysis = load_bases(
        sid, spec["base_codes"], start, end, root, depth_range=dr,
    )
    bases_baseline = load_bases(
        sid, spec["base_codes"], BASELINE_START, BASELINE_END, root, depth_range=dr,
    )
    derived_analysis = compute_derived_from_bases(bases_analysis, dc, spec, area=area)
    derived_baseline = compute_derived_from_bases(bases_baseline, dc, spec, area=area)

    # 3. Anomaly w.r.t. baseline climatology (TERM_4: calendar-month means).
    anom = compute_anomaly(derived_analysis, compute_monthly_climatology(derived_baseline))

    # 4. Reduce depth + space (cumulative → sum; otherwise → area-weighted mean).
    if is_cumulative(spec):
        if "depth" in anom.dims:
            dz = get_dz(anom, depth_dim="depth")
            anom = (anom * dz).sum(dim="depth", skipna=True)
        spatial = [d for d in get_spatial_dims(anom) if d in anom.dims]
        ts_monthly = anom.sum(dim=spatial, skipna=True)
    else:
        if "depth" in anom.dims:
            anom = anom.mean(dim="depth", skipna=True)
        ts_monthly = area_weighted_mean(anom, area)

    ts_annual = ts_monthly.resample(time="YE").mean(skipna=True)
    if ts_annual.sizes.get("time", 0) < 3:
        return {"type": "boolean", "value": None,
                "note": "insufficient years for trend"}

    # 5. Trend significance at alpha = 0.1.
    sig = compute_trend_significance(ts_annual, alpha=ALPHA)
    is_sig = bool(sig["is_significant"].values)
    slope = float(sig["slope"].values)
    positive = (task_meta["verb_change_direction"] == "increased")
    answer = is_sig and ((slope > 0) == positive)
    return {
        "type": "boolean", "value": answer,
        "is_significant": is_sig, "alpha": ALPHA, "slope": slope, "slope_unit": "per year",
        "p_value": float(sig["p_value"].values), "baseline_period": f"{BASELINE_START} ~ {BASELINE_END}",
    }


def _solve_L3_6(data_meta, task_meta):
    """Difference in {stat} of {implicit_derived_variable} between
    {qualified_years} and non-{qualified_years} months in {region}."""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    dc, spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    start, end = task_meta["start_month"], task_meta["end_month"]

    BASELINE_START, BASELINE_END = "1995-01", "2014-12"

    # 1. Derived in region + implicit_depth (cumulative → per-cell totals via area).
    bases = load_bases_for_derived(
        sid, spec, start, end, root,
        region_spec=region_spec, implicit_depth=task_meta["implicit_depth"],
    )
    if bases is None:
        return {"type": "scalar", "value": float("nan"),
                "note": "incompatible: sea floor with inventory-type derived"}
    area = subset_region(load_area_for_model(eid, sid, root), region_spec)
    derived = compute_derived_from_bases(bases, dc, spec, area=area)

    # 2. Qualified-time mask, baseline-relative (1995–2014).
    sst_a = load_variable_auto_experiment(sid, "thetao", start, end, root)
    sst_b = load_variable_auto_experiment(
        sid, "thetao", BASELINE_START, BASELINE_END, root,
    )
    def _surf(da):
        if "depth" not in da.dims:
            return da
        out = da.isel(depth=0)
        # isel keeps `depth` (and any depth-attached coord like `dz`) as scalar
        # non-dim coords; they later broadcast to (time,) and clash with
        # `derived`'s (depth: 1) dim coord during alignment.
        drop = [c for c in ("depth", "dz") if c in out.coords]
        return out.drop_vars(drop) if drop else out

    sst_a_surf = _surf(sst_a)
    sst_b_surf = _surf(sst_b)
    area_full = load_area_for_model(eid, sid, root)
    qual_mask = classify_qualified_time(
        task_meta["qualified_time"],
        sid, start, end, root,
        sst_analysis=sst_a_surf,
        sst_baseline=sst_b_surf,
        area=area_full,
    )

    # 3. Align time and split.
    derived = derived.sel(time=qual_mask["time"])
    qual_aligned = qual_mask.sel(time=derived["time"])
    derived_q = derived.where(qual_aligned, drop=True)
    derived_nq = derived.where(~qual_aligned, drop=True)
    if derived_q.sizes.get("time", 0) == 0 or derived_nq.sizes.get("time", 0) == 0:
        return {"type": "scalar", "value": float("nan"),
                "note": f"empty composite for '{task_meta['qualified_time']}'"}

    # 4. Reduce to scalar (cumulative → sum; non-cumulative → area-weighted/unweighted stat).
    stat_code = task_meta["statistical_operator_code"]
    cumulative = is_cumulative(spec)

    def _reduce(da):
        if "depth" in da.dims and cumulative:
                dz = get_dz(da, depth_dim="depth")
                da = (da * dz).sum(dim="depth", skipna=True)
        if cumulative:
            spatial = [d for d in get_spatial_dims(da) if d in da.dims]
            ts = da.sum(dim=spatial, skipna=True)             # 1-D in time
            return float(compute_basic_statistic(ts, stat_code).values)
        return float(
            compute_basic_statistic_with_area(da, stat_code, area=area).values
        )

    val_q = _reduce(derived_q)
    val_nq = _reduce(derived_nq)
    return {
        "type": "scalar",
        "value": val_q - val_nq,                              # qualified − non-qualified
        "qualified_value": val_q,
        "non_qualified_value": val_nq,
        "n_qualified_months": int(derived_q.sizes["time"]),
        "n_non_qualified_months": int(derived_nq.sizes["time"]),
        "baseline_period_for_SSTAnoma": f"{BASELINE_START} ~ {BASELINE_END}",
    }


def _solve_L3_7(data_meta, task_meta):
    """Ratio of mean {derived_variable_1} to mean {derived_variable_2}."""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    it_start, it_end = resolve_implicit_time_period(task_meta["implicit_time_period_code"])

    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    def _mean_derived(nl):
        dc, spec = resolve_derived_variable(nl)
        bases = load_bases_for_derived(
            sid, spec, it_start, it_end, root,
            region_spec=region_spec,
            implicit_depth=task_meta["implicit_depth"],
        )
        if bases is None:
            return float("nan")
        derived = compute_derived_from_bases(bases, dc, spec, area=area)

        if is_cumulative(spec):
            if "depth" in derived.dims:
                dz = get_dz(derived, depth_dim="depth")
                derived = (derived * dz).sum(dim="depth", skipna=True)
            spatial = [d for d in get_spatial_dims(derived) if d in derived.dims]
            ts = derived.sum(dim=spatial, skipna=True)
        else:
            if "depth" in derived.dims:
                derived = derived.mean(dim="depth", skipna=True)
            ts = area_weighted_mean(derived, area)

        return float(ts.mean(dim="time", skipna=True).values)

    m1 = _mean_derived(task_meta["derived_variable_1"])
    m2 = _mean_derived(task_meta["derived_variable_2"])
    if not (np.isfinite(m1) and np.isfinite(m2)) or m2 == 0:
        return {"type": "scalar", "value": float("nan"),
                "mean_1": m1, "mean_2": m2}
    return {"type": "scalar", "value": m1 / m2,
            "mean_1": m1, "mean_2": m2}


def _solve_L3_8(data_meta, task_meta):
    """Change in mean {depth_integrated_inventory} within {qualified_area}
    of {implicit_region} between two periods."""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    dc, spec = resolve_derived_variable(task_meta["depth_integrated_inventory"])
    region_spec = get_region_spec(task_meta["implicit_region_code"])
    qa_criteria = resolve_qualified_area_criteria(task_meta["qualified_area"])

    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    def _period_mean(start, end):
        # Inventory over full water column → per-cell totals (mol/cell) via area.
        bases = load_bases(
            sid, spec["base_codes"], start, end, root, region_spec=region_spec,
        )
        inv = compute_derived_from_bases(bases, dc, spec, area=area)

         # Qualified-area mask (with fallback chain inside).
        mask, _ = build_qualified_area_mask(
            sid, qa_criteria, start, end, root, region_spec=region_spec,
        )
        if mask is None:
            return float("nan")
        
        # Cumulative reduction: mask → (defensive) depth-sum → spatial sum → time mean.
        inv = inv.where(mask)
        if "depth" in inv.dims:
            dz = get_dz(inv, depth_dim="depth")
            inv = (inv * dz).sum(dim="depth", skipna=True)
        spatial = [d for d in get_spatial_dims(inv) if d in inv.dims]
        ts = inv.sum(dim=spatial, skipna=True)               # 1-D in time
        return float(ts.mean(dim="time", skipna=True).values)

    v1 = _period_mean(task_meta["start_month_1"], task_meta["end_month_1"])
    v2 = _period_mean(task_meta["start_month_2"], task_meta["end_month_2"])
    if not (np.isfinite(v1) and np.isfinite(v2)):
        return {"type": "scalar", "value": float("nan"),
                "period_1_value": v1, "period_2_value": v2}
    return {"type": "scalar", "value": v2 - v1,
            "period_1_value": v1, "period_2_value": v2}


def _solve_L3_9(data_meta, task_meta):
    """Change in mean volume fraction classified as {descriptive_noun}."""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    noun = task_meta["descriptive_noun"]
    criteria = resolve_descriptive_noun(noun)
    implicit_depth = task_meta["implicit_surface_depth"]

    # Variables to attempt to load (top-level + derived's bases if any).
    needed_vars = set(criteria["variables"])
    if "derived" in criteria:
        _, dspec = resolve_derived_variable(criteria["derived"])
        needed_vars.update(dspec["base_codes"])

    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    def _frac_for(start, end):
        # Per-variable lazy load (tolerate missing for fallback-chain criteria).
        bases = {}
        for var in needed_vars:
            try:
                v = load_variable_auto_experiment(sid, var, start, end, root)
            except FileNotFoundError:
                continue
            v = subset_region(v, region_spec)
            if "depth" in v.dims:
                v = apply_implicit_depth(v, implicit_depth)
            bases[var] = v
        if not bases:
            return float("nan")

        # Per-month mask (no climatology pre-reduction).
        mask = build_descriptive_noun_mask(bases, noun)
        if mask is None:
            return float("nan")

        ref = next(iter(bases.values()))
        valid = ref.notnull()

        vf_ts = compute_volume_fraction(mask, area, valid=valid)   # 1-D in time
        return float(vf_ts.mean(dim="time", skipna=True).values)

    f1 = _frac_for(task_meta["start_month_1"], task_meta["end_month_1"])
    f2 = _frac_for(task_meta["start_month_2"], task_meta["end_month_2"])
    if not (np.isfinite(f1) and np.isfinite(f2)):
        return {"type": "scalar", "value": float("nan"),
                "period_1_fraction": f1, "period_2_fraction": f2}
    return {"type": "scalar", "value": f1 - f2,
            "period_1_fraction": f1, "period_2_fraction": f2,
            "value_unit": "dimensionless"}


def _solve_L3_10(data_meta, task_meta):
    """Inter-model spread in {implicit_depth} {variable} climatology over {region}.

    Each model's regional climatology is collapsed to a single area-weighted
    scalar, which is grid-independent — i.e. the equivalent of regridding
    to a common 1° grid before regional averaging.
    """
    source_id = data_meta["source_id"]
    root = data_meta["data_root"]
    var = task_meta["variable_code"]
    region_spec = get_region_spec(task_meta["region_code"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    implicit_depth = task_meta["implicit_depth"]

    THRESHOLD = 0.5
    RES = 0.1

    # Common 0.1° target grid (handles dateline-crossing regions).
    lat_min = float(region_spec["lat_min"])
    lat_max = float(region_spec["lat_max"])
    lon_min = float(region_spec["lon_min"]) % 360
    lon_max = float(region_spec["lon_max"]) % 360
    target_lat = np.arange(lat_min, lat_max + RES / 2, RES)
    if lon_min <= lon_max:
        target_lon = np.arange(lon_min, lon_max + RES / 2, RES)
    else:
        target_lon = np.concatenate([
            np.arange(lon_min, 360, RES),
            np.arange(0, lon_max + RES / 2, RES),
        ])

    model_clims = []
    for sid in source_id:
        try:
            da = load_variable_auto_experiment(sid, var, start, end, root)
            da = subset_region(da, region_spec)
            if "depth" in da.dims:
                da = apply_implicit_depth(da, implicit_depth)
            clim = compute_climatology(da)
            if "depth" in clim.dims:
                clim = clim.mean(dim="depth", skipna=True)
            clim = regrid_to_grid(clim, target_lat, target_lon, method="linear")
            model_clims.append(clim)
        except (FileNotFoundError, KeyError, ValueError):
            continue

    if len(model_clims) < 3:
        return {"type": "boolean", "value": None,
                "n_models": len(model_clims),
                "note": f"only {len(model_clims)} models succeeded; need >= 3"}

    # Stack along 'model' dim → per-cell normalized spread.
    stacked = xr.concat(model_clims, dim="model")
    mean_per_cell = stacked.mean(dim="model", skipna=True)
    std_per_cell = stacked.std(dim="model", skipna=True, ddof=1)
    cv_per_cell = xr.where(
        np.abs(mean_per_cell) > 1e-12,
        std_per_cell / np.abs(mean_per_cell),
        np.nan,
    )

    # cos(lat)-weighted spatial mean of the cv field.
    cos_lat = np.cos(np.deg2rad(cv_per_cell["lat"]))
    cv_mean = float(
        cv_per_cell.weighted(cos_lat.fillna(0)).mean(skipna=True).values
    )
    is_sig = bool(np.isfinite(cv_mean) and cv_mean > THRESHOLD)

    return {
        "type": "boolean",
        "value": is_sig,
        "n_models": len(model_clims),
        "mean_normalized_spread": cv_mean,
        "threshold": THRESHOLD,
    }


def _solve_L3_11(data_meta, task_meta):
    """Inter-model std of {implicit_depth} {variable} anomaly
    (recent 1995-2014 vs pre-industrial 1850-1900) over {region}."""
    source_id = data_meta["source_id"]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    var = task_meta["variable_code"]
    region_spec = get_region_spec(task_meta["region_code"])
    implicit_depth = task_meta["implicit_depth"]

    RECENT_START, RECENT_END = "1995-01", "2014-12"
    PI_START, PI_END = "1850-01", "1900-12"
    RES = 0.1

    # Common 0.1° target grid (handles dateline-crossing regions).
    lat_min = float(region_spec["lat_min"])
    lat_max = float(region_spec["lat_max"])
    lon_min = float(region_spec["lon_min"]) % 360
    lon_max = float(region_spec["lon_max"]) % 360
    target_lat = np.arange(lat_min, lat_max + RES / 2, RES)
    if lon_min <= lon_max:
        target_lon = np.arange(lon_min, lon_max + RES / 2, RES)
    else:
        target_lon = np.concatenate([
            np.arange(lon_min, 360, RES),
            np.arange(0, lon_max + RES / 2, RES),
        ])

    def _period_clim_2d(sid, start, end):
        """Per-model 2-D climatology on the common 0.1° grid."""
        da = load_variable_auto_experiment(sid, var, start, end, root)
        da = subset_region(da, region_spec)
        if "depth" in da.dims:
            da = apply_implicit_depth(da, implicit_depth)
        clim = compute_climatology(da)                       # time mean
        if "depth" in clim.dims:
            clim = clim.mean(dim="depth", skipna=True)        # depth mean
        return regrid_to_grid(clim, target_lat, target_lon, method="linear")

    def _cos_weighted_mean(da):
        cos_lat = np.cos(np.deg2rad(da["lat"]))
        return float(da.weighted(cos_lat.fillna(0)).mean(skipna=True).values)

    model_anomalies = []
    for sid in source_id:
        try:
            recent = _period_clim_2d(sid, RECENT_START, RECENT_END)
            pi = _period_clim_2d(sid, PI_START, PI_END)
            model_anomalies.append(recent - pi)               # per-grid anomaly
        except (FileNotFoundError, KeyError, ValueError):
            continue

    if len(model_anomalies) < 3:
        return {"type": "scalar", "value": float("nan"),
                "n_models": len(model_anomalies),
                "note": f"only {len(model_anomalies)} models; need >= 3"}

    # Stack along 'model' dim → per-cell std → cos(lat)-weighted spatial mean.
    stacked = xr.concat(model_anomalies, dim="model")
    std_per_cell = stacked.std(dim="model", skipna=True, ddof=1)
    std_mean = _cos_weighted_mean(std_per_cell)

    # Diagnostic: per-model regional mean anomaly.
    per_model_means = [_cos_weighted_mean(a) for a in model_anomalies]

    return {
        "type": "scalar",
        "value": std_mean,
        "n_models": len(model_anomalies),
        "ensemble_mean_anomaly": float(np.mean(per_model_means)),
        "recent_period": f"{RECENT_START} ~ {RECENT_END}",
        "preindustrial_period": f"{PI_START} ~ {PI_END}",
    }


def _solve_L3_12(data_meta, task_meta):
    """Has the {term_for_depth_layer} depth in {region} shifted deeper at a
    statistically significant rate over {time_period}?"""
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    term = task_meta["term_for_depth_layer"]

    ALPHA = 0.1

    base_codes = ["thetao", "so"] if term == "mixed layer" else ["thetao"]
    ts_bases = load_bases(
        sid, base_codes, start, end, root, region_spec=region_spec,
    )
    if term == "mixed layer":
        depth_field = compute_mixed_layer_depth(
            ts_bases["thetao"], ts_bases["so"],
        )
    elif term == "thermocline":
        depth_field = compute_thermocline_depth(ts_bases["thetao"])
    else:
        raise ValueError(f"Unknown depth-layer term: '{term}'")

    area = subset_region(load_area_for_model(eid, sid, root), region_spec)
    ts = area_weighted_mean(depth_field, area)
    sig = compute_trend_significance(ts, alpha=ALPHA)

    is_sig = bool(sig["is_significant"].values)
    slope = float(sig["slope"].values)
    answer = is_sig and (slope > 0)                   # "deeper" = depth increasing
    return {
        "type": "boolean",
        "value": answer,
        "is_significant": is_sig,
        "alpha": ALPHA,
        "slope": slope,
        "slope_unit": "m per month",
        "p_value": float(sig["p_value"].values),
    }


def _solve_L3_13(data_meta, task_meta):
    """Multi-model ensemble mean change in {implicit_derived_variable} in
    {implicit_region} in {implicit_depth} between time_period_1 and
    time_period_2.

    Change = period_2 − period_1.

    Cumulative branch: per-model regional total time-mean → ensemble mean
    of period scalars → difference.

    Non-cumulative branch: per-model time-mean climatology → depth mean →
    bilinear regrid to 0.1° → per-grid ensemble mean per period →
    cos(lat)-weighted spatial mean per period → difference.
    """
    source_id = data_meta["source_id"]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    dc, spec = resolve_derived_variable(task_meta["implicit_derived_variable"])
    region_spec = get_region_spec(task_meta["implicit_region_code"])
    s1, e1 = task_meta["start_month_1"], task_meta["end_month_1"]
    s2, e2 = task_meta["start_month_2"], task_meta["end_month_2"]
    implicit_depth = task_meta["implicit_depth"]

    cumulative = is_cumulative(spec)

    # =================== Cumulative branch ===================
    if cumulative:
        def _period_scalar(sid, start, end):
            area = subset_region(load_area_for_model(eid, sid, root), region_spec)
            bases = load_bases_for_derived(
                sid, spec, start, end, root,
                region_spec=region_spec, implicit_depth=implicit_depth,
            )
            if bases is None:
                return float("nan")
            d = compute_derived_from_bases(bases, dc, spec, area=area)
            if "depth" in d.dims:
                dz = get_dz(d, depth_dim="depth")
                d = (d * dz).sum(dim="depth", skipna=True)
            spatial = [dim for dim in get_spatial_dims(d) if dim in d.dims]
            ts = d.sum(dim=spatial, skipna=True)
            return float(ts.mean(dim="time", skipna=True).values)

        v1, v2 = [], []
        for sid in source_id:
            try:
                a = _period_scalar(sid, s1, e1)
                b = _period_scalar(sid, s2, e2)
                if np.isfinite(a) and np.isfinite(b):
                    v1.append(a)
                    v2.append(b)
            except (FileNotFoundError, KeyError, ValueError):
                continue

        if not v1:
            return {"type": "scalar", "value": float("nan"),
                    "note": "no models succeeded"}

        v1_ens = float(np.mean(v1))
        v2_ens = float(np.mean(v2))
        per_model_changes = [b - a for a, b in zip(v1, v2)]
        return {
            "type": "scalar",
            "value": v2_ens - v1_ens,
            "n_models": len(v1),
            "period_1_ensemble_value": v1_ens,
            "period_2_ensemble_value": v2_ens,
            "model_changes": per_model_changes,
            "ensemble_std": (
                float(np.std(per_model_changes, ddof=1))
                if len(per_model_changes) > 1 else 0.0
            ),
        }

    # =================== Non-cumulative branch ===================
    RES = 0.1
    lat_min = float(region_spec["lat_min"])
    lat_max = float(region_spec["lat_max"])
    lon_min = float(region_spec["lon_min"]) % 360
    lon_max = float(region_spec["lon_max"]) % 360
    target_lat = np.arange(lat_min, lat_max + RES / 2, RES)
    if lon_min <= lon_max:
        target_lon = np.arange(lon_min, lon_max + RES / 2, RES)
    else:
        target_lon = np.concatenate([
            np.arange(lon_min, 360, RES),
            np.arange(0, lon_max + RES / 2, RES),
        ])

    def _period_clim_2d(sid, start, end):
        bases = load_bases_for_derived(
            sid, spec, start, end, root,
            region_spec=region_spec, implicit_depth=implicit_depth,
        )
        if bases is None:
            return None
        d = compute_derived_from_bases(bases, dc, spec)         # no area
        clim = compute_climatology(d)                            # time mean
        if "depth" in clim.dims:
            clim = clim.mean(dim="depth", skipna=True)           # depth mean
        return regrid_to_grid(clim, target_lat, target_lon, method="linear")

    def _cos_weighted_mean(da):
        cos_lat = np.cos(np.deg2rad(da["lat"]))
        return float(da.weighted(cos_lat.fillna(0)).mean(skipna=True).values)

    clims_p1, clims_p2 = [], []
    for sid in source_id:
        try:
            c1 = _period_clim_2d(sid, s1, e1)
            c2 = _period_clim_2d(sid, s2, e2)
            if c1 is not None and c2 is not None:
                clims_p1.append(c1)
                clims_p2.append(c2)
        except (FileNotFoundError, KeyError, ValueError):
            continue

    if not clims_p1:
        return {"type": "scalar", "value": float("nan"),
                "note": "no models succeeded"}

    # Per-grid ensemble mean per period.
    ens_p1 = xr.concat(clims_p1, dim="model").mean(dim="model", skipna=True)
    ens_p2 = xr.concat(clims_p2, dim="model").mean(dim="model", skipna=True)
    v1_ens = _cos_weighted_mean(ens_p1)
    v2_ens = _cos_weighted_mean(ens_p2)

    # Per-model regional change scalars (for ensemble_std diagnostic).
    per_model_changes = [
        _cos_weighted_mean(c2 - c1) for c1, c2 in zip(clims_p1, clims_p2)
    ]

    return {
        "type": "scalar",
        "value": v2_ens - v1_ens,
        "n_models": len(clims_p1),
        "period_1_ensemble_value": v1_ens,
        "period_2_ensemble_value": v2_ens,
        "model_changes": per_model_changes,
        "ensemble_std": (
            float(np.std(per_model_changes, ddof=1))
            if len(per_model_changes) > 1 else 0.0
        ),
    }


def _solve_L3_14(data_meta, task_meta):
    """Lag in months (1..12) at which the correlation between two derived
    variables, both restricted to {region} and {depth_range}, is maximized
    over {time_period}.

    Per variable:
      - Cumulative: dz-weighted depth sum + area-weighted spatial sum
        → regional total time series.
      - Non-cumulative: depth mean + area-weighted spatial mean
        → regional mean time series.
    Each series is deseasonalized (subtract analysis-period monthly
    climatology) so the lag is not dominated by the seasonal cycle.

    Lag correlation: r(k) = pearson(var1[t], var2[t+k]) for k = 1..12.
    Convention: positive k means var1 leads var2 by k months.
    Returned best_lag = argmax_k |r(k)| (robust to sign of expected
    correlation; pairs like hke vs aou are intrinsically anti-correlated).
    """
    sid = data_meta["source_id"][0]
    eid = data_meta["experiment_id"]
    root = data_meta["data_root"]
    region_spec = get_region_spec(task_meta["region_code"])
    start, end = task_meta["start_month"], task_meta["end_month"]
    depth_range = (task_meta["depth_lower"], task_meta["depth_upper"])

    dc1, spec1 = resolve_derived_variable(
        task_meta["correlated_derived_variable_1"],
    )
    dc2, spec2 = resolve_derived_variable(
        task_meta["correlated_derived_variable_2"],
    )

    area = subset_region(load_area_for_model(eid, sid, root), region_spec)

    def _regional_ts(dc, spec):
        bases = load_bases_for_derived(
            sid, spec, start, end, root,
            region_spec=region_spec, depth_range=depth_range,
        )
        if bases is None:
            return None
        if is_cumulative(spec):
            d = compute_derived_from_bases(bases, dc, spec, area=area)
            if "depth" in d.dims:
                dz = get_dz(d, depth_dim="depth")
                d = (d * dz).sum(dim="depth", skipna=True)
            spatial = [dim for dim in get_spatial_dims(d) if dim in d.dims]
            ts = d.sum(dim=spatial, skipna=True)
        else:
            d = compute_derived_from_bases(bases, dc, spec)
            if "depth" in d.dims:
                d = d.mean(dim="depth", skipna=True)
            ts = area_weighted_mean(d, area)
        # Deseasonalize: subtract analysis-period monthly climatology.
        clim = ts.groupby("time.month").mean(dim="time", skipna=True)
        return (ts.groupby("time.month") - clim).values

    a1 = _regional_ts(dc1, spec1)
    a2 = _regional_ts(dc2, spec2)
    if a1 is None or a2 is None:
        return {"type": "scalar", "value": float("nan"),
                "note": "incompatible variable / sample"}

    n = min(len(a1), len(a2))
    a1, a2 = a1[:n], a2[:n]

    LAGS = list(range(0, 13)) # 0-12
    rs = {}
    for k in LAGS:
        x = a1[: n - k]
        y = a2[k:]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 24:                       # need ≥2 yr after pairing
            rs[k] = float("nan")
            continue
        xv, yv = x[valid], y[valid]
        if np.std(xv) == 0 or np.std(yv) == 0:
            rs[k] = float("nan")
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            rs[k] = float(np.corrcoef(xv, yv)[0, 1])

    finite_rs = {k: r for k, r in rs.items() if np.isfinite(r)}
    if not finite_rs:
        return {"type": "scalar", "value": float("nan"),
                "note": "no valid lags"}

    # argmax of |r| — pairs in this task have mixed signs of expected
    # correlation, so |r| is the appropriate "strength" metric.
    best_lag = max(finite_rs, key=lambda k: abs(finite_rs[k]))
    best_r = finite_rs[best_lag]

    return {
        "type": "scalar",
        "value": int(best_lag),
        "unit": "months",
        "best_r": best_r,
        "lag_correlations": rs,
    }


# ===================================================================
# Main entry point
# ===================================================================

_SOLVERS = {
    "L3_1":  _solve_L3_1,   "L3_2":  _solve_L3_2,
    "L3_3":  _solve_L3_3,   "L3_4":  _solve_L3_4,
    "L3_5":  _solve_L3_5,   "L3_6":  _solve_L3_6,
    "L3_7":  _solve_L3_7,   "L3_8":  _solve_L3_8,
    "L3_9":  _solve_L3_9,   "L3_10": _solve_L3_10,
    "L3_11": _solve_L3_11,  "L3_12": _solve_L3_12,
    "L3_13": _solve_L3_13,  "L3_14": _solve_L3_14,
}

_MULTI_MODEL_TASKS = {"L3_10", "L3_11", "L3_13"}


def generate_level3_tasks(config, task_id):
    """
    Generate level 3 tasks based on the task ID.

    Returns:
        - task_id (str): The ID of the task.
        - data_metadata (dict): Describe the data requirements for the task.
        - task_metadata (dict): Describe the specific parameters for the task template.
        - answer_metadata (dict): Type of expected answer + computed answer + units.
    """
    experiment_dict = config.cmip6.experiment
    experiment_id = random.choice(list(experiment_dict.keys()))
    all_source_id = experiment_dict[experiment_id]
    source_id = random.choice(all_source_id)

    data_metadata = {
        "experiment_id": experiment_id,
        "source_id": list(all_source_id) if task_id in _MULTI_MODEL_TASKS
                      else [source_id],
        "data_root": config.cmip6.root,
    }


    if task_id == "L3_1":
        # What is the decadal linear trend in the {implicit_depth} {implicit_derived_variable} within the {qualified_area} inside the {region} over {time_period_yr}?
        implicit_depth = sample_implicit_depth()
        implicit_derived_dict = sample_implicit_derived_variable()
        region_code, region = sample_region()
        qualified_area, _ = sample_qualified_area()
        start_yr, end_yr, time_period_yr = sample_time_period_yr(experiment=experiment_id, min_years=20, max_years=50)

        task_metadata = {
            "implicit_depth": implicit_depth,
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "region_code": region_code,
            "region": region,
            "qualified_area": qualified_area,
            "start_year": start_yr,
            "end_year": end_yr,
            "time_period_yr": time_period_yr,
        }
        task_metadata.update({"ambiguous_terms": [
            "decadal linear trend", implicit_depth, implicit_derived_dict["derived_nl"], qualified_area
        ]})
        data_metadata.update({"variable_id": implicit_derived_dict["base_codes"], "start_time": start_yr, "end_time": end_yr})


    elif task_id == "L3_2":
        # Has the {implicit_derived_variable} anomaly in the {region} in the {term_for_depth_layer} depth {verb_change_direction} at a statistically significant rate over {time_period}?
        implicit_derived_dict = sample_implicit_derived_variable()
        region_code, region = sample_region()
        layer_var_codes, term_for_depth_layer = sample_term_for_depth_layer()
        verb_change_direction = sample_verb_change_direction()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=300)

        task_metadata = {
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "region_code": region_code,
            "region": region,
            "term_for_depth_layer": term_for_depth_layer,
            "verb_change_direction": verb_change_direction,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            implicit_derived_dict["derived_nl"], "anomaly", "average", term_for_depth_layer, "statistically significant"
        ]})
        # Solver also loads a 1995-2014 baseline; cover both windows.
        overall_start = min(start_month, "1995-01")
        overall_end   = max(end_month,   "2014-12")
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id": implicit_derived_dict["base_codes"] + layer_var_codes,
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L3_3":
        # What is the Pearson correlation coefficient between {implicit_derived_variable} anomaly and {variable} anomaly in the {region} in the {implicit_depth} over {time_period}?
        implicit_derived_dict = sample_implicit_derived_variable()
        var_code, var_nl = sample_variable_with_depth()
        region_code, region = sample_region()
        implicit_depth = sample_implicit_depth()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=36, max_months=300)

        task_metadata = {
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "variable_code": var_code,
            "variable": var_nl,
            "region_code": region_code,
            "region": region,
            "implicit_depth": implicit_depth,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            implicit_derived_dict["derived_nl"], "anomaly", implicit_depth, "Pearson correlation", "mean"
        ]})
        # Solver also loads a 1995-2014 baseline; cover both windows.
        overall_start = min(start_month, "1995-01")
        overall_end   = max(end_month,   "2014-12")
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id": list(set(
                implicit_derived_dict["base_codes"] + [var_code])),
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L3_4":
        # Is the decadal trend in {depth_integrated_inventory} significantly correlated at zero lag with the decadal trend of {implicit_derived_variable} when both are restricted to {enso_phase} months over {time_period} and  within the {implicit_region} in the {implicit_depth}?
        implicit_region_code, implicit_region = sample_implicit_region()
        implicit_depth = sample_implicit_depth()
        enso_phase = sample_enso_phase()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=36, max_months=300)

        while True:
            depth_integrated_inventory_dict = sample_depth_integrated_inventory()
            implicit_derived_dict = sample_implicit_derived_variable()
            if depth_integrated_inventory_dict["derived_nl"] != implicit_derived_dict["derived_nl"]:
                break

        task_metadata = {
            "depth_integrated_inventory": depth_integrated_inventory_dict["derived_nl"],
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "implicit_depth": implicit_depth,
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "enso_phase": enso_phase,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }

        task_metadata.update({"ambiguous_terms": [
            "annual trend", depth_integrated_inventory_dict["derived_nl"], "significantly", 
            implicit_derived_dict["derived_nl"], enso_phase, implicit_region, implicit_depth
        ]})

        # Solver builds ENSO phase labels via the fixed 1971-2000 baseline,
        # so cover [ENSO baseline start, analysis end] (and analysis start,
        # in case it's earlier than the baseline).
        metadata_variables = list(set(
            depth_integrated_inventory_dict["base_codes"]
            + implicit_derived_dict["base_codes"]
            + ["thetao"]))                   # SST needed for ENSO baseline
        overall_start = min(start_month, ENSO_BASELINE_START)
        overall_end   = max(end_month,   ENSO_BASELINE_END)
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id": metadata_variables,
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L3_5":
        # Has the annual-mean {implicit_derived_variable} anomaly in the global ocean between {depth_range} {verb_change_direction} significantly during {time_period}?
        implicit_derived_dict = sample_implicit_derived_variable()
        primary_var = implicit_derived_dict["base_codes"][0]
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, primary_var)
        if dep_bnds is None: return None
        lower, upper, depth_range = sample_depth_range(dep_bnds=dep_bnds)
        verb_change_direction = sample_verb_change_direction()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=300)

        task_metadata = {
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth_range,
            "verb_change_direction": verb_change_direction,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "annual-mean", implicit_derived_dict["derived_nl"], "anomaly", "significantly"
        ]})
        # Solver also loads a 1995-2014 baseline; cover both windows.
        overall_start = min(start_month, "1995-01")
        overall_end   = max(end_month,   "2014-12")
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id": implicit_derived_dict["base_codes"],
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L3_6":
        # What is the difference in the {statistical_operator} {implicit_derived_variable} within the {region} in the {implicit_depth} between {qualified_time} and non-{qualified_time} during {time_period}?
        stat_code, stat_op = sample_statistical_operator()
        implicit_derived_dict = sample_implicit_derived_variable()
        region_code, region = sample_region()
        implicit_depth = sample_implicit_depth()
        qualified_time = sample_qualified_time()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=24, max_months=300)

        task_metadata = {
            "statistical_operator_code": stat_code,
            "statistical_operator": stat_op,
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "region_code": region_code,
            "region": region,
            "implicit_depth": implicit_depth,
            "qualified_time": qualified_time,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "difference", implicit_derived_dict["derived_nl"], implicit_depth, qualified_time
        ]})
        # Solver loads a 1995-2014 SST baseline (for heatwave classification)
        # and ENSO/PDO classification uses the fixed 1971-2000 baseline; the
        # analysis window may sit anywhere. Span the union of all windows.
        overall_start = min(start_month, ENSO_BASELINE_START)        # 1971-01
        overall_end   = max(end_month,   "2014-12")
        data_metadata.update({
            "experiment_id": experiment_for_period(overall_start, overall_end),
            "variable_id": list(set(
                implicit_derived_dict["base_codes"] + ["thetao"])),
            "start_time": overall_start,
            "end_time":   overall_end,
        })


    elif task_id == "L3_7":
        # What is the ratio of the mean {derived_variable_1} to the mean {derived_variable_2} in the {implicit_depth} within the {region} during {implicit_time_period}?
        complex_ratio_dict = sample_complex_ratio()
        implicit_depth = sample_implicit_depth()
        region_code, region = sample_region()
        implicit_time_period_code, implicit_time_period = sample_implicit_time_period()

        task_metadata = {
            "derived_variable_1": complex_ratio_dict["var1_derived_nl"],
            "derived_variable_2": complex_ratio_dict["var2_derived_nl"],
            "implicit_depth": implicit_depth,
            "region_code": region_code,
            "region": region,
            "implicit_time_period_code": implicit_time_period_code,
            "implicit_time_period": implicit_time_period,
        }

        task_metadata.update({"ambiguous_terms": [
            "mean", complex_ratio_dict["var1_derived_nl"], complex_ratio_dict["var2_derived_nl"], 
            implicit_depth, implicit_time_period
        ]})

        metadata_variables = list(set(
            complex_ratio_dict["var1_base_codes"]
            + complex_ratio_dict["var2_base_codes"]))
        # Solver loads only the implicit time period; pin experiment_id and
        # start/end to that window (was previously left as random experiment_id
        # with no time window).
        it_start, it_end = resolve_implicit_time_period(implicit_time_period_code)
        data_metadata.update({
            "experiment_id": experiment_for_period(it_start, it_end),
            "variable_id": metadata_variables,
            "start_time": it_start,
            "end_time":   it_end,
        })


    elif task_id == "L3_8":
        # By how much does the mean of ocean volume-integrated {depth_integrated_inventory} within the {qualified_area} in the {implicit_region} change between {time_period_1} and {time_period_2}?
        depth_integrated_inventory_dict = sample_depth_integrated_inventory()
        qualified_area, _ = sample_qualified_area()
        implicit_region_code, implicit_region = sample_implicit_region()

        while True:
            start_month_1, end_month_1, time_period_1 = sample_time_period(experiment=experiment_id, max_months=120)
            start_month_2, end_month_2, time_period_2 = sample_time_period(experiment=experiment_id, max_months=120)
            if time_period_1 != time_period_2:
                break

        task_metadata = {
            "depth_integrated_inventory": depth_integrated_inventory_dict["derived_nl"],
            "qualified_area": qualified_area,
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "start_month_1": start_month_1,
            "end_month_1": end_month_1,
            "time_period_1": time_period_1,
            "start_month_2": start_month_2,
            "end_month_2": end_month_2,
            "time_period_2": time_period_2,
        }
        task_metadata.update({"ambiguous_terms": [
            "mean", depth_integrated_inventory_dict["derived_nl"], qualified_area, implicit_region, "change"
        ]})
        metadata_variables = depth_integrated_inventory_dict["base_codes"]
        data_metadata.update({"variable_id": metadata_variables})


    elif task_id == "L3_9":
        # What is the change in the mean volume fraction of {implicit_surface_depth} waters classified as {descriptive_noun} in the {region} over {time_period_1} relative to {time_period_2}?
        implicit_surface_depth = sample_implicit_depth(surface=True)
        descriptive_noun = sample_descriptive_noun()
        region_code, region = sample_region()

        while True:
            start_month_1, end_month_1, time_period_1 = sample_time_period(experiment=experiment_id, min_months=36, max_months=300)
            start_month_2, end_month_2, time_period_2 = sample_time_period(experiment=experiment_id, min_months=36, max_months=300)
            if time_period_1 != time_period_2:
                break

        criteria = resolve_descriptive_noun(descriptive_noun)

        task_metadata = {
            "implicit_surface_depth": implicit_surface_depth,
            "descriptive_noun": descriptive_noun,
            "region_code": region_code,
            "region": region,
            "start_month_1": start_month_1,
            "end_month_1": end_month_1,
            "time_period_1": time_period_1,
            "start_month_2": start_month_2,
            "end_month_2": end_month_2,
            "time_period_2": time_period_2,
        }
        task_metadata.update({"ambiguous_terms": [
            "change", "mean", implicit_surface_depth, descriptive_noun
        ]})
        data_metadata.update({"variable_id": criteria["variables"]})


    elif task_id == "L3_10":
        # In the {region}, is the inter-model spread in {implicit_depth} {variable} climatology significant during the {time_period}, after vertically averaging each model and regridding all models to a common 1° grid?
        region_code, region = sample_region()
        implicit_depth = sample_implicit_depth()
        var_code, var_nl = sample_variable_with_depth()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=36, max_months=300)

        task_metadata = {
            "region_code": region_code,
            "region": region,
            "implicit_depth": implicit_depth,
            "variable_code": var_code,
            "variable": var_nl,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "inter-model", "spread", implicit_depth, "climatology", "significant", "average", "regrid"
        ]})
        data_metadata.update({"variable_id": [var_code], "start_month": start_month, "end_month": end_month})


    elif task_id == "L3_11":
        # What is the inter-model standard deviation of the {implicit_depth} {variable} anomaly in the {region} in the recent period relative to the pre-industrial climatology?
        implicit_depth = sample_implicit_depth()
        var_code, var_nl = sample_variable_with_depth()
        region_code, region = sample_region()

        task_metadata = {
            "implicit_depth": implicit_depth,
            "variable_code": var_code,
            "variable": var_nl,
            "region_code": region_code,
            "region": region,
        }
        task_metadata.update({"ambiguous_terms": [
            "inter-model", implicit_depth, "anomaly", "recent", "pre-industrial", "climatology"
        ]})
        # Solver hardcodes RECENT 1995-2014 and PI 1850-1900 — both fully
        # inside historical, regardless of the randomly-sampled experiment_id.
        data_metadata.update({
            "experiment_id": "historical",
            "variable_id": [var_code],
            "start_time": "1850-01",
            "end_time":   "2014-12",
        })


    elif task_id == "L3_12":
        # Has the {term_for_depth_layer} depth in the {region} shifted deeper at a statistically significant rate from {time_period}?
        layer_var_codes, term_for_depth_layer = sample_term_for_depth_layer()
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=300)

        task_metadata = {
            "term_for_depth_layer": term_for_depth_layer,
            "region_code": region_code,
            "region": region,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            term_for_depth_layer, "statistically significant"
        ]})
        data_metadata.update({"variable_id": ["thetao", "so"], "start_month": start_month, "end_month": end_month})


    elif task_id == "L3_13":
        # What is the multi-model ensemble mean change in average {implicit_derived_variable} in the {implicit_region} in the {implicit_depth} between {time_period_1} and {time_period_2}?
        implicit_derived_dict = sample_implicit_derived_variable()
        implicit_region_code, implicit_region = sample_implicit_region()
        implicit_depth = sample_implicit_depth()

        while True:
            start_month_1, end_month_1, time_period_1 = sample_time_period(experiment=experiment_id, max_months=120)
            start_month_2, end_month_2, time_period_2 = sample_time_period(experiment=experiment_id, max_months=120)
            if time_period_1 != time_period_2:
                break

        task_metadata = {
            "implicit_derived_variable": implicit_derived_dict["derived_nl"],
            "implicit_region_code": implicit_region_code,
            "implicit_region": implicit_region,
            "implicit_depth": implicit_depth,
            "start_month_1": start_month_1,
            "end_month_1": end_month_1,
            "time_period_1": time_period_1,
            "start_month_2": start_month_2,
            "end_month_2": end_month_2,
            "time_period_2": time_period_2,
        }
        task_metadata.update({"ambiguous_terms": [
            "multi-model ensemble", "change", "average", implicit_derived_dict["derived_nl"], implicit_region, implicit_depth
        ]})
        data_metadata.update({"variable_id": implicit_derived_dict["base_codes"]})


    elif task_id == "L3_14":
        # What is the lag in months at which the correlation between the {correlated_derived_variable_1} and the {correlated_derived_variable_2} which are both restricted to {region} and {depth_range} is maximized over {time_period}?
        corr_var_dict = sample_correlated_derived_variable_pair()
        primary_var = corr_var_dict["var1_base_codes"][0]
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, primary_var)
        if dep_bnds is None: return None
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=300)
        region_code, region = sample_region()
        lower, upper, depth_range = sample_depth_range(dep_bnds=dep_bnds)

        task_metadata = {
            "correlated_derived_variable_1": corr_var_dict["var1_derived_nl"],
            "correlated_derived_variable_2": corr_var_dict["var2_derived_nl"],
            "region_code": region_code,
            "region": region,
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth_range,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        task_metadata.update({"ambiguous_terms": [
            "lag", "correlation", corr_var_dict["var1_derived_nl"], corr_var_dict["var2_derived_nl"],
        ]})
        data_metadata.update({"variable_id": list(set(
            corr_var_dict["var1_base_codes"]
            + corr_var_dict["var2_base_codes"]
        )), "start_month": start_month, "end_month": end_month})


    else:
        raise ValueError(f"Unknown task_id: '{task_id}'. Must be one of L3_1 to L3_14.")


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multiple fill values.*", category=xr.SerializationWarning)
        warnings.filterwarnings("ignore", message=".*All-NaN slice.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*", category=RuntimeWarning)
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