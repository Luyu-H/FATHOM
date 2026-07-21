from __future__ import annotations

import random
import math
import warnings
import xarray as xr

from src.utils.random_sampling import *
from src.utils.data_io_and_subset import (
    load_depth_levels,
    get_region_spec,
    load_variable_from_meta,
    load_area_weights,
    subset_time,
    subset_region,
    subset_depth,
    subset_data,
)
from src.utils.numerical_diagnostics import (
    compute_basic_statistic,
    compute_basic_statistic_with_area,
    compute_inventory,
    compute_vertical_gradient,
    compute_linear_trend,
    compute_trend_significance,
    evaluate_condition_or_compare,
    parse_threshold,
    area_weighted_mean,
    annual_area_weighted_mean,
    compute_pearson_correlation,
)
from src.utils.domain_knowledge import get_variable_unit


# ===================================================================
# Per-task solvers
# ===================================================================

def _solve_L1_1(data_meta, task_meta):
    """{stat_op} of {variable} at {depth_range} over {region} during {time_period}."""
    data = load_variable_from_meta(data_meta)
    data = subset_data(data, task_meta)
    area = subset_region(load_area_weights(data_meta), get_region_spec(task_meta["region_code"]))
    result = compute_basic_statistic_with_area(data, task_meta["statistical_operator_code"], area=area)
    return {"type": "scalar", "value": float(result.values), "unit": get_variable_unit(task_meta["variable_code"]),}


def _solve_L1_2(data_meta, task_meta):
    """Vertically integrated {variable} inventory."""
    data = load_variable_from_meta(data_meta)
    data = subset_data(data, task_meta)

    area = load_area_weights(data_meta)
    area = subset_region(area, get_region_spec(task_meta["region_code"]))

    result = compute_inventory(data, area=area)
    # average over time
    result = result.mean(dim="time", skipna=True)

    # unit: [conc] × m(depth) × m²(area) → base quantity only
    # e.g. "mol m-3" → "mol",  "kg m-3" → "kg"
    base_unit = get_variable_unit(task_meta["variable_code"])
    quantity = base_unit.strip().split()[0] if base_unit.strip() else ""

    return {"type": "scalar", "value": float(result.mean().values), "unit": quantity,}
 

def _solve_L1_3(data_meta, task_meta):
    """{stat_op} vertical gradient of {variable}."""
    data = load_variable_from_meta(data_meta)
    data = subset_data(data, task_meta)
    grad = compute_vertical_gradient(data)
    area = subset_region(load_area_weights(data_meta), get_region_spec(task_meta["region_code"]))
    result = compute_basic_statistic_with_area(grad, task_meta["statistical_operator_code"], area=area)
    return {"type": "scalar", "value": float(result.values), 
            "unit": f"{get_variable_unit(task_meta['variable_code'])} per meter",}
 

def _solve_L1_4(data_meta, task_meta):
    """Is linear trend in area-weighted mean significant at p < α?"""
    data = load_variable_from_meta(data_meta)
    data = subset_data(data, task_meta)

    if "depth" in data.dims:
        data = data.mean(dim="depth", skipna=True)
 
    area = load_area_weights(data_meta)
    area = subset_region(area, get_region_spec(task_meta["region_code"]))
 
    ts = area_weighted_mean(data, area)
    sig = compute_trend_significance(ts, alpha=task_meta["significance_level"])
    return {
        "type": "boolean",
        "value": bool(sig["is_significant"].values),
        "slope": float(sig["slope"].values),
        "slope_unit": f"{get_variable_unit(task_meta['variable_code'])} per month",
        "p_value": float(sig["p_value"].values),
    }
 

def _solve_L1_5(data_meta, task_meta):
    """Linear trend in annual mean global {derived_variable} (inventory)."""
    base = load_variable_from_meta(data_meta)
    base = subset_time(base, (task_meta["start_month"], task_meta["end_month"]))
    base = subset_depth(base, (task_meta["depth_lower"], task_meta["depth_upper"]))
 
    # All derived variables from sample_base_and_derived_variable() are inventories
    area = load_area_weights(data_meta)
    inventory = compute_inventory(base, area=area)

    annual_ts = inventory.resample(time="YE").mean()
    trend = compute_linear_trend(annual_ts)
    
    base_unit = get_variable_unit(task_meta["base_variable_code"])
    quantity = base_unit.strip().split()[0] if base_unit.strip() else ""
    return {"type": "scalar", "value": float(trend.values), "unit": f"{quantity} per year"}
 

def _solve_L1_6(data_meta, task_meta):
    """{stat_op} of global ocean {derived_variable} (inventory)."""
    base = load_variable_from_meta(data_meta)
    base = subset_time(base, (task_meta["start_month"], task_meta["end_month"]))
    
    area = load_area_weights(data_meta)
    inventory = compute_inventory(base, area=area)
    result = compute_basic_statistic(inventory, task_meta["statistical_operator_code"], dims="time")
    
    base_unit = get_variable_unit(task_meta["base_variable_code"])
    quantity = base_unit.strip().split()[0] if base_unit.strip() else ""
    return {"type": "scalar", "value": float(result.values), "unit": quantity,}
 

def _solve_L1_7(data_meta, task_meta):
    """Is linear trend in annual global area-weighted {variable} positive?"""
    data = load_variable_from_meta(data_meta)
    data = subset_time(data, (task_meta["start_month"], task_meta["end_month"]))

    if "depth" in data.dims:
        data = data.mean(dim="depth", skipna=True)
 
    area = load_area_weights(data_meta)
    annual_ts = annual_area_weighted_mean(data, area)
    slope = float(compute_linear_trend(annual_ts).values)
    return {"type": "boolean", "value": slope > 0, "slope": slope, "slope_unit": f"{get_variable_unit(task_meta['variable_code'])} per year",}
 

def _solve_L1_8(data_meta, task_meta):
    """Compare {stat} of {var1} vs {var2} across two region/time combos."""
    depth_val = task_meta["depth_code"]  # numeric from sample_depth()
 
    def _one_side(var_code, region_code, start, end):
        dm = {**data_meta, "variable_id": var_code, "start_time": start, "end_time": end}
        da = load_variable_from_meta(dm, variable_id=var_code)
        da = subset_time(da, (start, end))
        da = subset_region(da, get_region_spec(region_code))
        if "depth" in da.dims:
            da = da.sel(depth=depth_val, method="nearest")
        area = subset_region(load_area_weights(data_meta), get_region_spec(region_code))
        return compute_basic_statistic_with_area(da, task_meta["statistical_code"], area=area)
 
    lhs = _one_side(task_meta["variable_1_code"], task_meta["region_1_code"],
                     task_meta["start_month_1"],   task_meta["end_month_1"])
    rhs = _one_side(task_meta["variable_2_code"], task_meta["region_2_code"],
                     task_meta["start_month_2"],   task_meta["end_month_2"])
 
    lv, rv = float(lhs.values), float(rhs.values)
    result = evaluate_condition_or_compare(lv, task_meta["comparison_operator_code"], rv)
    return {"type": "boolean", "value": bool(result), "lhs": lv, "rhs": rv, "lhs_unit": get_variable_unit(task_meta["variable_1_code"]), "rhs_unit": get_variable_unit(task_meta["variable_2_code"]),}
 

def _solve_L1_9(data_meta, task_meta):
    """{stat} of {variable} where spatial condition holds."""
    # ---- main variable ----
    data = load_variable_from_meta(data_meta)
    data = subset_time(data, (task_meta["start_month"], task_meta["end_month"]))
    data = subset_depth(data, (task_meta["depth_lower"], task_meta["depth_upper"]))
 
    # ---- condition variable → reduce to 2-D (lat, lon) spatial mask ----
    cond_code = task_meta["condition_variable_code"]
    cond_dm = {**data_meta, "variable_id": cond_code}
    cond = load_variable_from_meta(cond_dm, variable_id=cond_code)
    cond = subset_time(cond, (task_meta["start_month"], task_meta["end_month"]))
 
    # Reduce over time *and* depth so the mask is purely spatial
    reduce_dims = ["time"]
    if "depth" in cond.dims:
        reduce_dims.append("depth")
    cond_reduced = compute_basic_statistic(
        cond, task_meta["condition_statistical_code"], dims=reduce_dims,
    )
 
    # sample_threshold() returns strings like "28 °C" — extract numeric value
    threshold_value = parse_threshold(task_meta["condition_threshold"])
    mask = evaluate_condition_or_compare(
        cond_reduced, task_meta["condition_comparison_operator_code"], threshold_value,
    )
    
    area = load_area_weights(data_meta)
    result = compute_basic_statistic_with_area(
        data.where(mask), task_meta["statistical_code"], area=area,
    )
    return {"type": "scalar", "value": float(result.values), 
            "unit": get_variable_unit(task_meta["variable_code"]),}
 

def _solve_L1_10(data_meta, task_meta):
    """Is {stat} of {variable} over {region} {cmp} {threshold}?"""
    data = load_variable_from_meta(data_meta)
    data = subset_data(data, task_meta, do_depth=False)
    
    area = subset_region(load_area_weights(data_meta), get_region_spec(task_meta["region_code"]))
    lv = float(compute_basic_statistic_with_area(data, task_meta["statistical_operator_code"], area=area).values)
    threshold_value = parse_threshold(task_meta["threshold"])
    result = evaluate_condition_or_compare(
        lv, task_meta["comparison_operator_code"], threshold_value,
    )
    return {"type": "boolean", "value": bool(result), 
            "statistic_value": lv, "statistic_unit": get_variable_unit(task_meta["variable_code"]), 
            "threshold_value": threshold_value, "threshold_unit": get_variable_unit(task_meta["variable_code"]),}
 

def _solve_L1_11(data_meta, task_meta):
    """Pearson r between annual area-weighted mean {var1} and {var2}."""
    region_spec = get_region_spec(task_meta["region_code"])
    area = subset_region(load_area_weights(data_meta), region_spec)

    if any(area.sizes[d] == 0 for d in area.dims):
        return {"type": "scalar", "value": float("nan"), "p_value": float("nan")}
 
    def _annual_ts(var_code):
        dm = {**data_meta, "variable_id": var_code,
              "start_time": task_meta["start_month"], "end_time": task_meta["end_month"]}
        da = load_variable_from_meta(dm, variable_id=var_code)
        da = subset_time(da, (task_meta["start_month"], task_meta["end_month"]))
        da = subset_region(da, region_spec)
        if "depth" in da.dims:
            da = da.mean(dim="depth", skipna=True)
        return annual_area_weighted_mean(da, area)
 
    ts1 = _annual_ts(task_meta["variable_1_code"])
    ts2 = _annual_ts(task_meta["variable_2_code"])
 
    corr = compute_pearson_correlation(ts1, ts2)
    return {"type": "scalar", "value": corr["r"], "p_value": corr["p_value"]}
 

def _solve_L1_12(data_meta, task_meta):
    """Difference between {stat_op} of {var1} at topmost and {stat_op} of {var2} at bottommost,
    area-weighted average over {region} during {time_period}."""
    region_spec = get_region_spec(task_meta["region_code"])
    stat_code = task_meta["statistical_operator_code"]

    def _load_and_stat(var_code, position):
        dm = {**data_meta, "variable_id": var_code,
              "start_time": task_meta["start_month"], "end_time": task_meta["end_month"]}
        da = load_variable_from_meta(dm, variable_id=var_code)
        da = subset_time(da, (task_meta["start_month"], task_meta["end_month"]))
        da = subset_region(da, region_spec)
        if "depth" in da.dims:
            da = da.isel(depth=0) if position == "surface" else da.isel(depth=-1)
        # dims=None → reduce all dims → scalar
        area = subset_region(load_area_weights(data_meta), region_spec)
        return float(compute_basic_statistic_with_area(da, stat_code, area=area).values)
    
    top_val = _load_and_stat(task_meta["variable_1_code"], "surface")
    bot_val = _load_and_stat(task_meta["variable_2_code"], "floor")

    unit_1 = get_variable_unit(task_meta["variable_1_code"])
    unit_2 = get_variable_unit(task_meta["variable_2_code"])
    return {
        "type": "scalar",
        "value": top_val - bot_val,
        "lhs": top_val,
        "rhs": bot_val,
        "lhs_unit": unit_1,
        "rhs_unit": unit_2,
    }


# ===================================================================
# Main entry point
# ===================================================================

_SOLVERS = {
    "L1_1":  _solve_L1_1,   "L1_2":  _solve_L1_2,
    "L1_3":  _solve_L1_3,   "L1_4":  _solve_L1_4,
    "L1_5":  _solve_L1_5,   "L1_6":  _solve_L1_6,
    "L1_7":  _solve_L1_7,   "L1_8":  _solve_L1_8,
    "L1_9":  _solve_L1_9,   "L1_10": _solve_L1_10,
    "L1_11": _solve_L1_11,  "L1_12": _solve_L1_12,
}


def generate_level1_tasks(config, task_id):
    """
    Generate level 1 tasks based on the task ID.

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


    if task_id == "L1_1":
        # What is the {statistical_operator} of {variable} at {depth_range} over the {region} during {time_period}?
        stat_code, stat_op = sample_statistical_operator()
        var_code, var_nl = sample_variable_with_depth()
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_code)
        if dep_bnds is None: return None
        lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds)
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        task_metadata = {
            "statistical_operator_code": stat_code,
            "statistical_operator": stat_op,
            "variable_code": var_code,
            "variable": var_nl,

            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "region_code": region_code,
            "region": region,
            
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})


    elif task_id == "L1_2":
        # What is the vertically integrated {variable} inventory from {depth_range} over the {region} during {time_period}?

        vertical_integrated_vars = [
            'dissic', 'dissoc', 'talk', 'chl', 
            'no3', 'po4', 'si', 'dfe', 'o2',
        ]
        while True:
            var_code, var_nl = sample_variable()
            if var_code in vertical_integrated_vars:
                break
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_code)
        if dep_bnds is None: return None

        if var_code in ["chl"]:
            lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds, max_depth=500)  # for chlorophyll and phytoplankton, limit to upper 500 m
        else:
            lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds)

        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        task_metadata = {
            "variable_code": var_code,
            "variable": var_nl,
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "region_code": region_code,
            "region": region,

            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})


    elif task_id == "L1_3":
        # What is the {statistical_operator} vertical gradient of {variable} between {depth_range} over the {region} during {time_period}?
        vertical_gradient_vars = {
            "thetao", "so",
            "o2",
            "no3", "po4", "si", "dfe",
            "dissic", "talk", "ph",
            "chl",
        }
        
        while True:
            var_code, var_nl = sample_variable()
            if var_code in vertical_gradient_vars:
                break
            
        stat_code, stat_op = sample_statistical_operator()
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_code)
        if dep_bnds is None: return None
        lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds)
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        task_metadata = {
            "statistical_operator_code": stat_code, 
            "statistical_operator": stat_op,
            "variable_code": var_code,
            "variable": var_nl,
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "region_code": region_code,
            "region": region,

            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})


    elif task_id == "L1_4":
        # Is the linear trend in the area-weighted mean {variable} over the {depth_range} of the {region} during {time_period} statistically significant at p<{significance_level}?
        var_code, var_nl = sample_variable_with_depth()
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_code)
        if dep_bnds is None: return None
        lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds)
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=240)
        significance_level = sample_significance_level()

        task_metadata = {
            "variable_code": var_code,
            "variable": var_nl,
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "region_code": region_code,
            "region": region,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
            "significance_level": significance_level,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})


    elif task_id == "L1_5":
        # What is the linear trend in the global annual {depth_range} {derived_variable} computed from {base_variable} over {time_period}?
        var_dict = sample_base_and_derived_variable()
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_dict["base_var_code"])
        if dep_bnds is None: return None
        lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds)
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=240)  

        task_metadata = {
            "base_variable_code": var_dict["base_var_code"],
            "base_variable": var_dict["base_var_nl"],
            "derived_variable_code": var_dict["derived_var_code"],
            "derived_variable": var_dict["derived_var_nl"],
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({"variable_id": var_dict["base_var_code"],
                              "start_time": start_month, "end_time": end_month})
        

    elif task_id == "L1_6":
        # What is the {statistical_operator} of global ocean {derived_variable} during {time_period}?
        stat_code, stat_op = sample_statistical_operator()
        var_dict = sample_base_and_derived_variable()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        task_metadata = {
            "statistical_operator_code": stat_code,
            "statistical_operator": stat_op,
            "base_variable_code": var_dict["base_var_code"],
            "base_variable": var_dict["base_var_nl"],
            "derived_variable_code": var_dict["derived_var_code"],
            "derived_variable": var_dict["derived_var_nl"],
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({"variable_id": var_dict["base_var_code"],
                              "start_time": start_month, "end_time": end_month})
        

    elif task_id == "L1_7":
        # Is the linear trend in the annual global area-weighted mean of {variable} over {time_period} positive?
        var_code, var_nl = sample_variable()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=60, max_months=240)  

        task_metadata = {
            "variable_code": var_code,
            "variable": var_nl,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})


    elif task_id == "L1_8":
        # Is the {statistical_operator} of {variable_1} over the {region_1} during {time_period_1} at {depth} {comparison_operator} the {statistical_operator} of {variable_2} over the {region_2} during {time_period_2} at {depth}?
        stat_code, stat_op = sample_statistical_operator()
        var_dict = sample_two_variables_for_comparison()
        region1_code, region1 = sample_region()
        region2_code, region2 = sample_region()
        start_month1, end_month1, time_period1 = sample_time_period(experiment=experiment_id, max_months=120)
        start_month2, end_month2, time_period2 = sample_time_period(experiment=experiment_id, max_months=120)
        comparison_code, comparison_op = sample_comparison_operator()

        dep1 = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_dict["var1_code"])
        dep2 = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_dict["var2_code"])
        if dep1 is not None and dep2 is not None:
            common_max = min(float(max(dep1)), float(max(dep2)))
            candidates = [float(d) for d in dep1 if float(d) <= common_max]
            if not candidates: return None
            depth = random.choice(candidates)
            depth_nl = f"{depth} m"
        elif dep1 is not None:
            depth = float(random.choice(dep1))
            depth_nl = f"{depth} m"
        else:
            depth, depth_nl = sample_depth()

        task_metadata = {
            "statistical_code": stat_code,
            "statistical_operator": stat_op,

            "variable_1_code": var_dict["var1_code"],
            "variable_1": var_dict["var1_nl"],
            "variable_2_code": var_dict["var2_code"],
            "variable_2": var_dict["var2_nl"],

            "region_1_code": region1_code,
            "region_1": region1,
            "region_2_code": region2_code,
            "region_2": region2,

            "start_month_1": start_month1,
            "end_month_1": end_month1,
            "time_period_1": time_period1,
            "start_month_2": start_month2,
            "end_month_2": end_month2,
            "time_period_2": time_period2,
            "comparison_operator_code": comparison_code,
            "comparison_operator": comparison_op,
            "depth_code": depth,
            "depth": depth_nl,
        }
        data_metadata.update({
            "variable_id": var_dict["var1_code"],
            "variable_id_2": var_dict["var2_code"],
            "start_time": min(start_month1, start_month2),
            "end_time": max(end_month1, end_month2),
        })
        

    elif task_id == "L1_9":
        # What is the {statistical_operator} {variable} at {depth_range} over regions where {condition} during {time_period}?
        stat_code, stat_op = sample_statistical_operator()
        var_code, var_nl = sample_variable_with_depth()
        dep_bnds = load_depth_levels(config.cmip6.root, experiment_id, source_id, var_code)
        if dep_bnds is None: return None
        lower, upper, depth = sample_depth_range(dep_bnds=dep_bnds)
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        while True:
            cond_statistical_code, cond_statistical_op = sample_statistical_operator()
            if cond_statistical_op not in {"standard deviation", "variance", "range"}:
                break
        
        cond_var_code, cond_var_nl = sample_variable()
        cond_comparison_code, cond_comparison_op = sample_comparison_operator()
        cond_threshold = sample_threshold(cond_var_code)
        condition = f"the {cond_statistical_op} of {cond_var_nl} {cond_comparison_op} {cond_threshold}"

        task_metadata = {
            "statistical_code": stat_code,
            "statistical_operator": stat_op,
            "variable_code": var_code,
            "variable": var_nl,
            "depth_lower": lower,
            "depth_upper": upper,
            "depth_range": depth,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
            "condition": condition,

            "condition_statistical_code": cond_statistical_code,
            "condition_statistical_op": cond_statistical_op, 
            "condition_variable_code": cond_var_code,
            "condition_variable": cond_var_nl,
            "condition_comparison_operator_code": cond_comparison_code,
            "condition_comparison_operator": cond_comparison_op,
            "condition_threshold": cond_threshold,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})
        
        
    elif task_id == "L1_10":
        # Is the {statistical_operator} of {variable} over the {region} during {time_period} {comparison_operator} {threshold}?
        while True:
            cond_statistical_code, cond_statistical_op = sample_statistical_operator()
            if cond_statistical_op not in {"standard deviation", "variance", "range"}:
                break
        var_code, var_nl = sample_variable()
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)
        comparison_code, comparison_op = sample_comparison_operator()
        threshold = sample_threshold(var_code)

        task_metadata = {
            "statistical_operator_code": cond_statistical_code,
            "statistical_operator": cond_statistical_op,
            "variable_code": var_code,
            "variable": var_nl,
            "region_code": region_code,
            "region": region,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
            "comparison_operator_code": comparison_code,
            "comparison_operator": comparison_op,
            "threshold": threshold,
        }
        data_metadata.update({"variable_id": var_code, "start_time": start_month, "end_time": end_month})


    elif task_id == "L1_11":
        # What is the Pearson correlation coefficient between annual area-weighted mean {variable_1} and annual area-weighted mean {variable_2} over the {region} during {time_period}?
        var_dict = sample_two_variables_for_correlation()
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, min_months=36, max_months=180)

        task_metadata = {
            "variable_1_code": var_dict["var1_code"],
            "variable_1": var_dict["var1_nl"],
            "variable_2_code": var_dict["var2_code"],
            "variable_2": var_dict["var2_nl"],
            "region_code": region_code,
            "region": region,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({
            "variable_id": var_dict["var1_code"],
            "variable_id_2": var_dict["var2_code"],
            "start_time": start_month,
            "end_time": end_month,
        })
        

    elif task_id == "L1_12":
        # What is the difference between the {statistical_operator} of {variable_1} in the topmost layer and the {statistical_operator} of {variable_2} in the  bottommost layer, both limited over the {region} during {time_period}?
        stat_code, stat_op = sample_statistical_operator()
        var_dict = sample_two_variables_for_comparison()
        region_code, region = sample_region()
        start_month, end_month, time_period = sample_time_period(experiment=experiment_id, max_months=120)

        task_metadata = {
            "statistical_operator_code": stat_code,
            "statistical_operator": stat_op,
            
            "variable_1_code": var_dict["var1_code"],
            "variable_1": var_dict["var1_nl"],
            "variable_2_code": var_dict["var2_code"],
            "variable_2": var_dict["var2_nl"],

            "region_code": region_code,
            "region": region,
            "start_month": start_month,
            "end_month": end_month,
            "time_period": time_period,
        }
        data_metadata.update({
            "variable_id": var_dict["var1_code"],
            "variable_id_2": var_dict["var2_code"],
            "start_time": start_month,
            "end_time": end_month,
        })
        

    else:
        raise ValueError(f"Unknown task_id: '{task_id}'. Must be one of L1_1 to L1_12.")


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multiple fill values.*", category=xr.SerializationWarning)
        warnings.filterwarnings("ignore", message=".*All-NaN slice.*", category=RuntimeWarning)
        answer_metadata = _SOLVERS[task_id](data_metadata, task_metadata)

    # reject tasks whose answer is NaN / contains NaN
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